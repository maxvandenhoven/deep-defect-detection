import time

import torch
import torch.utils.data as data

from src.logging import Logger
from src.model import CustomModuleType


def train(
    model: CustomModuleType,
    train_dataloader: data.DataLoader,
    valid_dataloader: data.DataLoader,
    logger: Logger,
    max_epochs: int = 25,
    patience: int = 3,
    device: str = "cuda",
) -> int:
    """Training loop that implements validation, logging, and checkpointing for a
        single model on a single set of training and validation datasets.

    Args:
        model (CustomModuleType): model to optimize.
        train_dataloader (data.DataLoader): training data.
        valid_dataloader (data.DataLoader): validation data.
        logger (Logger): logger object with set output folder.
        max_epochs (int, optional): maximum number of epochs to train. Defaults to 25.
        patience (int, optional): within how many epochs teh validation loss must
            improve before terminating optimization. Defaults to 3.
        device (str, optional): accelerator device to use. Defaults to "cuda".

    Returns:
        int: best validation loss over all epochs.
    """
    # Keep track of number of epochs without improvement for early stopping
    counter = 0
    valid_loss_min = torch.inf

    # Export model hyperparametrs and move the model to the desired accelerator device
    logger.export_hyperparameters(model)
    model.to(device)

    for epoch in range(max_epochs):
        epoch_start_time = time.time()

        # Set model in training mode to enable certain types of layers correctly
        model.train()
        train_start_time = time.time()

        # Initialize training loss and accuracy accumulators
        total_loss = 0
        total_correct = 0
        num_batches = 0
        num_instances = 0

        for images, labels in train_dataloader:
            # Put data on same device as model
            images = images.to(device)
            labels = labels.to(device)

            # Zero out gradients, run the forward pass of the batch, compute the loss,
            # compute the gradients, and perform an optimization step
            model.optimizer.zero_grad()
            outputs = model(images)
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # Accumulate training metrics
            total_loss += loss.item()
            total_correct += (outputs.round() == labels).sum().item()
            num_batches += 1
            num_instances += len(labels)

        # Logging
        train_end_time = time.time()
        logger.log("train_loss", total_loss / num_batches)
        logger.log("train_accuracy", total_correct / num_instances)
        logger.log("train_time", train_end_time - train_start_time)

        # Put model in evaluation mode to disable certain layers typed correctly
        model.eval()
        valid_start_time = time.time()

        # Initialize validation metric accumulators
        total_loss = 0
        total_correct = 0
        num_batches = 0
        num_instances = 0

        # Run inference on validation data without computing gradients
        with torch.no_grad():
            for images, labels in valid_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = model.criterion(outputs, labels)

                total_loss += loss.item()
                total_correct += (outputs.round() == labels).sum().item()
                num_batches += 1
                num_instances += len(labels)

        # Logging
        valid_end_time = time.time()
        valid_loss = total_loss / num_batches
        logger.log("valid_loss", valid_loss)
        logger.log("valid_accuracy", total_correct / num_instances)
        logger.log("valid_time", valid_end_time - valid_start_time)

        epoch_end_time = time.time()
        logger.log("epoch", epoch)
        logger.log("epoch_time", epoch_end_time - epoch_start_time)

        # Early stopping and checkpointing. Counter is increased if best validation
        # loss is not improved. Checkpoints are generated when a new best model is found
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            counter = 0
            logger.checkpoint(model)
        else:
            counter += 1

        if counter >= patience:
            break

        # Final logging
        logger.log("valid_loss_min", valid_loss_min)
        logger.export_logs()

    # Move model back to CPU for easier testing inference
    model.to("cpu")

    return valid_loss_min
