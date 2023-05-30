import json
import pathlib

import torch
import torch.utils.data as data

from src.model import WFDefectDetector


def evaluate_checkpoint(
    run_dir: pathlib.Path, test_dataloader: data.DataLoader
) -> None:
    """Load model checkpoint and evaluate it on the test set.

    Args:
        run_dir (pathlib.Path): folder of the run to test. Must contain checkpoint.pth
            and hyperparameters.json.
        test_dataloader (data.DataLoader): data to test on.
    """
    # Recreate model with the same hyperparameters used during training. This is needed
    # to ensure weights and parameters line up with the saved checkpoint
    with open(run_dir / "hyperparameters.json") as file:
        hyperparameters = json.load(file)
    model = WFDefectDetector(**hyperparameters)

    # Load in checkpoint
    model.load_state_dict(torch.load(run_dir / "checkpoint.pth"))

    # Put model in evaluation mode to disable certain layers typed correctly
    model.eval()

    # Initialize testing loss and accuracy accumulators
    total_loss = 0
    total_correct = 0
    num_batches = 0
    num_instances = 0

    # Run inference on test data without computing gradients
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            loss = model.criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.round() == labels).sum().item()
            num_batches += 1
            num_instances += len(labels)

    # Accumulate test loss and accuracy
    test_loss = total_loss / num_batches
    test_accuracy = total_correct / num_instances

    # Logging
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")
