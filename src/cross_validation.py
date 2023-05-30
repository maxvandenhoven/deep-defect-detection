import pathlib

from ksuid import ksuid
from torchvision import datasets

from src.data import stratified_kfold_generator
from src.logging import Logger
from src.model import WFDefectDetector
from src.training import train


def kfold_train(
    dataset: datasets.ImageFolder,
    n_splits: int = 5,
    shuffle: bool = True,
    batch_size: int = 16,
    random_state: int = 42,
    kernel_size: int = 3,
    base_channels: int = 16,
    num_conv_blocks: int = 2,
    num_fc_features: int = 64,
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    max_epochs: int = 25,
    patience: int = 3,
    device: str = "cuda",
) -> float:
    """Use K-fold cross-validation to evaluate the performance of a set of
        hyperparameters. Performance is calculated by averaging the best validation
        losses of the generated (non-overlapping and stratified) folds, giving the
        most accurate metric of model performance, as early stopping and checkpointing
        is used to save the model with the best performance in terms of validation loss.
        We choose to track validation loss instead of validation accuracy, as this is
        most common in practice and a more fine-grained measure given the relatively
        small dataset.

    Args:
        dataset (datasets.ImageFolder): full (training) dataset to generate fold from.
        n_splits (int, optional): number of folds. Defaults to 5.
        shuffle (bool, optional): whether to shuffle data in folds. Defaults to True.
        batch_size (int, optional): batch size of dataloaders. Defaults to 16.
        random_state (int, optional): random state for reproducibility. Defaults to 42.
        kernel_size (int, optional): kernel size of all convolutional layers in the
            model. Must be in [3, 5, 7]. Defaults to 3.
        base_channels (int, optional): number of filters in the first conv layer, which
            is doubled in subsequent conv layers. Must be in [8, 16, 32] Defaults to 16.
        num_conv_blocks (int, optional): number of conv-act-pool blocks in the model.
            Must be in [2, 3]. Defaults to 2.
        num_fc_features (int, optional): number of neurons in the dense layer right after
            flattening. Defaults to 64.
        activation (str, optional): activation function used everywhere. Must be in
            ["tanh", "relu", "leakyrelu"]. Defaults to "relu".
        optimizer (str, optional): optimizer used for training with default paraeters,
            except for learning rate. Must be in ["sgd", "rmsprop", "adam"]. Defaults to
            "adam".
        learning_rate (float, optional): learning rate. Defaults to 0.001.
        max_epochs (int, optional): maximum number of epochs to train. Defaults to 25.
        patience (int, optional): within how many epochs teh validation loss must
            improve before terminating optimization. Defaults to 3.
        device (str, optional): accelerator device to use. Defaults to "cuda".

    Returns:
        float: best validation loss after training.
    """
    # Generate random ID for the trial. KSUIDs are used so that trials are sorted by
    # creation time automatically in file explorers
    trial_id = ksuid()
    print(f"Trial ID: {trial_id}")

    # Create fold generator to output sets of training and valid dataloaders corresponding
    # to folds obtained from stratified splitting on the image targets
    fold_generator = stratified_kfold_generator(
        dataset=dataset,
        n_splits=n_splits,
        shuffle=shuffle,
        batch_size=batch_size,
        random_state=random_state,
    )

    # Accumulate best validation loss of each fold to compute average best validation loss
    total_valid_loss = 0

    for fold_idx, train_dataloader, valid_dataloader in fold_generator:
        # Create folder to log fold run, such that all models are saved
        logger = Logger(pathlib.Path(f"runs_kfold/trial_{trial_id}/fold_{fold_idx}"))

        model = WFDefectDetector(
            kernel_size=kernel_size,
            base_channels=base_channels,
            num_conv_blocks=num_conv_blocks,
            num_fc_features=num_fc_features,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )

        valid_loss_min = train(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            logger=logger,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )

        print(f"  Fold {fold_idx} best valid loss: {valid_loss_min}")
        total_valid_loss += valid_loss_min

    # Compute average best validation loss by averaging best validation losses of folds
    average_valid_loss = total_valid_loss / n_splits
    print(f"  Average best valid loss: {average_valid_loss}")
    print()

    return average_valid_loss
