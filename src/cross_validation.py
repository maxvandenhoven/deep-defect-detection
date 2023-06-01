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
    max_epochs: int = 25,
    patience: int = 3,
    device: str = "cuda",
    log_folder: pathlib.Path = None,
    trial_id: str = None,
    **model_kwargs,
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
        max_epochs (int, optional): maximum number of epochs to train. Defaults to 25.
        patience (int, optional): within how many epochs the validation loss must
            improve before terminating optimization. Defaults to 3.
        device (str, optional): accelerator device to use. Defaults to "cuda".
        log_folder (pathlib.Path, optional): overwrite default random log folder.
        **model_kwargs: keyword arguments passed to the model

    Returns:
        float: best validation loss after training.
    """
    # Generate random ID for the trial. KSUIDs are used so that trials are sorted by
    # creation time automatically in file explorers
    if trial_id is None:
        trial_id = f"trial_{ksuid()}"
    print(f"Trial ID: {trial_id}")

    # Determine folder to log model to. Note that trial ID does not necessarily need to
    # correspond with the logging folder
    if log_folder is None:
        log_folder = "runs_kfold"
    base_log_folder = pathlib.Path(f"{log_folder}/{trial_id}")

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
        logger = Logger(base_log_folder / f"fold_{fold_idx}")

        model = WFDefectDetector(**model_kwargs)

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
