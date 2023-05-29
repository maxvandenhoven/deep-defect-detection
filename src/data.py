from typing import Generator, Sequence

import torch
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torchvision import datasets, transforms


def load_dataset(
    filepath: str,
    resolution: tuple[int] = (224, 224),
    random_transform: list[nn.Module] = None,
) -> datasets.ImageFolder:
    """Build a TorchVision dataset from a folder of images. Images are first center-
        cropped on the WF part, similar to the provided support code. The image is then
        resized to the specified resolution, which is 224x224 by default, following
        common computer vision models/datasets. Random transforms can also be specified
        to improve accuracy with data augmentation.

        The expected folder structure is shown below, which leads to "ng" being class 0
        and "ok" being class 1. This does not matter for the model, but is important to
        keep in mind for evaluating the model.
        <filepath>
            ng
                photo_##.jpg
                photo_##.jpg
                ...
            ok
                photo_##.jpg
                photo_##.jpg
                ...

    Args:
        filepath (str): root folder of the dataset.
        resolution (list[int], optional): desired resolution. Defaults to [224, 224].
        random_trasnform (list[nn.Module]): list of TorchVision transforms that
            randomly augment image (e.g., random flip, rotate, and color augmentations).

    Returns:
        datasets.ImageFolder: TorchVision dataset (subclass of torch Dataset class).
    """
    # Used to avoid issues with setting an empty list as a default parameter value
    if random_transform is None:
        random_transform = []

    # First crop images on the part of interest, then resize to desired resolution.
    transform = transforms.Compose(
        [
            transforms.CenterCrop(size=(2400, 1200)),
            transforms.Resize(size=resolution),
            *random_transform,
            transforms.ToTensor(),
        ]
    )

    # Targets/labels are converted to floats for compatibility with model predictions
    target_transform = lambda target: torch.tensor(target, dtype=torch.float32)

    return datasets.ImageFolder(
        root=filepath, transform=transform, target_transform=target_transform
    )


def get_dataloader(
    dataset: datasets.ImageFolder,
    indices: Sequence[int] = None,
    shuffle: bool = True,
    batch_size: int = 16,
) -> data.DataLoader:
    """Build dataloader for subset (fold) of dataset.

    Args:
        dataset (datasets.ImageFolder): entire dataset.
        indices (Sequence[int], optional): indices of subset (fold) to make dataloader
            for. Defaults to None, returning the entire dataset as a dataloader.
        shuffle (bool, optional): whether to shuffle images. Defaults to True.
        batch_size (int, optional): batch size to use. Defaults to 64.

    Returns:
        data.DataLoader: dataloader corresponding to dataset subset.
    """
    # Allow indices to be omitted in order to convert entire dataset to dataloader
    # (useful for testing the final model on the test set   )
    if indices is not None:
        subset = data.Subset(dataset=dataset, indices=indices)
    else:
        subset = dataset

    return data.DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def stratified_kfold_generator(
    dataset: datasets.ImageFolder,
    n_splits: int = 5,
    shuffle: bool = True,
    batch_size: int = 16,
    random_state: int = 42,
) -> Generator[tuple[int, data.DataLoader, data.DataLoader], None, None]:
    """Build train and validation dataloaders using k-fold stratified splitting.

    Args:
        dataset (data.Dataset): dataset to split.
        n_splits (int, optional): number of folds. Defaults to 5.
        shuffle (bool, optional): whether to shuffle data. Defaults to True.
        batch_size (int, optional): batch size of dataloaders. Defaults to 16.
        random_state (int, optional): random state for reproducibility. Defaults to 42.

    Yields:
        Generator[tuple[int, data.DataLoader, data.DataLoader], None, None]: generator
            objects yielding fold index, train- and validation dataloaders for a fold.
    """
    kf = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)

    # Perform stratified splitting on the image targets to keep things
    # perfectly balanced, as all things should be
    fold_generator = kf.split(X=dataset, y=dataset.targets)

    for fold_idx, (train_indices, valid_indices) in enumerate(fold_generator):
        train_dataloader = get_dataloader(
            dataset=dataset,
            indices=train_indices,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        valid_dataloader = get_dataloader(
            dataset=dataset,
            indices=valid_indices,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        yield fold_idx, train_dataloader, valid_dataloader
