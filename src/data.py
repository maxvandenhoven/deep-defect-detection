from typing import Sequence

import torch.utils.data as data
from torch import nn
from torchvision import datasets, transforms


def load_dataset(
    filepath: str,
    resolution: tuple[int] = (224, 224),
    random_transform: list[nn.Module] = None,
) -> datasets.VisionDataset:
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
        VisionDataset: TorchVision dataset (subclass of torch Dataset class).
    """
    if random_transform is None:
        random_transform = []

    transform = transforms.Compose(
        [
            transforms.CenterCrop(size=(2400, 1200)),
            transforms.Resize(size=resolution),
            *random_transform,
            transforms.ToTensor(),
        ]
    )

    return datasets.ImageFolder(root=filepath, transform=transform)


def get_fold_dataloader(
    dataset: data.Dataset,
    indices: Sequence[int],
    shuffle: bool = True,
    batch_size: int = 64,
) -> data.DataLoader:
    """Build dataloader for subset (fold) of dataset.

    Args:
        dataset (data.Dataset): entire dataset.
        indices (Sequence[int]): indices of subset (fold) to make dataloader for.
        shuffle (bool, optional): whether to shuffle images. Defaults to True.
        batch_size (int, optional): batch size to use. Defaults to 64.

    Returns:
        data.DataLoader: _description_
    """
    dataset_subset = data.Subset(dataset=dataset, indices=indices)
    return data.DataLoader(
        dataset=dataset_subset, batch_size=batch_size, shuffle=shuffle
    )
