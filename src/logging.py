import json
import pathlib
import shutil

import torch

from src.model import CustomModuleType


class Logger:
    def __init__(self, log_folder: pathlib.Path) -> None:
        """Initialize logger object and create logging folder.

        Args:
            log_folder (pathlib.Path): output folder of logs and checkpoints.
        """
        self.log_folder = log_folder

        # Track logged values in a dictionary to simply log method
        self.logs: dict[str, list] = {}

        if self.log_folder.exists():
            shutil.rmtree(self.log_folder)
        self.log_folder.mkdir(parents=True)

    def log(self, label: str, value: float) -> None:
        """Log a numerical value.

        Args:
            label (str): name of value being logged.
            value (float | str): value being logged.
        """
        if label not in self.logs:
            self.logs[label] = []

        self.logs[label].append(value)

    def export_hyperparameters(self, model: CustomModuleType) -> None:
        """Save dictionary of model hyperparameters to file in logging folder.

        Args:
            model (CustomModuleType): model to save hyperparameters of. Must implement
                `hyperparameters` property.
        """
        with open(self.log_folder / "hyperparameters.json", "w+") as file:
            json.dump(model.hyperparameters, file, indent=4)

    def export_logs(self) -> None:
        """Save logs to file in logging folder."""
        with open(self.log_folder / "logs.json", "w+") as file:
            json.dump(self.logs, file, indent=4)

    def checkpoint(self, model: CustomModuleType) -> None:
        """Save model checkpoint to logging folder.

        Args:
            model (CustomModelType): model to save.
        """
        torch.save(model.state_dict(), self.log_folder / "checkpoint.pth")
