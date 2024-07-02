import numpy as np
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Tuple


def set_seeds(seed: int) -> None:
    """Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_mnist_dataloader(batch_size: int=64) -> Tuple[DataLoader, DataLoader]:
    """Prepare MNIST dataloader.

    Args:
        batch_size (int): Batch size for training and validation dataloader.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and Validation dataloader
    """
    # Transformations applied on each image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Mean and Std Deviation for MNIST
        ]
    )
    # Loading MNIST dataset
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)

    return train_loader, val_loader


def load_neps_checkpoint(
        previous_pipeline_directory: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> Tuple[int, nn.Module, torch.optim.Optimizer]:
    """Load checkpoint state to be used by NePS.
    
    Args:
        previous_pipeline_directory (Path): Directory where checkpoint is saved.
        model (nn.Module): Model to be loaded.
        optimizer (torch.optim.Optimizer): Optimizer to be loaded.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Scheduler to be loaded.

    Returns:
        Steps (int): Number of steps the model was trained for.
        model (nn.Module): Model with loaded state.
        optimizer (torch.optim.Optimizer): Optimizer with loaded state.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Scheduler with loaded state.
    """
    steps = None
    if previous_pipeline_directory is not None:
        checkpoint = torch.load(previous_pipeline_directory / "checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler = None
        if "steps" in checkpoint:
            steps = checkpoint["steps"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
    return steps, model, optimizer, scheduler


def save_neps_checkpoint(
    pipeline_directory: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save checkpoint state to be used by NePS.
    
    Args:
        pipeline_directory (Path): Directory where checkpoint is saved.
        epoch (int): Number of steps the model was trained for.
        model (nn.Module): Model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer to be saved.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Scheduler to be saved.
    """
    _save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "steps": epoch,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        _save_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(
        _save_dict,
        pipeline_directory / "checkpoint.pth",
    )
