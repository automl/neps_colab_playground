from pathlib import Path
import time
import torch
import torch.nn as nn
from typing import Union

from model import SimpleCNN
from utils import (
    get_optimizer,
    get_scheduler,
    load_neps_checkpoint,
    prepare_mnist_dataloader,
    save_neps_checkpoint,
    total_gradient_l2_norm,
    train_one_epoch,
    validate_model
)

from neps.plot.tensorboard_eval import tblogger


def training_pipeline(
    # neps parameters for load-save of checkpoints
    pipeline_directory: Union[Path, None] = None,
    previous_pipeline_directory: Union[Path, None] = None,
    # hyperparameters
    batch_size: int = 128,
    num_layers: int = 2,
    num_neurons: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    optimizer: str = "adamw",
    dropout: bool = True,
    # fidelity control
    epochs: int = 10,
    subsample: float = 1.0,
    # other parameters
    log_tensorboard: bool = False,
    verbose: bool = True,
    allow_checkpointing: bool = False,
):
    """Training pipeline for a simple CNN on MNIST dataset.

    This is a standard pipeline to train and validate models. 
    The only exclusive requirement to interface NePS are:
    * Arguments that pass hyperparameters
    * (Optional) Using tblogger to log tensorboard metrics supported by NePS
    * Returning a dictionary with keys "loss", "cost", and "info_dict"
        * "loss" must be a minimizing metric
    """
    # Load data
    _start = time.time()
    (
        train_loader,
        val_loader,
        (num_channels, image_height, image_width),
        num_classes
    ) = prepare_mnist_dataloader(batch_size=batch_size, subsample_fraction=subsample)
    data_load_time = time.time() - _start

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = SimpleCNN(
        input_channels=num_channels,
        num_layers=num_layers,
        num_classes=num_classes,
        hidden_dim=num_neurons,
        image_height=image_height,
        image_width=image_width,
        dropout=dropout
    )
    model = model.to(device)

    # Instantiate loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize an optimizer
    optimizer_name = optimizer
    optimizer = get_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Initialize LR scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler="cosine",
        scheduler_args={
            "T_max": epochs,
            "eta_min": 1e-6
        }
    )

    # Load possible checkpoint
    start = time.time()
    steps = None
    if allow_checkpointing:
        steps, model, optimizer = load_neps_checkpoint(
            previous_pipeline_directory, model, optimizer, scheduler
        )
    checkpoint_load_time = time.time() - start

    train_start = time.time()
    validation_time = 0
    # Training loop
    steps = steps or 0  # accounting for continuation if checkpoint loaded
    for epoch in range(steps, epochs):
        # perform one epoch of training
        model.train()
        model, optimizer, scheduler, mean_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            device=device
        )

        # perform validation per epoch
        start = time.time()
        val_loss = validate_model(model, val_loader, criterion, device=device)
        validation_time += (time.time() - start)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"loss: {mean_loss:.5f}, "
                f"val loss: {val_loss:.5f}"
            )

        # refer https://automl.github.io/neps/latest/examples/convenience/neps_tblogger_tutorial/
        start = time.time()
        if log_tensorboard:
            tblogger.log(
                loss=val_loss,
                current_epoch=epoch+1,
                # write_summary_incumbent=True,  # this fails, need to fix?
                writer_config_scalar=True,
                writer_config_hparam=True,
                extra_data={
                    "train_loss": tblogger.scalar_logging(value=mean_loss),
                    "l2_norm": tblogger.scalar_logging(
                        value=total_gradient_l2_norm(model)
                    ),
                },
            )
        logging_time = time.time() - start
    training_time = time.time() - train_start - validation_time - logging_time

    # Save checkpoint
    if allow_checkpointing:
        save_neps_checkpoint(pipeline_directory, epoch, model, optimizer, scheduler)

    return {
        "loss": val_loss,  # validation loss in the last epoch
        "cost": time.time() - _start,
        "info_dict": {
            "training_loss": mean_loss,  # training loss in the last epoch
            "data_load_time": data_load_time,
            "training_time": training_time,
            "validation_time": validation_time,
            "checkpoint_load_time": checkpoint_load_time,
            "logging_time": logging_time,
            "hyperparameters": {
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "epochs": epochs,
            },
        }
    }