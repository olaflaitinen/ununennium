"""Core trainer for model training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from ununennium.training.callbacks import Callback


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_clip: float | None = 1.0
    accumulation_steps: int = 1
    log_interval: int = 10


class Trainer:
    """Core training loop with hooks for customization.

    Supports mixed precision, gradient accumulation, callbacks, and
    distributed training.

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     train_loader=train_loader,
        ... )
        >>> history = trainer.fit(epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module | Callable,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        scheduler: LRScheduler | None = None,
        callbacks: list[Callback] | None = None,
        config: TrainerConfig | None = None,
    ):
        """Initialize the trainer.

        Args:
            model: Neural network model.
            optimizer: Optimizer for training.
            loss_fn: Loss function.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            scheduler: Optional learning rate scheduler.
            callbacks: Optional list of callbacks.
            config: Trainer configuration.
        """
        self.config = config or TrainerConfig()

        self.model = model.to(self.config.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.callbacks = callbacks or []

        # Mixed precision scaler
        # Use 'cuda' device type for scaler if available, enabling usually only if on cuda
        self.scaler = (
            GradScaler("cuda")
            if (self.config.mixed_precision and self.config.device == "cuda")
            else None
        )

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False

    def fit(self, epochs: int | None = None) -> dict[str, list[float]]:
        """Run the training loop.

        Args:
            epochs: Number of epochs to train. Overrides config if provided.

        Returns:
            Dictionary mapping metric names to lists of values per epoch.
        """
        epochs = epochs or self.config.epochs
        history: dict[str, list[float]] = defaultdict(list)

        self._call_callbacks("on_train_start")

        for epoch in range(epochs):
            if self.should_stop:
                break

            self.current_epoch = epoch
            self._call_callbacks("on_epoch_start", epoch=epoch)

            # Training phase
            train_metrics = self._train_epoch()
            for k, v in train_metrics.items():
                history[f"train_{k}"].append(v)

            # Validation phase
            if self.val_loader is not None:
                val_metrics = self._validate()
                for k, v in val_metrics.items():
                    history[f"val_{k}"].append(v)
            else:
                val_metrics = {}

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Callbacks
            self._call_callbacks(
                "on_epoch_end",
                epoch=epoch,
                logs={**train_metrics, **val_metrics},
            )

        self._call_callbacks("on_train_end")

        return dict(history)

    def _train_epoch(self) -> dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            if hasattr(batch, "to"):
                batch = batch.to(self.config.device)
            elif isinstance(batch, (tuple, list)):
                batch = tuple(b.to(self.config.device) if hasattr(b, "to") else b for b in batch)

            # Forward pass
            # Use 'cuda' for autocast if mixed precision is on
            with autocast(device_type=self.config.device, enabled=self.scaler is not None):
                if hasattr(batch, "images"):
                    outputs = self.model(batch.images)
                    targets = batch.labels
                elif isinstance(batch, (tuple, list)):
                    outputs = self.model(batch[0])
                    targets = batch[1]
                else:
                    raise ValueError(f"Unknown batch type: {type(batch)}")

                loss = self.loss_fn(outputs, targets)
                loss = loss / self.config.accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.gradient_clip:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1

            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._call_callbacks(
                    "on_batch_end",
                    batch=batch_idx,
                    logs={"loss": loss.item() * self.config.accumulation_steps},
                )

        return {"loss": total_loss / num_batches}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Move to device
            if hasattr(batch, "to"):
                batch = batch.to(self.config.device)
            elif isinstance(batch, (tuple, list)):
                batch = tuple(b.to(self.config.device) if hasattr(b, "to") else b for b in batch)

            # Forward pass
            if hasattr(batch, "images"):
                outputs = self.model(batch.images)
                targets = batch.labels
            elif isinstance(batch, (tuple, list)):
                outputs = self.model(batch[0])
                targets = batch[1]
            else:
                raise ValueError(f"Unknown batch type: {type(batch)}")

            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

        return {"loss": total_loss / num_batches}

    def _call_callbacks(self, hook: str, **kwargs: Any) -> None:
        """Call all callbacks for a specific hook."""
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is not None:
                method(self, **kwargs)

    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save the checkpoint.
        """
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        if self.scheduler is not None and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        if self.scaler is not None and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
