"""Training callbacks for monitoring and control."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ununennium.training.trainer import Trainer


class Callback:
    """Base class for training callbacks.

    Override methods to hook into the training loop at various points.
    """

    def on_train_start(self, trainer: Trainer) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: Trainer, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: Trainer, batch: int) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer: Trainer, batch: int, logs: dict[str, float]) -> None:
        """Called at the end of each batch."""
        pass


class CheckpointCallback(Callback):
    """Save model checkpoints during training.

    Tracks the best models by a monitored metric and saves them.
    """

    def __init__(
        self,
        output_dir: Path | str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
    ):
        """Initialize checkpoint callback.

        Args:
            output_dir: Directory to save checkpoints.
            monitor: Metric name to monitor.
            mode: 'min' or 'max' - whether lower or higher is better.
            save_top_k: Number of best checkpoints to keep.
            save_last: Whether to save a 'last.pt' checkpoint.
        """
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last

        self.best_metrics: list[tuple[float, Path]] = []

    def on_train_start(self, _trainer: Trainer) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        metric = logs.get(self.monitor)
        if metric is None:
            return

        # Save checkpoint
        ckpt_path = self.output_dir / f"epoch_{epoch:04d}.pt"
        trainer.save_checkpoint(ckpt_path)

        # Track best checkpoints
        self.best_metrics.append((metric, ckpt_path))
        self.best_metrics.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Remove old checkpoints
        while len(self.best_metrics) > self.save_top_k:
            _, old_path = self.best_metrics.pop()
            if old_path.exists():
                old_path.unlink()

        # Save last checkpoint
        if self.save_last:
            last_path = self.output_dir / "last.pt"
            trainer.save_checkpoint(last_path)


class EarlyStoppingCallback(Callback):
    """Stop training when a metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 1e-4,
    ):
        """Initialize early stopping.

        Args:
            monitor: Metric name to monitor.
            patience: Number of epochs with no improvement to wait.
            mode: 'min' or 'max'.
            min_delta: Minimum change to qualify as improvement.
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def on_epoch_end(self, trainer: Trainer, _epoch: int, logs: dict[str, float]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (self.mode == "min" and current < self.best - self.min_delta) or (
            self.mode == "max" and current > self.best + self.min_delta
        )

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True


class ProgressCallback(Callback):
    """Print training progress."""

    def on_epoch_end(self, _trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        log_str = " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        print(f"Epoch {epoch + 1}: {log_str}")
