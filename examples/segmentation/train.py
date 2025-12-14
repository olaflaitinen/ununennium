"""Semantic Segmentation Example

This example demonstrates training a U-Net model for land cover
classification on synthetic data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ununennium.models import create_model
from ununennium.models.architectures import unet  # noqa: F401  # Register models
from ununennium.datasets import SyntheticDataset
from ununennium.core import GeoBatch
from ununennium.training import Trainer, CheckpointCallback, EarlyStoppingCallback
from ununennium.training.callbacks import ProgressCallback


def collate_fn(batch):
    """Custom collate function for GeoTensor datasets."""
    images = torch.stack([item[0].data for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels


def main():
    # Configuration
    config = {
        "in_channels": 12,
        "num_classes": 10,
        "batch_size": 8,
        "epochs": 50,
        "lr": 1e-4,
        "image_size": 256,
    }

    print("=" * 60)
    print("Ununennium - Semantic Segmentation Example")
    print("=" * 60)

    # Create datasets
    print("\n[1/4] Creating datasets...")
    train_dataset = SyntheticDataset(
        num_samples=500,
        num_channels=config["in_channels"],
        image_size=config["image_size"],
        num_classes=config["num_classes"],
        task="segmentation",
        seed=42,
    )
    val_dataset = SyntheticDataset(
        num_samples=100,
        num_channels=config["in_channels"],
        image_size=config["image_size"],
        num_classes=config["num_classes"],
        task="segmentation",
        seed=43,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create model
    print("\n[2/4] Creating model...")
    model = create_model(
        "unet_resnet50",
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        pretrained=False,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: U-Net with ResNet-50")
    print(f"   Parameters: {num_params:,}")

    # Setup training
    print("\n[3/4] Setting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    callbacks = [
        ProgressCallback(),
        EarlyStoppingCallback(monitor="val_loss", patience=10),
    ]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
    )

    # Train
    print("\n[4/4] Training...")
    print("-" * 40)
    history = trainer.fit(epochs=config["epochs"])

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
