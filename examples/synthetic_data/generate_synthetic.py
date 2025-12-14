"""Generate synthetic satellite imagery for testing."""

from ununennium.datasets import SyntheticDataset

dataset = SyntheticDataset(
    num_samples=100,
    num_channels=12,
    image_size=256,
    num_classes=10,
)

image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
