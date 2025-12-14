"""Classification training example."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ununennium.datasets import SyntheticDataset
from ununennium.models.backbones import ResNetBackbone
from ununennium.models.heads import ClassificationHead


def collate_fn(batch):
    images = torch.stack([item[0].data for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels


def main():
    # Dataset
    train_ds = SyntheticDataset(num_samples=500, task="classification", num_classes=10)
    train_loader = DataLoader(train_ds, batch_size=16, collate_fn=collate_fn)

    # Model
    backbone = ResNetBackbone(variant="resnet50", in_channels=12, pretrained=False)
    head = ClassificationHead(in_channels=2048, num_classes=10)

    class Classifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)
            return self.head([features[-1]])

    model = Classifier(backbone, head)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    for epoch in range(10):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
