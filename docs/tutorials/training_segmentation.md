# Semantic Segmentation Tutorial

Train a U-Net for land cover classification.

## Setup

```python
from ununennium.models import create_model
from ununennium.training import Trainer, CheckpointCallback
from ununennium.datasets import SyntheticDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# Dataset
train_ds = SyntheticDataset(num_samples=1000, task="segmentation")
val_ds = SyntheticDataset(num_samples=200, task="segmentation")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Model
model = create_model("unet_resnet50", in_channels=12, num_classes=10)

# Training
trainer = Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    loss_fn=nn.CrossEntropyLoss(),
    train_loader=train_loader,
    val_loader=val_loader,
)

trainer.fit(epochs=50)
```
