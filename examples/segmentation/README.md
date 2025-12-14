# Semantic Segmentation Example

Train a U-Net model for land cover classification.

## Usage

```bash
python train.py
```

## Configuration

Edit `train.py` to modify:
- `in_channels`: Number of spectral bands (default: 12)
- `num_classes`: Number of land cover classes (default: 10)
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `lr`: Learning rate
