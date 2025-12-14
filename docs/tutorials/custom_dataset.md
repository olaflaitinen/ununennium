# Custom Dataset Tutorial

Create custom datasets for your satellite imagery.

## Inherit from GeoDataset

```python
from ununennium.datasets.base import GeoDataset
from ununennium.core import GeoTensor
import ununennium as uu

class MyDataset(GeoDataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = uu.io.read_geotiff(self.image_paths[idx])
        label = uu.io.read_geotiff(self.label_paths[idx])
        return image, label.data.squeeze().long()

    @property
    def crs(self):
        return "EPSG:32632"
```

## Use with DataLoader

```python
from torch.utils.data import DataLoader

dataset = MyDataset(image_paths, label_paths)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```
