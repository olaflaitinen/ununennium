# Distributed Training Tutorial

Scale training across multiple GPUs.

## Single Node Multi-GPU

```bash
torchrun --nproc_per_node=4 train.py
```

## Training Script

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Model
model = create_model("unet_resnet50", in_channels=12, num_classes=10)
model = model.cuda()
model = DDP(model, device_ids=[local_rank])

# Train normally
trainer = Trainer(model=model, ...)
trainer.fit(epochs=100)
```
