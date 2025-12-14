# Benchmark Results

Device: cpu

| Model | Input | Batch | Latency(ms) | Img/s | Mem(MB) |
| --- | --- | --- | --- | --- | --- |
| unet_resnet18 | (12, 128, 128) | 8 | 1351.41 | 5.92 | 0.07 |
| unet_resnet50 | (12, 128, 128) | 4 | 1711.94 | 2.34 | 0.11 |
| unet_efficientnet_b0 | (12, 128, 128) | 6 | 1655.77 | 3.62 | 0.15 |
