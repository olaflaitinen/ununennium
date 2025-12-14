"""Run all benchmarks."""

from ununennium.benchmarks import benchmark_inference, benchmark_training
from ununennium.models import create_model
from ununennium.models.architectures import unet  # noqa: F401
import torch.nn as nn

model = create_model("unet_resnet50", in_channels=12, num_classes=10)

print("Inference benchmark:")
results = benchmark_inference(model, (1, 12, 256, 256), device="cpu", n_iterations=10)
print(f"  Avg latency: {results['avg_latency_ms']:.2f}ms")

print("Training benchmark:")
results = benchmark_training(
    model, nn.CrossEntropyLoss(),
    (4, 12, 256, 256), (4, 256, 256),
    device="cpu", n_iterations=5
)
print(f"  Avg iteration: {results['avg_iteration_ms']:.2f}ms")
