"""Convergence benchmarking script."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ununennium.models import create_model
from ununennium.datasets import SyntheticDataset
from ununennium.benchmarks.profiler import Profiler

def benchmark_convergence(
    model_name: str = "unet_resnet18",
    batch_size: int = 16,
    max_steps: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Benchmarking convergence for {model_name} on {device}...")
    
    # Setup
    model = create_model(model_name, in_channels=3, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    dataset = SyntheticDataset(num_samples=batch_size * 20, num_channels=3, task="segmentation")
    loader = DataLoader(dataset, batch_size=batch_size)
    
    losses = []
    profiler = Profiler()
    
    start_time = time.time()
    step = 0
    
    model.train()
    while step < max_steps:
        for images, labels in loader:
            if step >= max_steps:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            with profiler.section("step"):
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            
            losses.append(loss.item())
            step += 1
            
            if step % 10 == 0:
                print(f"Step {step}: Loss {loss.item():.4f}")

    total_time = time.time() - start_time
    print(f"\nConvergence Benchmark Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Steps/sec: {max_steps/total_time:.2f}")

if __name__ == "__main__":
    benchmark_convergence()
