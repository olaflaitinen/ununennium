"""Collocation point samplers for PINN training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class CollocationSampler(ABC):
    """Abstract base class for collocation point sampling."""

    @abstractmethod
    def sample(self, n_points: int) -> torch.Tensor:
        """Sample collocation points.

        Args:
            n_points: Number of points to sample.

        Returns:
            Tensor of shape (n_points, dim).
        """
        ...


class UniformSampler(CollocationSampler):
    """Uniform random sampling in a rectangular domain."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        device: str = "cpu",
    ):
        """Initialize uniform sampler.

        Args:
            bounds: List of (min, max) tuples for each dimension.
            device: Device for output tensors.
        """
        self.bounds = bounds
        self.dim = len(bounds)
        self.device = device

    def sample(self, n_points: int) -> torch.Tensor:
        points = torch.rand(n_points, self.dim, device=self.device)
        for i, (low, high) in enumerate(self.bounds):
            points[:, i] = points[:, i] * (high - low) + low
        return points


class LatinHypercubeSampler(CollocationSampler):
    """Latin Hypercube Sampling for better space coverage."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        device: str = "cpu",
    ):
        self.bounds = bounds
        self.dim = len(bounds)
        self.device = device

    def sample(self, n_points: int) -> torch.Tensor:
        # Create Latin Hypercube samples
        points = torch.zeros(n_points, self.dim, device=self.device)

        for i in range(self.dim):
            # Create evenly spaced intervals
            intervals = torch.arange(n_points, device=self.device) / n_points
            intervals = intervals + torch.rand(n_points, device=self.device) / n_points

            # Shuffle
            perm = torch.randperm(n_points, device=self.device)
            intervals = intervals[perm]

            # Scale to bounds
            low, high = self.bounds[i]
            points[:, i] = intervals * (high - low) + low

        return points


class AdaptiveSampler(CollocationSampler):
    """Adaptive sampling based on residual values.

    Samples more points where PDE residual is high.
    """

    def __init__(
        self,
        base_sampler: CollocationSampler,
        refinement_ratio: float = 0.5,
    ):
        """Initialize adaptive sampler.

        Args:
            base_sampler: Base sampler for initial points.
            refinement_ratio: Fraction of points to refine.
        """
        self.base_sampler = base_sampler
        self.refinement_ratio = refinement_ratio
        self._residuals: torch.Tensor | None = None
        self._points: torch.Tensor | None = None

    def sample(self, n_points: int) -> torch.Tensor:
        # Initial uniform sampling
        if self._residuals is None:
            return self.base_sampler.sample(n_points)

        # Adaptive refinement
        n_base = int(n_points * (1 - self.refinement_ratio))
        n_refine = n_points - n_base

        base_points = self.base_sampler.sample(n_base)

        # Sample near high-residual regions
        if self._points is not None and n_refine > 0:
            # Weight by residual magnitude
            weights = self._residuals.abs()
            weights = weights / weights.sum()

            # Sample indices
            indices = torch.multinomial(weights.flatten(), n_refine, replacement=True)
            centers = self._points[indices]

            # Add small perturbations
            perturbations = torch.randn_like(centers) * 0.1
            refine_points = centers + perturbations

            return torch.cat([base_points, refine_points], dim=0)

        return base_points

    def update_residuals(
        self,
        points: torch.Tensor,
        residuals: torch.Tensor,
    ) -> None:
        """Update residual information for adaptive sampling."""
        self._points = points.detach()
        self._residuals = residuals.detach()
