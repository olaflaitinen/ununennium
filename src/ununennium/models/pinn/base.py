"""Base classes for Physics-Informed Neural Networks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class PDEEquation(ABC):
    """Abstract base class for PDE definitions."""

    @abstractmethod
    def residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute PDE residual.

        Args:
            u: Solution field (network output).
            x: Spatial coordinates.
            t: Optional temporal coordinate.

        Returns:
            Residual tensor (should be zero for exact solution).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the PDE."""
        ...


class PINN(nn.Module):
    """Physics-Informed Neural Network base class.

    Combines data-driven learning with physical constraints through
    PDE residual minimization.

    Example:
        >>> equation = DiffusionEquation(diffusivity=0.1)
        >>> pinn = PINN(
        ...     network=MLP([2, 64, 64, 1]),
        ...     equation=equation,
        ... )
        >>> loss = pinn.compute_loss(x_data, u_data, x_collocation)
    """

    def __init__(
        self,
        network: nn.Module,
        equation: PDEEquation,
        lambda_data: float = 1.0,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
    ):
        super().__init__()
        self.network = network
        self.equation = equation
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the solution at given points.

        Args:
            x: Input coordinates (B, D) where D is spatial dimension.

        Returns:
            Solution values (B, 1) or (B, num_outputs).
        """
        return self.network(x)

    def compute_pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute PDE residual using automatic differentiation.

        Args:
            x: Spatial coordinates (requires_grad=True).
            t: Optional temporal coordinate.

        Returns:
            PDE residual values.
        """
        x = x.requires_grad_(True)
        if t is not None:
            t = t.requires_grad_(True)

        u = self.network(x if t is None else torch.cat([x, t], dim=-1))

        return self.equation.residual(u, x, t)

    def compute_loss(
        self,
        x_data: torch.Tensor | None = None,
        u_data: torch.Tensor | None = None,
        x_collocation: torch.Tensor | None = None,
        x_boundary: torch.Tensor | None = None,
        u_boundary: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute PINN loss components.

        Args:
            x_data: Observed data points.
            u_data: Observed solution values.
            x_collocation: Collocation points for PDE residual.
            x_boundary: Boundary points.
            u_boundary: Boundary values.

        Returns:
            Dictionary with loss components and total.
        """
        losses = {}
        total = torch.tensor(
            0.0, device=x_collocation.device if x_collocation is not None else "cpu"
        )

        # Data loss
        if x_data is not None and u_data is not None:
            u_pred = self.network(x_data)
            loss_data = nn.functional.mse_loss(u_pred, u_data)
            losses["data"] = loss_data
            total = total + self.lambda_data * loss_data

        # PDE residual loss
        if x_collocation is not None:
            residual = self.compute_pde_residual(x_collocation)
            loss_pde = (residual**2).mean()
            losses["pde"] = loss_pde
            total = total + self.lambda_pde * loss_pde

        # Boundary condition loss
        if x_boundary is not None and u_boundary is not None:
            u_bc_pred = self.network(x_boundary)
            loss_bc = nn.functional.mse_loss(u_bc_pred, u_boundary)
            losses["boundary"] = loss_bc
            total = total + self.lambda_bc * loss_bc

        losses["total"] = total
        return losses


class MLP(nn.Module):
    """Multi-layer perceptron for PINN solutions."""

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "tanh",
    ):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "silu":
                    layers.append(nn.SiLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
