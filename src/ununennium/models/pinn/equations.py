"""Common PDE equations for remote sensing applications."""

from __future__ import annotations

import torch

from ununennium.models.pinn.base import PDEEquation


def gradient(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute gradient using autograd."""
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian using autograd."""
    grad_u = gradient(u, x)
    lapl = torch.zeros_like(u)
    for i in range(x.shape[-1]):
        grad_u_i = grad_u[..., i : i + 1]
        grad2_u_i = torch.autograd.grad(
            grad_u_i,
            x,
            grad_outputs=torch.ones_like(grad_u_i),
            create_graph=True,
            retain_graph=True,
        )[0][..., i : i + 1]
        lapl = lapl + grad2_u_i
    return lapl


class DiffusionEquation(PDEEquation):
    """Heat/diffusion equation: ∂u/∂t = D∇²u

    Models thermal diffusion, pollutant spread, etc.
    """

    def __init__(self, diffusivity: float = 1.0):
        self.diffusivity = diffusivity

    @property
    def name(self) -> str:
        return "diffusion"

    def residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is None:
            # Steady-state diffusion: ∇²u = 0
            return laplacian(u, x)

        # Time-dependent diffusion
        du_dt = gradient(u, t)
        lapl_u = laplacian(u, x)
        return du_dt - self.diffusivity * lapl_u


class AdvectionEquation(PDEEquation):
    """Advection equation: ∂u/∂t + v·∇u = 0

    Models transport by flow (e.g., smoke, pollution).
    """

    def __init__(self, velocity: tuple[float, ...] = (1.0, 0.0)):
        self.velocity = torch.tensor(velocity)

    @property
    def name(self) -> str:
        return "advection"

    def residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grad_u = gradient(u, x)
        velocity = self.velocity.to(x.device).view(1, -1)
        advection = (velocity * grad_u).sum(dim=-1, keepdim=True)

        if t is None:
            return advection

        du_dt = gradient(u, t)
        return du_dt + advection


class AdvectionDiffusionEquation(PDEEquation):
    """Advection-diffusion equation: ∂u/∂t + v·∇u = D∇²u"""

    def __init__(
        self,
        velocity: tuple[float, ...] = (1.0, 0.0),
        diffusivity: float = 0.1,
    ):
        self.velocity = torch.tensor(velocity)
        self.diffusivity = diffusivity

    @property
    def name(self) -> str:
        return "advection_diffusion"

    def residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grad_u = gradient(u, x)
        velocity = self.velocity.to(x.device).view(1, -1)
        advection = (velocity * grad_u).sum(dim=-1, keepdim=True)
        diffusion = self.diffusivity * laplacian(u, x)

        if t is None:
            return advection - diffusion

        du_dt = gradient(u, t)
        return du_dt + advection - diffusion
