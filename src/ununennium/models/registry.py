"""Model registry for dynamic model creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", bound=nn.Module)

# Global model registry
_MODELS: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a model class.

    Args:
        name: Name to register the model under.

    Returns:
        Decorator function.

    Example:
        >>> @register_model("my_unet")
        ... class MyUNet(nn.Module):
        ...     pass
    """

    def decorator(cls: type[T]) -> type[T]:
        _MODELS[name] = cls
        return cls

    return decorator


def create_model(name: str, **kwargs: Any) -> nn.Module:
    """Create a model by name from the registry.

    Args:
        name: Registered model name.
        **kwargs: Model configuration arguments.

    Returns:
        Instantiated model.

    Raises:
        KeyError: If model name is not registered.

    Example:
        >>> model = create_model("unet", backbone="resnet50", num_classes=10)
    """
    if name not in _MODELS:
        available = ", ".join(sorted(_MODELS.keys()))
        raise KeyError(f"Unknown model: {name}. Available: {available}")

    return _MODELS[name](**kwargs)


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        Sorted list of model names.
    """
    return sorted(_MODELS.keys())


def get_model_class(name: str) -> Callable[..., nn.Module]:
    """Get the model class (not instance) by name.

    Args:
        name: Registered model name.

    Returns:
        Model class.
    """
    if name not in _MODELS:
        raise KeyError(f"Unknown model: {name}")
    return _MODELS[name]
