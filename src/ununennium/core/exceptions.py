"""Exception hierarchy for ununennium."""

from __future__ import annotations


class UnunenniumError(Exception):
    """Base exception for all ununennium errors."""

    pass


class GeoError(UnunenniumError):
    """Base exception for geospatial operations."""

    pass


class CRSError(GeoError):
    """Error related to coordinate reference system operations."""

    pass


class TransformError(GeoError):
    """Error related to affine transform operations."""

    pass


class BoundsError(GeoError):
    """Error related to bounding box operations."""

    pass


class IOError(UnunenniumError):
    """Error related to I/O operations."""

    pass


class InvalidFormatError(IOError):
    """File format is invalid or unsupported."""

    pass


class DataNotFoundError(IOError):
    """Requested data was not found."""

    pass


class ModelError(UnunenniumError):
    """Error related to model operations."""

    pass


class CheckpointError(ModelError):
    """Error loading or saving checkpoints."""

    pass


class ConfigurationError(UnunenniumError):
    """Error in configuration or parameters."""

    pass


class ValidationError(UnunenniumError):
    """Data validation failed."""

    pass


class ShapeError(ValidationError):
    """Tensor shape is invalid."""

    pass


class DTypeError(ValidationError):
    """Data type is invalid."""

    pass
