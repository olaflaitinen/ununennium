"""Band specifications for common satellite sensors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BandSpec:
    """Specification for a single spectral band.

    Attributes:
        name: Band identifier (e.g., "B02", "NIR").
        common_name: Common name (e.g., "blue", "nir").
        wavelength_nm: Center wavelength in nanometers.
        bandwidth_nm: Bandwidth in nanometers.
        resolution_m: Spatial resolution in meters.
        description: Human-readable description.
    """

    name: str
    common_name: str
    wavelength_nm: float
    bandwidth_nm: float
    resolution_m: float
    description: str = ""


# Sentinel-2 MSI L2A Band Specifications
SENTINEL2_BANDS: dict[str, BandSpec] = {
    "B01": BandSpec("B01", "coastal", 443, 20, 60, "Coastal aerosol"),
    "B02": BandSpec("B02", "blue", 490, 65, 10, "Blue"),
    "B03": BandSpec("B03", "green", 560, 35, 10, "Green"),
    "B04": BandSpec("B04", "red", 665, 30, 10, "Red"),
    "B05": BandSpec("B05", "rededge1", 705, 15, 20, "Red Edge 1"),
    "B06": BandSpec("B06", "rededge2", 740, 15, 20, "Red Edge 2"),
    "B07": BandSpec("B07", "rededge3", 783, 20, 20, "Red Edge 3"),
    "B08": BandSpec("B08", "nir", 842, 115, 10, "NIR"),
    "B8A": BandSpec("B8A", "nir08", 865, 20, 20, "NIR Narrow"),
    "B09": BandSpec("B09", "watervapor", 945, 20, 60, "Water Vapour"),
    "B11": BandSpec("B11", "swir16", 1610, 90, 20, "SWIR 1.6μm"),
    "B12": BandSpec("B12", "swir22", 2190, 180, 20, "SWIR 2.2μm"),
}

# Landsat-8/9 OLI Band Specifications
LANDSAT8_BANDS: dict[str, BandSpec] = {
    "B1": BandSpec("B1", "coastal", 443, 16, 30, "Coastal Aerosol"),
    "B2": BandSpec("B2", "blue", 482, 60, 30, "Blue"),
    "B3": BandSpec("B3", "green", 561, 57, 30, "Green"),
    "B4": BandSpec("B4", "red", 655, 37, 30, "Red"),
    "B5": BandSpec("B5", "nir", 865, 28, 30, "NIR"),
    "B6": BandSpec("B6", "swir16", 1609, 85, 30, "SWIR 1.6μm"),
    "B7": BandSpec("B7", "swir22", 2201, 187, 30, "SWIR 2.2μm"),
    "B8": BandSpec("B8", "pan", 590, 172, 15, "Panchromatic"),
    "B9": BandSpec("B9", "cirrus", 1373, 20, 30, "Cirrus"),
}

# MODIS Band Specifications (selected bands)
MODIS_BANDS: dict[str, BandSpec] = {
    "B1": BandSpec("B1", "red", 645, 50, 250, "Red"),
    "B2": BandSpec("B2", "nir", 858, 35, 250, "NIR"),
    "B3": BandSpec("B3", "blue", 469, 20, 500, "Blue"),
    "B4": BandSpec("B4", "green", 555, 20, 500, "Green"),
}


def get_band_names(sensor: str, resolution: int | None = None) -> list[str]:
    """Get band names for a sensor, optionally filtered by resolution.

    Args:
        sensor: Sensor name ("sentinel2", "landsat8", "modis").
        resolution: Optional resolution filter in meters.

    Returns:
        List of band names.
    """
    sensors = {
        "sentinel2": SENTINEL2_BANDS,
        "landsat8": LANDSAT8_BANDS,
        "modis": MODIS_BANDS,
    }

    bands = sensors.get(sensor.lower(), {})

    if resolution is None:
        return list(bands.keys())

    return [name for name, spec in bands.items() if spec.resolution_m == resolution]


def get_rgb_bands(sensor: str) -> tuple[str, str, str]:
    """Get RGB band names for a sensor.

    Args:
        sensor: Sensor name.

    Returns:
        Tuple of (red, green, blue) band names.
    """
    mappings = {
        "sentinel2": ("B04", "B03", "B02"),
        "landsat8": ("B4", "B3", "B2"),
        "modis": ("B1", "B4", "B3"),
    }
    return mappings.get(sensor.lower(), ("red", "green", "blue"))
