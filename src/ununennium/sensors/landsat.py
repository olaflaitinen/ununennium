"""Landsat sensor specification."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Landsat8:
    """Landsat-8 OLI sensor specification."""

    BANDS: ClassVar[dict[str, dict[str, str | int]]] = {
        "B1": {"wavelength": 443, "resolution": 30, "name": "Coastal"},
        "B2": {"wavelength": 482, "resolution": 30, "name": "Blue"},
        "B3": {"wavelength": 561, "resolution": 30, "name": "Green"},
        "B4": {"wavelength": 655, "resolution": 30, "name": "Red"},
        "B5": {"wavelength": 865, "resolution": 30, "name": "NIR"},
        "B6": {"wavelength": 1609, "resolution": 30, "name": "SWIR1"},
        "B7": {"wavelength": 2201, "resolution": 30, "name": "SWIR2"},
        "B8": {"wavelength": 590, "resolution": 15, "name": "Pan"},
        "B9": {"wavelength": 1373, "resolution": 30, "name": "Cirrus"},
    }
