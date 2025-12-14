"""Sentinel-2 sensor specification."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Sentinel2:
    """Sentinel-2 MSI sensor specification."""

    BANDS: ClassVar[dict[str, dict[str, str | int]]] = {
        "B01": {"wavelength": 443, "resolution": 60, "name": "Coastal"},
        "B02": {"wavelength": 490, "resolution": 10, "name": "Blue"},
        "B03": {"wavelength": 560, "resolution": 10, "name": "Green"},
        "B04": {"wavelength": 665, "resolution": 10, "name": "Red"},
        "B05": {"wavelength": 705, "resolution": 20, "name": "RedEdge1"},
        "B06": {"wavelength": 740, "resolution": 20, "name": "RedEdge2"},
        "B07": {"wavelength": 783, "resolution": 20, "name": "RedEdge3"},
        "B08": {"wavelength": 842, "resolution": 10, "name": "NIR"},
        "B8A": {"wavelength": 865, "resolution": 20, "name": "NIR_Narrow"},
        "B09": {"wavelength": 945, "resolution": 60, "name": "WaterVapor"},
        "B10": {"wavelength": 1375, "resolution": 60, "name": "Cirrus"},
        "B11": {"wavelength": 1610, "resolution": 20, "name": "SWIR1"},
        "B12": {"wavelength": 2190, "resolution": 20, "name": "SWIR2"},
    }

    @classmethod
    def get_10m_bands(cls) -> list[str]:
        return ["B02", "B03", "B04", "B08"]

    @classmethod
    def get_20m_bands(cls) -> list[str]:
        return ["B05", "B06", "B07", "B8A", "B11", "B12"]
