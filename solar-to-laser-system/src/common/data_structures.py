"""
Core data structures for the Solar-to-Laser System.

This module defines the common data structures used throughout the system.
"""

from datetime import datetime
from typing import Dict, List, Any, Callable, Tuple, Optional


class SolarData:
    """
    Data structure for solar panel metrics.
    
    Attributes:
        timestamp: Time when the data was collected
        voltage: Voltage in Volts
        current: Current in Amperes
        power: Power output in Watts
        temperature: Temperature in Celsius
        irradiance: Solar irradiance in W/m²
        metadata: Additional metadata
    """
    
    def __init__(
        self,
        timestamp: datetime,
        voltage: float,
        current: float,
        power: float,
        temperature: Optional[float] = None,
        irradiance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = timestamp
        self.voltage = voltage
        self.current = current
        self.power = power
        self.temperature = temperature
        self.irradiance = irradiance
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the SolarData object to a dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "voltage": self.voltage,
            "current": self.current,
            "power": self.power,
            "temperature": self.temperature,
            "irradiance": self.irradiance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolarData':
        """Create a SolarData object from a dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            voltage=data["voltage"],
            current=data["current"],
            power=data["power"],
            temperature=data.get("temperature"),
            irradiance=data.get("irradiance"),
            metadata=data.get("metadata", {})
        )


class AudioParameters:
    """
    Parameters for audio conversion.
    
    Attributes:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        frequency_mapping: Function to map solar voltage to frequency
        amplitude_mapping: Function to map solar current to amplitude
        timbre_mapping: Function to map solar power to timbre
        temporal_mapping: Function to map time variations to temporal evolution
    """
    
    def __init__(
        self,
        duration: float,
        frequency_mapping: Callable[[float], float],
        amplitude_mapping: Callable[[float], float],
        timbre_mapping: Callable[[float], List[float]],
        temporal_mapping: Callable[[List[float]], List[float]],
        sample_rate: int = 44100
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.frequency_mapping = frequency_mapping
        self.amplitude_mapping = amplitude_mapping
        self.timbre_mapping = timbre_mapping
        self.temporal_mapping = temporal_mapping
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the AudioParameters object to a dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            # Note: Functions cannot be serialized directly
            # Store function names or parameters instead
            "frequency_mapping_type": self.frequency_mapping.__name__,
            "amplitude_mapping_type": self.amplitude_mapping.__name__,
            "timbre_mapping_type": self.timbre_mapping.__name__,
            "temporal_mapping_type": self.temporal_mapping.__name__
        }


class VectorParameters:
    """
    Parameters for vector generation.
    
    Attributes:
        smoothing_factor: Factor for smoothing the vector paths
        interpolation_method: Method for interpolating between points
        scaling_factor: Factor for scaling the vectors
        normalization_range: Range for normalizing the vectors
        path_simplification: Factor for simplifying the vector paths
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.8,
        interpolation_method: str = "cubic",
        scaling_factor: float = 100.0,
        normalization_range: Tuple[float, float] = (-1.0, 1.0),
        path_simplification: float = 0.02
    ):
        self.smoothing_factor = smoothing_factor
        self.interpolation_method = interpolation_method
        self.scaling_factor = scaling_factor
        self.normalization_range = normalization_range
        self.path_simplification = path_simplification
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the VectorParameters object to a dictionary."""
        return {
            "smoothing_factor": self.smoothing_factor,
            "interpolation_method": self.interpolation_method,
            "scaling_factor": self.scaling_factor,
            "normalization_range": self.normalization_range,
            "path_simplification": self.path_simplification
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorParameters':
        """Create a VectorParameters object from a dictionary."""
        return cls(
            smoothing_factor=data.get("smoothing_factor", 0.8),
            interpolation_method=data.get("interpolation_method", "cubic"),
            scaling_factor=data.get("scaling_factor", 100.0),
            normalization_range=data.get("normalization_range", (-1.0, 1.0)),
            path_simplification=data.get("path_simplification", 0.02)
        )


class LaserParameters:
    """
    Parameters for laser control.
    
    Attributes:
        format: Format of the laser control file
        frame_rate: Frame rate in frames per second
        points_per_frame: Number of points per frame
        color_mode: Color mode (RGB, etc.)
        intensity: Intensity of the laser
        safety_limits: Safety limits for the laser
    """
    
    def __init__(
        self,
        format: str = "ILDA",
        frame_rate: int = 30,
        points_per_frame: int = 500,
        color_mode: str = "RGB",
        intensity: float = 0.8,
        safety_limits: Optional[Dict[str, float]] = None
    ):
        self.format = format
        self.frame_rate = frame_rate
        self.points_per_frame = points_per_frame
        self.color_mode = color_mode
        self.intensity = intensity
        self.safety_limits = safety_limits or {
            "max_intensity": 1.0,
            "max_scan_rate": 30000,
            "min_blanking_time": 0.001
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the LaserParameters object to a dictionary."""
        return {
            "format": self.format,
            "frame_rate": self.frame_rate,
            "points_per_frame": self.points_per_frame,
            "color_mode": self.color_mode,
            "intensity": self.intensity,
            "safety_limits": self.safety_limits
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LaserParameters':
        """Create a LaserParameters object from a dictionary."""
        return cls(
            format=data.get("format", "ILDA"),
            frame_rate=data.get("frame_rate", 30),
            points_per_frame=data.get("points_per_frame", 500),
            color_mode=data.get("color_mode", "RGB"),
            intensity=data.get("intensity", 0.8),
            safety_limits=data.get("safety_limits")
        )