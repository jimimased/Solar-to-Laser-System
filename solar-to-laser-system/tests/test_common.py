"""
Tests for the common module.
"""

import unittest
from datetime import datetime
from typing import Dict, Any, List

from src.common.data_structures import (
    SolarData,
    AudioParameters,
    VectorParameters,
    LaserParameters
)


class TestSolarData(unittest.TestCase):
    """Tests for the SolarData class."""
    
    def test_initialization(self):
        """Test initializing a SolarData object."""
        # Create a SolarData object
        data = SolarData(
            timestamp=datetime.now(),
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Assert that the attributes are set correctly
        self.assertEqual(data.voltage, 12.5)
        self.assertEqual(data.current, 2.1)
        self.assertEqual(data.power, 26.25)
        self.assertEqual(data.temperature, 25.0)
        self.assertEqual(data.irradiance, 800.0)
        self.assertEqual(data.metadata["panel_id"], "panel1")
    
    def test_to_dict(self):
        """Test converting a SolarData object to a dictionary."""
        # Create a SolarData object
        timestamp = datetime.now()
        data = SolarData(
            timestamp=timestamp,
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Convert to dictionary
        data_dict = data.to_dict()
        
        # Assert that the dictionary has the correct keys and values
        self.assertIn("timestamp", data_dict)
        self.assertIn("voltage", data_dict)
        self.assertIn("current", data_dict)
        self.assertIn("power", data_dict)
        self.assertIn("temperature", data_dict)
        self.assertIn("irradiance", data_dict)
        self.assertIn("metadata", data_dict)
        
        self.assertEqual(data_dict["timestamp"], timestamp)
        self.assertEqual(data_dict["voltage"], 12.5)
        self.assertEqual(data_dict["current"], 2.1)
        self.assertEqual(data_dict["power"], 26.25)
        self.assertEqual(data_dict["temperature"], 25.0)
        self.assertEqual(data_dict["irradiance"], 800.0)
        self.assertEqual(data_dict["metadata"]["panel_id"], "panel1")
    
    def test_from_dict(self):
        """Test creating a SolarData object from a dictionary."""
        # Create a dictionary
        timestamp = datetime.now()
        data_dict = {
            "timestamp": timestamp,
            "voltage": 12.5,
            "current": 2.1,
            "power": 26.25,
            "temperature": 25.0,
            "irradiance": 800.0,
            "metadata": {"panel_id": "panel1"}
        }
        
        # Create a SolarData object from the dictionary
        data = SolarData.from_dict(data_dict)
        
        # Assert that the attributes are set correctly
        self.assertEqual(data.timestamp, timestamp)
        self.assertEqual(data.voltage, 12.5)
        self.assertEqual(data.current, 2.1)
        self.assertEqual(data.power, 26.25)
        self.assertEqual(data.temperature, 25.0)
        self.assertEqual(data.irradiance, 800.0)
        self.assertEqual(data.metadata["panel_id"], "panel1")
    
    def test_str_representation(self):
        """Test the string representation of a SolarData object."""
        # Create a SolarData object
        data = SolarData(
            timestamp=datetime.now(),
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Get the string representation
        data_str = str(data)
        
        # Assert that the string contains the important information
        self.assertIn("SolarData", data_str)
        self.assertIn("voltage=12.5", data_str)
        self.assertIn("current=2.1", data_str)
        self.assertIn("power=26.25", data_str)


class TestAudioParameters(unittest.TestCase):
    """Tests for the AudioParameters class."""
    
    def test_initialization(self):
        """Test initializing an AudioParameters object."""
        # Create mapping functions
        def frequency_mapping(voltage):
            return 220 + 660 * (voltage / 24.0)
        
        def amplitude_mapping(current):
            return min(0.9, current / 10.0)
        
        def timbre_mapping(power):
            return [1.0, 0.5, 0.25, 0.125]
        
        def temporal_mapping(values):
            return values
        
        # Create an AudioParameters object
        params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=frequency_mapping,
            amplitude_mapping=amplitude_mapping,
            timbre_mapping=timbre_mapping,
            temporal_mapping=temporal_mapping
        )
        
        # Assert that the attributes are set correctly
        self.assertEqual(params.sample_rate, 44100)
        self.assertEqual(params.duration, 1.0)
        self.assertEqual(params.frequency_mapping(12.0), 220 + 660 * (12.0 / 24.0))
        self.assertEqual(params.amplitude_mapping(5.0), 0.5)
        self.assertEqual(params.timbre_mapping(60.0), [1.0, 0.5, 0.25, 0.125])
        self.assertEqual(params.temporal_mapping([1, 2, 3]), [1, 2, 3])
    
    def test_to_dict(self):
        """Test converting an AudioParameters object to a dictionary."""
        # Create mapping functions
        def frequency_mapping(voltage):
            return 220 + 660 * (voltage / 24.0)
        
        def amplitude_mapping(current):
            return min(0.9, current / 10.0)
        
        def timbre_mapping(power):
            return [1.0, 0.5, 0.25, 0.125]
        
        def temporal_mapping(values):
            return values
        
        # Create an AudioParameters object
        params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=frequency_mapping,
            amplitude_mapping=amplitude_mapping,
            timbre_mapping=timbre_mapping,
            temporal_mapping=temporal_mapping
        )
        
        # Convert to dictionary
        params_dict = params.to_dict()
        
        # Assert that the dictionary has the correct keys
        self.assertIn("sample_rate", params_dict)
        self.assertIn("duration", params_dict)
        
        # Functions cannot be serialized, so they are not included in the dictionary
        self.assertEqual(params_dict["sample_rate"], 44100)
        self.assertEqual(params_dict["duration"], 1.0)


class TestVectorParameters(unittest.TestCase):
    """Tests for the VectorParameters class."""
    
    def test_initialization(self):
        """Test initializing a VectorParameters object."""
        # Create a VectorParameters object
        params = VectorParameters(
            smoothing_factor=0.8,
            interpolation_method="cubic",
            scaling_factor=100.0,
            normalization_range=(-1.0, 1.0),
            path_simplification=0.02
        )
        
        # Assert that the attributes are set correctly
        self.assertEqual(params.smoothing_factor, 0.8)
        self.assertEqual(params.interpolation_method, "cubic")
        self.assertEqual(params.scaling_factor, 100.0)
        self.assertEqual(params.normalization_range, (-1.0, 1.0))
        self.assertEqual(params.path_simplification, 0.02)
    
    def test_to_dict(self):
        """Test converting a VectorParameters object to a dictionary."""
        # Create a VectorParameters object
        params = VectorParameters(
            smoothing_factor=0.8,
            interpolation_method="cubic",
            scaling_factor=100.0,
            normalization_range=(-1.0, 1.0),
            path_simplification=0.02
        )
        
        # Convert to dictionary
        params_dict = params.to_dict()
        
        # Assert that the dictionary has the correct keys and values
        self.assertIn("smoothing_factor", params_dict)
        self.assertIn("interpolation_method", params_dict)
        self.assertIn("scaling_factor", params_dict)
        self.assertIn("normalization_range", params_dict)
        self.assertIn("path_simplification", params_dict)
        
        self.assertEqual(params_dict["smoothing_factor"], 0.8)
        self.assertEqual(params_dict["interpolation_method"], "cubic")
        self.assertEqual(params_dict["scaling_factor"], 100.0)
        self.assertEqual(params_dict["normalization_range"], (-1.0, 1.0))
        self.assertEqual(params_dict["path_simplification"], 0.02)
    
    def test_from_dict(self):
        """Test creating a VectorParameters object from a dictionary."""
        # Create a dictionary
        params_dict = {
            "smoothing_factor": 0.8,
            "interpolation_method": "cubic",
            "scaling_factor": 100.0,
            "normalization_range": (-1.0, 1.0),
            "path_simplification": 0.02
        }
        
        # Create a VectorParameters object from the dictionary
        params = VectorParameters.from_dict(params_dict)
        
        # Assert that the attributes are set correctly
        self.assertEqual(params.smoothing_factor, 0.8)
        self.assertEqual(params.interpolation_method, "cubic")
        self.assertEqual(params.scaling_factor, 100.0)
        self.assertEqual(params.normalization_range, (-1.0, 1.0))
        self.assertEqual(params.path_simplification, 0.02)


class TestLaserParameters(unittest.TestCase):
    """Tests for the LaserParameters class."""
    
    def test_initialization(self):
        """Test initializing a LaserParameters object."""
        # Create a LaserParameters object
        params = LaserParameters(
            format="ILDA",
            frame_rate=30,
            points_per_frame=500,
            color_mode="RGB",
            intensity=0.8,
            safety_limits={"max_power": 0.9, "max_angle": 45.0}
        )
        
        # Assert that the attributes are set correctly
        self.assertEqual(params.format, "ILDA")
        self.assertEqual(params.frame_rate, 30)
        self.assertEqual(params.points_per_frame, 500)
        self.assertEqual(params.color_mode, "RGB")
        self.assertEqual(params.intensity, 0.8)
        self.assertEqual(params.safety_limits["max_power"], 0.9)
        self.assertEqual(params.safety_limits["max_angle"], 45.0)
    
    def test_to_dict(self):
        """Test converting a LaserParameters object to a dictionary."""
        # Create a LaserParameters object
        params = LaserParameters(
            format="ILDA",
            frame_rate=30,
            points_per_frame=500,
            color_mode="RGB",
            intensity=0.8,
            safety_limits={"max_power": 0.9, "max_angle": 45.0}
        )
        
        # Convert to dictionary
        params_dict = params.to_dict()
        
        # Assert that the dictionary has the correct keys and values
        self.assertIn("format", params_dict)
        self.assertIn("frame_rate", params_dict)
        self.assertIn("points_per_frame", params_dict)
        self.assertIn("color_mode", params_dict)
        self.assertIn("intensity", params_dict)
        self.assertIn("safety_limits", params_dict)
        
        self.assertEqual(params_dict["format"], "ILDA")
        self.assertEqual(params_dict["frame_rate"], 30)
        self.assertEqual(params_dict["points_per_frame"], 500)
        self.assertEqual(params_dict["color_mode"], "RGB")
        self.assertEqual(params_dict["intensity"], 0.8)
        self.assertEqual(params_dict["safety_limits"]["max_power"], 0.9)
        self.assertEqual(params_dict["safety_limits"]["max_angle"], 45.0)
    
    def test_from_dict(self):
        """Test creating a LaserParameters object from a dictionary."""
        # Create a dictionary
        params_dict = {
            "format": "ILDA",
            "frame_rate": 30,
            "points_per_frame": 500,
            "color_mode": "RGB",
            "intensity": 0.8,
            "safety_limits": {"max_power": 0.9, "max_angle": 45.0}
        }
        
        # Create a LaserParameters object from the dictionary
        params = LaserParameters.from_dict(params_dict)
        
        # Assert that the attributes are set correctly
        self.assertEqual(params.format, "ILDA")
        self.assertEqual(params.frame_rate, 30)
        self.assertEqual(params.points_per_frame, 500)
        self.assertEqual(params.color_mode, "RGB")
        self.assertEqual(params.intensity, 0.8)
        self.assertEqual(params.safety_limits["max_power"], 0.9)
        self.assertEqual(params.safety_limits["max_angle"], 45.0)


if __name__ == "__main__":
    unittest.main()