"""
Pytest configuration file.

This file contains fixtures and configuration for pytest.
"""

import os
import tempfile
import numpy as np
import torch
from datetime import datetime
import pytest
from unittest.mock import MagicMock

from src.common.data_structures import SolarData, AudioParameters, VectorParameters, LaserParameters
from src.data_collection.collector import SolarDataCollector
from src.data_collection.storage import InfluxDBStorage
from src.audio_conversion.converter import SolarToAudioConverter
from src.audio_conversion.synthesis import DirectMappingSynthesizer
from src.rave_processing.model import RAVEModel
from src.rave_processing.processor import RAVEProcessor
from src.vector_generation.generator import VectorGenerator
from src.vector_generation.mapping import DirectMapping
from src.laser_control.controller import LaserController, SimulationController
from src.laser_control.ilda import ILDAFile


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir.name
    temp_dir.cleanup()


@pytest.fixture
def solar_data():
    """Create test solar data."""
    return SolarData(
        timestamp=datetime.now(),
        voltage=12.5,
        current=2.1,
        power=26.25,
        temperature=25.0,
        irradiance=800.0,
        metadata={"panel_id": "panel1"}
    )


@pytest.fixture
def solar_data_list():
    """Create a list of test solar data."""
    return [
        SolarData(
            timestamp=datetime.now(),
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        ),
        SolarData(
            timestamp=datetime.now(),
            voltage=12.6,
            current=2.2,
            power=27.72,
            temperature=25.1,
            irradiance=810.0,
            metadata={"panel_id": "panel1"}
        ),
        SolarData(
            timestamp=datetime.now(),
            voltage=12.4,
            current=2.0,
            power=24.8,
            temperature=24.9,
            irradiance=790.0,
            metadata={"panel_id": "panel1"}
        )
    ]


@pytest.fixture
def audio_params():
    """Create test audio parameters."""
    return AudioParameters(
        sample_rate=44100,
        duration=1.0,
        frequency_mapping=lambda v: 220 + 660 * (v / 24.0),
        amplitude_mapping=lambda c: min(0.9, c / 10.0),
        timbre_mapping=lambda p: [1.0, 0.5, 0.25, 0.125],
        temporal_mapping=lambda values: values
    )


@pytest.fixture
def vector_params():
    """Create test vector parameters."""
    return VectorParameters(
        smoothing_factor=0.8,
        interpolation_method="cubic",
        scaling_factor=100.0,
        normalization_range=(-1.0, 1.0),
        path_simplification=0.02
    )


@pytest.fixture
def laser_params():
    """Create test laser parameters."""
    return LaserParameters(
        format="ILDA",
        frame_rate=30,
        points_per_frame=500,
        color_mode="RGB",
        intensity=0.8,
        safety_limits={"max_power": 0.9, "max_angle": 45.0}
    )


@pytest.fixture
def mock_storage():
    """Create a mock storage."""
    return MagicMock(spec=InfluxDBStorage)


@pytest.fixture
def collector(mock_storage):
    """Create a test collector."""
    return SolarDataCollector(storage=mock_storage)


@pytest.fixture
def synthesizer(audio_params):
    """Create a test synthesizer."""
    return DirectMappingSynthesizer(audio_params)


@pytest.fixture
def converter(synthesizer):
    """Create a test converter."""
    return SolarToAudioConverter(synthesizer=synthesizer)


@pytest.fixture
def test_audio():
    """Create test audio data."""
    return np.sin(np.linspace(0, 2 * np.pi, 44100))


@pytest.fixture
def mock_rave_model():
    """Create a mock RAVE model."""
    mock_model = MagicMock(spec=RAVEModel)
    mock_model.encode.return_value = torch.randn(1, 16, 100)
    mock_model.decode.return_value = np.sin(np.linspace(0, 2 * np.pi, 44100))
    return mock_model


@pytest.fixture
def processor(mock_rave_model):
    """Create a test processor."""
    return RAVEProcessor(model=mock_rave_model)


@pytest.fixture
def test_latent():
    """Create test latent representation."""
    return torch.randn(1, 16, 100)


@pytest.fixture
def mapping():
    """Create a test mapping."""
    return DirectMapping()


@pytest.fixture
def generator(mapping):
    """Create a test generator."""
    return VectorGenerator(mapping=mapping)


@pytest.fixture
def test_vectors():
    """Create test vectors."""
    return np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5]
    ])


@pytest.fixture
def controller():
    """Create a test controller."""
    return SimulationController()


@pytest.fixture
def ilda_file(test_vectors):
    """Create a test ILDA file."""
    ilda_file = ILDAFile()
    
    # Add a frame with the test vectors
    frame = np.zeros((len(test_vectors), 5))
    frame[:, 0:2] = test_vectors
    frame[:, 2:5] = 1.0  # Set RGB to white
    
    ilda_file.add_frame(frame)
    
    return ilda_file


@pytest.fixture
def audio_path(temp_dir):
    """Create a path for test audio file."""
    return os.path.join(temp_dir, "audio.wav")


@pytest.fixture
def latent_path(temp_dir):
    """Create a path for test latent file."""
    return os.path.join(temp_dir, "latent.pt")


@pytest.fixture
def processed_audio_path(temp_dir):
    """Create a path for test processed audio file."""
    return os.path.join(temp_dir, "processed.wav")


@pytest.fixture
def vector_path(temp_dir):
    """Create a path for test vector file."""
    return os.path.join(temp_dir, "vectors.svg")


@pytest.fixture
def laser_path(temp_dir):
    """Create a path for test laser file."""
    return os.path.join(temp_dir, "laser.ild")


@pytest.fixture
def simulation_path(temp_dir):
    """Create a path for test simulation file."""
    return os.path.join(temp_dir, "simulation.mp4")


@pytest.fixture
def test_svg_file(temp_dir, test_vectors):
    """Create a test SVG file."""
    svg_path = os.path.join(temp_dir, "test.svg")
    
    # Create a simple SVG file
    with open(svg_path, "w") as f:
        f.write("""
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <path d="M 0,0 L 50,50 L 100,100 L 50,0 L 0,50 Z" fill="none" stroke="black" />
        </svg>
        """)
    
    return svg_path


@pytest.fixture
def test_wav_file(temp_dir, test_audio):
    """Create a test WAV file."""
    wav_path = os.path.join(temp_dir, "test.wav")
    
    # Create a WAV file
    import wave
    import struct
    
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        for sample in test_audio:
            wav_file.writeframes(struct.pack("h", int(sample * 32767)))
    
    return wav_path


@pytest.fixture
def test_ild_file(temp_dir, ilda_file):
    """Create a test ILDA file."""
    ild_path = os.path.join(temp_dir, "test.ild")
    
    # Save the ILDA file
    ilda_file.save(ild_path)
    
    return ild_path