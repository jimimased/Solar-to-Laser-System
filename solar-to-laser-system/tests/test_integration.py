"""
Integration tests for the Solar-to-Laser System.
"""

import unittest
import os
import tempfile
import time
import numpy as np
import torch
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

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
from src.laser_control.ilda import ILDAFile, convert_svg_to_ilda


@pytest.mark.integration
class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration tests for the Solar-to-Laser System pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test data
        self.solar_data = [
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
        
        # Set up audio parameters
        self.audio_params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=lambda v: 220 + 660 * (v / 24.0),
            amplitude_mapping=lambda c: min(0.9, c / 10.0),
            timbre_mapping=lambda p: [1.0, 0.5, 0.25, 0.125],
            temporal_mapping=lambda values: values
        )
        
        # Set up vector parameters
        self.vector_params = VectorParameters(
            smoothing_factor=0.8,
            interpolation_method="cubic",
            scaling_factor=100.0,
            normalization_range=(-1.0, 1.0),
            path_simplification=0.02
        )
        
        # Set up laser parameters
        self.laser_params = LaserParameters(
            format="ILDA",
            frame_rate=30,
            points_per_frame=500,
            color_mode="RGB",
            intensity=0.8,
            safety_limits={"max_power": 0.9, "max_angle": 45.0}
        )
        
        # Create output file paths
        self.audio_path = os.path.join(self.output_dir, "audio.wav")
        self.latent_path = os.path.join(self.output_dir, "latent.pt")
        self.processed_audio_path = os.path.join(self.output_dir, "processed.wav")
        self.vector_path = os.path.join(self.output_dir, "vectors.svg")
        self.laser_path = os.path.join(self.output_dir, "laser.ild")
        self.simulation_path = os.path.join(self.output_dir, "simulation.mp4")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    @patch.object(RAVEModel, "encode")
    @patch.object(RAVEModel, "decode")
    def test_end_to_end_pipeline(self, mock_decode, mock_encode):
        """Test the end-to-end pipeline from solar data to laser control."""
        # Mock the RAVE model's encode and decode methods
        mock_latent = torch.randn(1, 16, 100)
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        mock_encode.return_value = mock_latent
        mock_decode.return_value = mock_audio
        
        # 1. Data Collection
        # Create a mock storage
        mock_storage = MagicMock()
        
        # Create a collector with the mock storage
        collector = SolarDataCollector(storage=mock_storage)
        
        # Store the solar data
        for data in self.solar_data:
            collector.store_data(data)
        
        # Assert that the storage.store method was called for each data point
        self.assertEqual(mock_storage.store.call_count, len(self.solar_data))
        
        # 2. Audio Conversion
        # Create a synthesizer
        synthesizer = DirectMappingSynthesizer(self.audio_params)
        
        # Create a converter with the synthesizer
        converter = SolarToAudioConverter(synthesizer=synthesizer)
        
        # Convert the solar data to audio
        audio = converter.convert_multiple(self.solar_data)
        
        # Save the audio to a file
        converter.save_to_file(audio, self.audio_path)
        
        # Assert that the audio file was created
        self.assertTrue(os.path.exists(self.audio_path))
        
        # 3. RAVE Processing
        # Create a mock RAVE model
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_latent
        mock_model.decode.return_value = mock_audio
        
        # Create a processor with the mock model
        processor = RAVEProcessor(model=mock_model)
        
        # Process the audio
        latent, processed_audio = processor.process_audio(audio)
        
        # Save the latent representation and processed audio
        processor.save_latent(latent, self.latent_path)
        converter.save_to_file(processed_audio, self.processed_audio_path)
        
        # Assert that the latent file and processed audio file were created
        self.assertTrue(os.path.exists(self.latent_path))
        self.assertTrue(os.path.exists(self.processed_audio_path))
        
        # 4. Vector Generation
        # Create a mapping
        mapping = DirectMapping()
        
        # Create a generator with the mapping
        generator = VectorGenerator(mapping=mapping)
        
        # Generate vectors from the latent representation
        vectors = generator.generate_vectors(latent)
        
        # Smooth and normalize the vectors
        vectors = generator.smooth_vectors(vectors, smoothing_factor=self.vector_params.smoothing_factor)
        vectors = generator.normalize_vectors(vectors, range_min=self.vector_params.normalization_range[0], range_max=self.vector_params.normalization_range[1])
        
        # Save the vectors to an SVG file
        generator.save_to_svg(vectors, self.vector_path)
        
        # Assert that the SVG file was created
        self.assertTrue(os.path.exists(self.vector_path))
        
        # 5. Laser Control
        # Create a simulation controller
        controller = SimulationController()
        
        # Convert the SVG file to ILDA format
        ilda_file = convert_svg_to_ilda(self.vector_path)
        
        # Save the ILDA file
        ilda_file.save(self.laser_path)
        
        # Assert that the ILDA file was created
        self.assertTrue(os.path.exists(self.laser_path))
        
        # Play the ILDA file
        controller.play_ilda_file(ilda_file)
        
        # Save the simulation
        controller.save_simulation(self.simulation_path)
        
        # Assert that the simulation file was created
        self.assertTrue(os.path.exists(self.simulation_path))


@pytest.mark.integration
class TestDataToAudio(unittest.TestCase):
    """Integration tests for the data collection to audio conversion pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test data
        self.solar_data = [
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
        
        # Set up audio parameters
        self.audio_params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=lambda v: 220 + 660 * (v / 24.0),
            amplitude_mapping=lambda c: min(0.9, c / 10.0),
            timbre_mapping=lambda p: [1.0, 0.5, 0.25, 0.125],
            temporal_mapping=lambda values: values
        )
        
        # Create output file paths
        self.audio_path = os.path.join(self.output_dir, "audio.wav")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_data_to_audio_pipeline(self):
        """Test the pipeline from data collection to audio conversion."""
        # 1. Data Collection
        # Create a mock storage
        mock_storage = MagicMock()
        
        # Create a collector with the mock storage
        collector = SolarDataCollector(storage=mock_storage)
        
        # Store the solar data
        for data in self.solar_data:
            collector.store_data(data)
        
        # Assert that the storage.store method was called for each data point
        self.assertEqual(mock_storage.store.call_count, len(self.solar_data))
        
        # 2. Audio Conversion
        # Create a synthesizer
        synthesizer = DirectMappingSynthesizer(self.audio_params)
        
        # Create a converter with the synthesizer
        converter = SolarToAudioConverter(synthesizer=synthesizer)
        
        # Convert the solar data to audio
        audio = converter.convert_multiple(self.solar_data)
        
        # Save the audio to a file
        converter.save_to_file(audio, self.audio_path)
        
        # Assert that the audio file was created
        self.assertTrue(os.path.exists(self.audio_path))
        
        # Assert that the audio has the correct sample rate
        import wave
        with wave.open(self.audio_path, "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), self.audio_params.sample_rate)


@pytest.mark.integration
class TestAudioToVector(unittest.TestCase):
    """Integration tests for the audio conversion to vector generation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test audio
        self.audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        
        # Create output file paths
        self.audio_path = os.path.join(self.output_dir, "audio.wav")
        self.latent_path = os.path.join(self.output_dir, "latent.pt")
        self.processed_audio_path = os.path.join(self.output_dir, "processed.wav")
        self.vector_path = os.path.join(self.output_dir, "vectors.svg")
        
        # Save the audio to a file
        import wave
        import struct
        with wave.open(self.audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            for sample in self.audio:
                wav_file.writeframes(struct.pack("h", int(sample * 32767)))
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    @patch.object(RAVEModel, "encode")
    @patch.object(RAVEModel, "decode")
    def test_audio_to_vector_pipeline(self, mock_decode, mock_encode):
        """Test the pipeline from audio conversion to vector generation."""
        # Mock the RAVE model's encode and decode methods
        mock_latent = torch.randn(1, 16, 100)
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        mock_encode.return_value = mock_latent
        mock_decode.return_value = mock_audio
        
        # 1. RAVE Processing
        # Create a mock RAVE model
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_latent
        mock_model.decode.return_value = mock_audio
        
        # Create a processor with the mock model
        processor = RAVEProcessor(model=mock_model)
        
        # Process the audio
        latent, processed_audio = processor.process_audio(self.audio)
        
        # Save the latent representation and processed audio
        processor.save_latent(latent, self.latent_path)
        
        # Create a converter
        converter = SolarToAudioConverter()
        
        # Save the processed audio to a file
        converter.save_to_file(processed_audio, self.processed_audio_path)
        
        # Assert that the latent file and processed audio file were created
        self.assertTrue(os.path.exists(self.latent_path))
        self.assertTrue(os.path.exists(self.processed_audio_path))
        
        # 2. Vector Generation
        # Create a mapping
        mapping = DirectMapping()
        
        # Create a generator with the mapping
        generator = VectorGenerator(mapping=mapping)
        
        # Generate vectors from the latent representation
        vectors = generator.generate_vectors(latent)
        
        # Save the vectors to an SVG file
        generator.save_to_svg(vectors, self.vector_path)
        
        # Assert that the SVG file was created
        self.assertTrue(os.path.exists(self.vector_path))


@pytest.mark.integration
class TestVectorToLaser(unittest.TestCase):
    """Integration tests for the vector generation to laser control pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test vectors
        self.vectors = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        
        # Create output file paths
        self.vector_path = os.path.join(self.output_dir, "vectors.svg")
        self.laser_path = os.path.join(self.output_dir, "laser.ild")
        self.simulation_path = os.path.join(self.output_dir, "simulation.mp4")
        
        # Create a generator
        self.generator = VectorGenerator()
        
        # Save the vectors to an SVG file
        self.generator.save_to_svg(self.vectors, self.vector_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_vector_to_laser_pipeline(self):
        """Test the pipeline from vector generation to laser control."""
        # 1. Laser Control
        # Create a simulation controller
        controller = SimulationController()
        
        # Convert the SVG file to ILDA format
        ilda_file = convert_svg_to_ilda(self.vector_path)
        
        # Save the ILDA file
        ilda_file.save(self.laser_path)
        
        # Assert that the ILDA file was created
        self.assertTrue(os.path.exists(self.laser_path))
        
        # Play the ILDA file
        controller.play_ilda_file(ilda_file)
        
        # Save the simulation
        controller.save_simulation(self.simulation_path)
        
        # Assert that the simulation file was created
        self.assertTrue(os.path.exists(self.simulation_path))


if __name__ == "__main__":
    unittest.main()