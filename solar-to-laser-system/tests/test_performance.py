"""
Performance tests for the Solar-to-Laser System.
"""

import unittest
import os
import tempfile
import time
import numpy as np
import torch
from datetime import datetime, timedelta
import random
from unittest.mock import patch, MagicMock

import pytest

from src.common.data_structures import SolarData, AudioParameters, VectorParameters, LaserParameters
from src.data_collection.collector import SolarDataCollector
from src.data_collection.storage import InfluxDBStorage
from src.audio_conversion.converter import SolarToAudioConverter
from src.audio_conversion.synthesis import DirectMappingSynthesizer, FMSynthesizer, GranularSynthesizer
from src.rave_processing.model import RAVEModel
from src.rave_processing.processor import RAVEProcessor
from src.vector_generation.generator import VectorGenerator
from src.vector_generation.mapping import DirectMapping, PCAMapping, TSNEMapping
from src.laser_control.controller import LaserController, SimulationController
from src.laser_control.ilda import ILDAFile, convert_svg_to_ilda


@pytest.mark.performance
class TestDataCollectionPerformance(unittest.TestCase):
    """Performance tests for the data collection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock storage
        self.mock_storage = MagicMock()
        
        # Create a collector with the mock storage
        self.collector = SolarDataCollector(storage=self.mock_storage)
        
        # Create test data
        self.small_dataset = [
            SolarData(
                timestamp=datetime.now() + timedelta(seconds=i),
                voltage=12.0 + random.uniform(-0.5, 0.5),
                current=2.0 + random.uniform(-0.2, 0.2),
                power=24.0 + random.uniform(-1.0, 1.0),
                temperature=25.0 + random.uniform(-0.5, 0.5),
                irradiance=800.0 + random.uniform(-20.0, 20.0),
                metadata={"panel_id": "panel1"}
            )
            for i in range(100)
        ]
        
        self.medium_dataset = [
            SolarData(
                timestamp=datetime.now() + timedelta(seconds=i),
                voltage=12.0 + random.uniform(-0.5, 0.5),
                current=2.0 + random.uniform(-0.2, 0.2),
                power=24.0 + random.uniform(-1.0, 1.0),
                temperature=25.0 + random.uniform(-0.5, 0.5),
                irradiance=800.0 + random.uniform(-20.0, 20.0),
                metadata={"panel_id": "panel1"}
            )
            for i in range(1000)
        ]
        
        self.large_dataset = [
            SolarData(
                timestamp=datetime.now() + timedelta(seconds=i),
                voltage=12.0 + random.uniform(-0.5, 0.5),
                current=2.0 + random.uniform(-0.2, 0.2),
                power=24.0 + random.uniform(-1.0, 1.0),
                temperature=25.0 + random.uniform(-0.5, 0.5),
                irradiance=800.0 + random.uniform(-20.0, 20.0),
                metadata={"panel_id": "panel1"}
            )
            for i in range(10000)
        ]
    
    def test_store_data_small(self):
        """Test storing a small dataset."""
        # Measure the time to store the data
        start_time = time.time()
        
        for data in self.small_dataset:
            self.collector.store_data(data)
        
        end_time = time.time()
        
        # Calculate the time per data point
        total_time = end_time - start_time
        time_per_point = total_time / len(self.small_dataset)
        
        # Assert that the time per data point is below a threshold
        self.assertLess(time_per_point, 0.001)  # 1 millisecond per data point
        
        # Print the performance metrics
        print(f"Small dataset ({len(self.small_dataset)} points): {total_time:.4f} seconds, {time_per_point * 1000:.4f} ms per point")
    
    def test_store_data_medium(self):
        """Test storing a medium dataset."""
        # Measure the time to store the data
        start_time = time.time()
        
        for data in self.medium_dataset:
            self.collector.store_data(data)
        
        end_time = time.time()
        
        # Calculate the time per data point
        total_time = end_time - start_time
        time_per_point = total_time / len(self.medium_dataset)
        
        # Assert that the time per data point is below a threshold
        self.assertLess(time_per_point, 0.001)  # 1 millisecond per data point
        
        # Print the performance metrics
        print(f"Medium dataset ({len(self.medium_dataset)} points): {total_time:.4f} seconds, {time_per_point * 1000:.4f} ms per point")
    
    def test_store_data_large(self):
        """Test storing a large dataset."""
        # Measure the time to store the data
        start_time = time.time()
        
        for data in self.large_dataset:
            self.collector.store_data(data)
        
        end_time = time.time()
        
        # Calculate the time per data point
        total_time = end_time - start_time
        time_per_point = total_time / len(self.large_dataset)
        
        # Assert that the time per data point is below a threshold
        self.assertLess(time_per_point, 0.001)  # 1 millisecond per data point
        
        # Print the performance metrics
        print(f"Large dataset ({len(self.large_dataset)} points): {total_time:.4f} seconds, {time_per_point * 1000:.4f} ms per point")


@pytest.mark.performance
class TestAudioConversionPerformance(unittest.TestCase):
    """Performance tests for the audio conversion module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Set up audio parameters
        self.audio_params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=lambda v: 220 + 660 * (v / 24.0),
            amplitude_mapping=lambda c: min(0.9, c / 10.0),
            timbre_mapping=lambda p: [1.0, 0.5, 0.25, 0.125],
            temporal_mapping=lambda values: values
        )
        
        # Create synthesizers
        self.direct_synthesizer = DirectMappingSynthesizer(self.audio_params)
        self.fm_synthesizer = FMSynthesizer(self.audio_params)
        self.granular_synthesizer = GranularSynthesizer(self.audio_params)
        
        # Create converters
        self.direct_converter = SolarToAudioConverter(synthesizer=self.direct_synthesizer)
        self.fm_converter = SolarToAudioConverter(synthesizer=self.fm_synthesizer)
        self.granular_converter = SolarToAudioConverter(synthesizer=self.granular_synthesizer)
        
        # Create test data
        self.solar_data = SolarData(
            timestamp=datetime.now(),
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Create output file paths
        self.audio_path = os.path.join(self.output_dir, "audio.wav")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_direct_mapping_performance(self):
        """Test the performance of the direct mapping synthesizer."""
        # Measure the time to convert the data
        start_time = time.time()
        
        audio = self.direct_converter.convert(self.solar_data)
        
        end_time = time.time()
        
        # Calculate the conversion time
        conversion_time = end_time - start_time
        
        # Assert that the conversion time is below a threshold
        self.assertLess(conversion_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Direct mapping conversion: {conversion_time * 1000:.4f} ms")
    
    def test_fm_synthesis_performance(self):
        """Test the performance of the FM synthesizer."""
        # Measure the time to convert the data
        start_time = time.time()
        
        audio = self.fm_converter.convert(self.solar_data)
        
        end_time = time.time()
        
        # Calculate the conversion time
        conversion_time = end_time - start_time
        
        # Assert that the conversion time is below a threshold
        self.assertLess(conversion_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"FM synthesis conversion: {conversion_time * 1000:.4f} ms")
    
    def test_granular_synthesis_performance(self):
        """Test the performance of the granular synthesizer."""
        # Measure the time to convert the data
        start_time = time.time()
        
        audio = self.granular_converter.convert(self.solar_data)
        
        end_time = time.time()
        
        # Calculate the conversion time
        conversion_time = end_time - start_time
        
        # Assert that the conversion time is below a threshold
        self.assertLess(conversion_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Granular synthesis conversion: {conversion_time * 1000:.4f} ms")
    
    def test_save_to_file_performance(self):
        """Test the performance of saving audio to a file."""
        # Create audio data
        audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        
        # Measure the time to save the audio
        start_time = time.time()
        
        self.direct_converter.save_to_file(audio, self.audio_path)
        
        end_time = time.time()
        
        # Calculate the save time
        save_time = end_time - start_time
        
        # Assert that the save time is below a threshold
        self.assertLess(save_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Save to file: {save_time * 1000:.4f} ms")


@pytest.mark.performance
class TestRAVEProcessingPerformance(unittest.TestCase):
    """Performance tests for the RAVE processing module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test audio
        self.short_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))  # 1 second
        self.medium_audio = np.sin(np.linspace(0, 2 * np.pi, 44100 * 10))  # 10 seconds
        self.long_audio = np.sin(np.linspace(0, 2 * np.pi, 44100 * 60))  # 60 seconds
        
        # Create output file paths
        self.latent_path = os.path.join(self.output_dir, "latent.pt")
        self.processed_audio_path = os.path.join(self.output_dir, "processed.wav")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    @patch.object(RAVEModel, "encode")
    @patch.object(RAVEModel, "decode")
    def test_process_short_audio(self, mock_decode, mock_encode):
        """Test processing short audio."""
        # Mock the RAVE model's encode and decode methods
        mock_latent = torch.randn(1, 16, 100)
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        mock_encode.return_value = mock_latent
        mock_decode.return_value = mock_audio
        
        # Create a mock RAVE model
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_latent
        mock_model.decode.return_value = mock_audio
        
        # Create a processor with the mock model
        processor = RAVEProcessor(model=mock_model)
        
        # Measure the time to process the audio
        start_time = time.time()
        
        latent, processed_audio = processor.process_audio(self.short_audio)
        
        end_time = time.time()
        
        # Calculate the processing time
        processing_time = end_time - start_time
        
        # Assert that the processing time is below a threshold
        self.assertLess(processing_time, 0.5)  # 500 milliseconds
        
        # Print the performance metrics
        print(f"Short audio processing: {processing_time * 1000:.4f} ms")
    
    @patch.object(RAVEModel, "encode")
    @patch.object(RAVEModel, "decode")
    def test_process_medium_audio(self, mock_decode, mock_encode):
        """Test processing medium audio."""
        # Mock the RAVE model's encode and decode methods
        mock_latent = torch.randn(1, 16, 1000)
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100 * 10))
        mock_encode.return_value = mock_latent
        mock_decode.return_value = mock_audio
        
        # Create a mock RAVE model
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_latent
        mock_model.decode.return_value = mock_audio
        
        # Create a processor with the mock model
        processor = RAVEProcessor(model=mock_model)
        
        # Measure the time to process the audio
        start_time = time.time()
        
        latent, processed_audio = processor.process_audio(self.medium_audio)
        
        end_time = time.time()
        
        # Calculate the processing time
        processing_time = end_time - start_time
        
        # Assert that the processing time is below a threshold
        self.assertLess(processing_time, 5.0)  # 5 seconds
        
        # Print the performance metrics
        print(f"Medium audio processing: {processing_time:.4f} seconds")
    
    @patch.object(RAVEModel, "encode")
    @patch.object(RAVEModel, "decode")
    def test_process_long_audio(self, mock_decode, mock_encode):
        """Test processing long audio."""
        # Mock the RAVE model's encode and decode methods
        mock_latent = torch.randn(1, 16, 6000)
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100 * 60))
        mock_encode.return_value = mock_latent
        mock_decode.return_value = mock_audio
        
        # Create a mock RAVE model
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_latent
        mock_model.decode.return_value = mock_audio
        
        # Create a processor with the mock model
        processor = RAVEProcessor(model=mock_model)
        
        # Measure the time to process the audio
        start_time = time.time()
        
        latent, processed_audio = processor.process_audio(self.long_audio)
        
        end_time = time.time()
        
        # Calculate the processing time
        processing_time = end_time - start_time
        
        # Assert that the processing time is below a threshold
        self.assertLess(processing_time, 30.0)  # 30 seconds
        
        # Print the performance metrics
        print(f"Long audio processing: {processing_time:.4f} seconds")


@pytest.mark.performance
class TestVectorGenerationPerformance(unittest.TestCase):
    """Performance tests for the vector generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test latent representations
        self.small_latent = torch.randn(1, 16, 100)
        self.medium_latent = torch.randn(1, 16, 1000)
        self.large_latent = torch.randn(1, 16, 10000)
        
        # Create mappings
        self.direct_mapping = DirectMapping()
        self.pca_mapping = PCAMapping(n_components=2)
        self.tsne_mapping = TSNEMapping(n_components=2)
        
        # Create generators
        self.direct_generator = VectorGenerator(mapping=self.direct_mapping)
        self.pca_generator = VectorGenerator(mapping=self.pca_mapping)
        self.tsne_generator = VectorGenerator(mapping=self.tsne_mapping)
        
        # Create output file paths
        self.vector_path = os.path.join(self.output_dir, "vectors.svg")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_direct_mapping_performance(self):
        """Test the performance of the direct mapping."""
        # Measure the time to generate vectors
        start_time = time.time()
        
        vectors = self.direct_generator.generate_vectors(self.small_latent)
        
        end_time = time.time()
        
        # Calculate the generation time
        generation_time = end_time - start_time
        
        # Assert that the generation time is below a threshold
        self.assertLess(generation_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Direct mapping generation: {generation_time * 1000:.4f} ms")
    
    def test_pca_mapping_performance(self):
        """Test the performance of the PCA mapping."""
        # Measure the time to generate vectors
        start_time = time.time()
        
        vectors = self.pca_generator.generate_vectors(self.small_latent)
        
        end_time = time.time()
        
        # Calculate the generation time
        generation_time = end_time - start_time
        
        # Assert that the generation time is below a threshold
        self.assertLess(generation_time, 0.5)  # 500 milliseconds
        
        # Print the performance metrics
        print(f"PCA mapping generation: {generation_time * 1000:.4f} ms")
    
    def test_tsne_mapping_performance(self):
        """Test the performance of the t-SNE mapping."""
        # Measure the time to generate vectors
        start_time = time.time()
        
        vectors = self.tsne_generator.generate_vectors(self.small_latent)
        
        end_time = time.time()
        
        # Calculate the generation time
        generation_time = end_time - start_time
        
        # Assert that the generation time is below a threshold
        self.assertLess(generation_time, 5.0)  # 5 seconds
        
        # Print the performance metrics
        print(f"t-SNE mapping generation: {generation_time:.4f} seconds")
    
    def test_save_to_svg_performance(self):
        """Test the performance of saving vectors to an SVG file."""
        # Create vectors
        vectors = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        
        # Measure the time to save the vectors
        start_time = time.time()
        
        self.direct_generator.save_to_svg(vectors, self.vector_path)
        
        end_time = time.time()
        
        # Calculate the save time
        save_time = end_time - start_time
        
        # Assert that the save time is below a threshold
        self.assertLess(save_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Save to SVG: {save_time * 1000:.4f} ms")


@pytest.mark.performance
class TestLaserControlPerformance(unittest.TestCase):
    """Performance tests for the laser control module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test vectors
        self.small_vectors = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        
        self.medium_vectors = np.array([
            [x / 100, y / 100]
            for x in range(100)
            for y in range(100)
        ])
        
        self.large_vectors = np.array([
            [x / 1000, y / 1000]
            for x in range(1000)
            for y in range(1000)
        ])
        
        # Create a generator
        self.generator = VectorGenerator()
        
        # Create output file paths
        self.small_vector_path = os.path.join(self.output_dir, "small_vectors.svg")
        self.medium_vector_path = os.path.join(self.output_dir, "medium_vectors.svg")
        self.large_vector_path = os.path.join(self.output_dir, "large_vectors.svg")
        
        self.laser_path = os.path.join(self.output_dir, "laser.ild")
        self.simulation_path = os.path.join(self.output_dir, "simulation.mp4")
        
        # Save the vectors to SVG files
        self.generator.save_to_svg(self.small_vectors, self.small_vector_path)
        self.generator.save_to_svg(self.medium_vectors[:1000], self.medium_vector_path)  # Limit to 1000 points for medium
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_convert_svg_to_ilda_small(self):
        """Test converting a small SVG file to ILDA format."""
        # Measure the time to convert the SVG file
        start_time = time.time()
        
        ilda_file = convert_svg_to_ilda(self.small_vector_path)
        
        end_time = time.time()
        
        # Calculate the conversion time
        conversion_time = end_time - start_time
        
        # Assert that the conversion time is below a threshold
        self.assertLess(conversion_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Small SVG to ILDA conversion: {conversion_time * 1000:.4f} ms")
    
    def test_convert_svg_to_ilda_medium(self):
        """Test converting a medium SVG file to ILDA format."""
        # Measure the time to convert the SVG file
        start_time = time.time()
        
        ilda_file = convert_svg_to_ilda(self.medium_vector_path)
        
        end_time = time.time()
        
        # Calculate the conversion time
        conversion_time = end_time - start_time
        
        # Assert that the conversion time is below a threshold
        self.assertLess(conversion_time, 1.0)  # 1 second
        
        # Print the performance metrics
        print(f"Medium SVG to ILDA conversion: {conversion_time * 1000:.4f} ms")
    
    def test_save_ilda_file(self):
        """Test saving an ILDA file."""
        # Create an ILDA file
        ilda_file = convert_svg_to_ilda(self.small_vector_path)
        
        # Measure the time to save the ILDA file
        start_time = time.time()
        
        ilda_file.save(self.laser_path)
        
        end_time = time.time()
        
        # Calculate the save time
        save_time = end_time - start_time
        
        # Assert that the save time is below a threshold
        self.assertLess(save_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Save ILDA file: {save_time * 1000:.4f} ms")
    
    def test_play_ilda_file(self):
        """Test playing an ILDA file."""
        # Create an ILDA file
        ilda_file = convert_svg_to_ilda(self.small_vector_path)
        
        # Create a simulation controller
        controller = SimulationController()
        
        # Measure the time to play the ILDA file
        start_time = time.time()
        
        controller.play_ilda_file(ilda_file)
        
        end_time = time.time()
        
        # Calculate the play time
        play_time = end_time - start_time
        
        # Assert that the play time is below a threshold
        self.assertLess(play_time, 0.1)  # 100 milliseconds
        
        # Print the performance metrics
        print(f"Play ILDA file: {play_time * 1000:.4f} ms")
    
    def test_save_simulation(self):
        """Test saving a simulation."""
        # Create an ILDA file
        ilda_file = convert_svg_to_ilda(self.small_vector_path)
        
        # Create a simulation controller
        controller = SimulationController()
        
        # Play the ILDA file
        controller.play_ilda_file(ilda_file)
        
        # Measure the time to save the simulation
        start_time = time.time()
        
        controller.save_simulation(self.simulation_path)
        
        end_time = time.time()
        
        # Calculate the save time
        save_time = end_time - start_time
        
        # Assert that the save time is below a threshold
        self.assertLess(save_time, 1.0)  # 1 second
        
        # Print the performance metrics
        print(f"Save simulation: {save_time * 1000:.4f} ms")


if __name__ == "__main__":
    unittest.main()