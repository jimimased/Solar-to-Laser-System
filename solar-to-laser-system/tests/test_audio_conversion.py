"""
Tests for the audio conversion module.
"""

import unittest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.common.data_structures import SolarData, AudioParameters
from src.audio_conversion.converter import SolarToAudioConverter
from src.audio_conversion.synthesis import (
    DirectMappingSynthesizer,
    FMSynthesizer,
    GranularSynthesizer
)


class TestSolarToAudioConverter(unittest.TestCase):
    """Tests for the SolarToAudioConverter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_synthesizer = MagicMock()
        self.converter = SolarToAudioConverter(synthesizer=self.mock_synthesizer)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_convert_single_data_point(self):
        """Test converting a single solar data point to audio."""
        # Create a SolarData object
        data = SolarData(
            timestamp="2025-04-22T12:00:00Z",
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Mock the synthesizer to return a simple sine wave
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        self.mock_synthesizer.synthesize.return_value = mock_audio
        
        # Call the convert method
        result = self.converter.convert(data)
        
        # Assert that the synthesizer.synthesize method was called with the correct data
        self.mock_synthesizer.synthesize.assert_called_once()
        synth_args = self.mock_synthesizer.synthesize.call_args[0]
        self.assertEqual(synth_args[0], data)
        
        # Assert that the result is the mock audio
        np.testing.assert_array_equal(result, mock_audio)
    
    def test_convert_multiple_data_points(self):
        """Test converting multiple solar data points to audio."""
        # Create a list of SolarData objects
        data_points = [
            SolarData(
                timestamp="2025-04-22T12:00:00Z",
                voltage=12.5,
                current=2.1,
                power=26.25,
                temperature=25.0,
                irradiance=800.0,
                metadata={"panel_id": "panel1"}
            ),
            SolarData(
                timestamp="2025-04-22T12:01:00Z",
                voltage=12.6,
                current=2.2,
                power=27.72,
                temperature=25.1,
                irradiance=810.0,
                metadata={"panel_id": "panel1"}
            )
        ]
        
        # Mock the synthesizer to return simple sine waves
        mock_audio1 = np.sin(np.linspace(0, 2 * np.pi, 44100))
        mock_audio2 = np.sin(np.linspace(0, 4 * np.pi, 44100))
        self.mock_synthesizer.synthesize.side_effect = [mock_audio1, mock_audio2]
        
        # Call the convert_multiple method
        result = self.converter.convert_multiple(data_points)
        
        # Assert that the synthesizer.synthesize method was called twice with the correct data
        self.assertEqual(self.mock_synthesizer.synthesize.call_count, 2)
        self.assertEqual(self.mock_synthesizer.synthesize.call_args_list[0][0][0], data_points[0])
        self.assertEqual(self.mock_synthesizer.synthesize.call_args_list[1][0][0], data_points[1])
        
        # Assert that the result is the concatenation of the mock audio
        expected_result = np.concatenate([mock_audio1, mock_audio2])
        np.testing.assert_array_equal(result, expected_result)
    
    def test_save_to_file(self):
        """Test saving audio to a file."""
        # Create a simple sine wave
        audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        
        # Call the save_to_file method
        output_path = os.path.join(self.output_dir, "test.wav")
        self.converter.save_to_file(audio, output_path)
        
        # Assert that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Assert that the file is a valid WAV file
        import wave
        with wave.open(output_path, "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 44100)
            self.assertEqual(wav_file.getnframes(), 44100)
    
    def test_convert_and_save(self):
        """Test converting solar data and saving to a file."""
        # Create a SolarData object
        data = SolarData(
            timestamp="2025-04-22T12:00:00Z",
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Mock the synthesizer to return a simple sine wave
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        self.mock_synthesizer.synthesize.return_value = mock_audio
        
        # Call the convert_and_save method
        output_path = os.path.join(self.output_dir, "test.wav")
        self.converter.convert_and_save(data, output_path)
        
        # Assert that the synthesizer.synthesize method was called with the correct data
        self.mock_synthesizer.synthesize.assert_called_once()
        synth_args = self.mock_synthesizer.synthesize.call_args[0]
        self.assertEqual(synth_args[0], data)
        
        # Assert that the file was created
        self.assertTrue(os.path.exists(output_path))


class TestDirectMappingSynthesizer(unittest.TestCase):
    """Tests for the DirectMappingSynthesizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=lambda v: 220 + 660 * (v / 24.0),  # Map voltage to frequency
            amplitude_mapping=lambda c: min(0.9, c / 10.0),  # Map current to amplitude
            timbre_mapping=lambda p: [1.0, 0.5, 0.25, 0.125],  # Map power to harmonic amplitudes
            temporal_mapping=lambda values: values  # Identity mapping for temporal evolution
        )
        self.synthesizer = DirectMappingSynthesizer(self.params)
    
    def test_synthesize(self):
        """Test synthesizing audio from solar data."""
        # Create a SolarData object
        data = SolarData(
            timestamp="2025-04-22T12:00:00Z",
            voltage=12.0,
            current=5.0,
            power=60.0,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Call the synthesize method
        result = self.synthesizer.synthesize(data)
        
        # Assert that the result is a numpy array with the correct length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), int(self.params.sample_rate * self.params.duration))
        
        # Assert that the result is not all zeros
        self.assertFalse(np.all(result == 0))
        
        # Assert that the amplitude is within the expected range
        self.assertLessEqual(np.max(np.abs(result)), 0.9)


class TestFMSynthesizer(unittest.TestCase):
    """Tests for the FMSynthesizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=lambda v: 220 + 660 * (v / 24.0),  # Map voltage to frequency
            amplitude_mapping=lambda c: min(0.9, c / 10.0),  # Map current to amplitude
            timbre_mapping=lambda p: [1.0, 0.5, 0.25, 0.125],  # Map power to harmonic amplitudes
            temporal_mapping=lambda values: values  # Identity mapping for temporal evolution
        )
        self.synthesizer = FMSynthesizer(self.params)
    
    def test_synthesize(self):
        """Test synthesizing audio from solar data."""
        # Create a SolarData object
        data = SolarData(
            timestamp="2025-04-22T12:00:00Z",
            voltage=12.0,
            current=5.0,
            power=60.0,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Call the synthesize method
        result = self.synthesizer.synthesize(data)
        
        # Assert that the result is a numpy array with the correct length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), int(self.params.sample_rate * self.params.duration))
        
        # Assert that the result is not all zeros
        self.assertFalse(np.all(result == 0))
        
        # Assert that the amplitude is within the expected range
        self.assertLessEqual(np.max(np.abs(result)), 0.9)


class TestGranularSynthesizer(unittest.TestCase):
    """Tests for the GranularSynthesizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = AudioParameters(
            sample_rate=44100,
            duration=1.0,
            frequency_mapping=lambda v: 220 + 660 * (v / 24.0),  # Map voltage to frequency
            amplitude_mapping=lambda c: min(0.9, c / 10.0),  # Map current to amplitude
            timbre_mapping=lambda p: [1.0, 0.5, 0.25, 0.125],  # Map power to harmonic amplitudes
            temporal_mapping=lambda values: values  # Identity mapping for temporal evolution
        )
        self.synthesizer = GranularSynthesizer(self.params)
    
    def test_synthesize(self):
        """Test synthesizing audio from solar data."""
        # Create a SolarData object
        data = SolarData(
            timestamp="2025-04-22T12:00:00Z",
            voltage=12.0,
            current=5.0,
            power=60.0,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Call the synthesize method
        result = self.synthesizer.synthesize(data)
        
        # Assert that the result is a numpy array with the correct length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), int(self.params.sample_rate * self.params.duration))
        
        # Assert that the result is not all zeros
        self.assertFalse(np.all(result == 0))
        
        # Assert that the amplitude is within the expected range
        self.assertLessEqual(np.max(np.abs(result)), 0.9)


if __name__ == "__main__":
    unittest.main()