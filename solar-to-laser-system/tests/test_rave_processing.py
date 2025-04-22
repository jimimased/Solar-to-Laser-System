"""
Tests for the RAVE processing module.
"""

import unittest
import numpy as np
import os
import tempfile
import torch
from unittest.mock import patch, MagicMock

from src.rave_processing.model import RAVEModel
from src.rave_processing.processor import RAVEProcessor


class TestRAVEModel(unittest.TestCase):
    """Tests for the RAVEModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "model.pt")
        
        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = torch.randn(1, 16, 100)  # Mock latent representation
        self.mock_model.decode.return_value = torch.randn(1, 1, 44100)  # Mock audio output
        
        # Save the mock model
        torch.save({"model": self.mock_model}, self.model_path)
        
        # Initialize the RAVE model
        self.rave_model = RAVEModel(model_path=self.model_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    @patch("torch.load")
    def test_load_model(self, mock_torch_load):
        """Test loading the RAVE model."""
        # Mock torch.load to return a mock model
        mock_torch_load.return_value = {"model": self.mock_model}
        
        # Load the model
        model = RAVEModel(model_path=self.model_path)
        
        # Assert that torch.load was called with the correct path
        mock_torch_load.assert_called_once_with(self.model_path, map_location="cuda")
        
        # Assert that the model was loaded correctly
        self.assertEqual(model.model, self.mock_model)
    
    def test_encode(self):
        """Test encoding audio with the RAVE model."""
        # Create a mock audio input
        audio = np.random.randn(44100)
        
        # Mock the model's encode method
        mock_latent = torch.randn(1, 16, 100)
        self.rave_model.model.encode.return_value = mock_latent
        
        # Encode the audio
        latent = self.rave_model.encode(audio)
        
        # Assert that the model's encode method was called
        self.rave_model.model.encode.assert_called_once()
        
        # Assert that the latent representation has the correct shape
        self.assertEqual(latent.shape, (1, 16, 100))
    
    def test_decode(self):
        """Test decoding latent representation with the RAVE model."""
        # Create a mock latent representation
        latent = torch.randn(1, 16, 100)
        
        # Mock the model's decode method
        mock_audio = torch.randn(1, 1, 44100)
        self.rave_model.model.decode.return_value = mock_audio
        
        # Decode the latent representation
        audio = self.rave_model.decode(latent)
        
        # Assert that the model's decode method was called
        self.rave_model.model.decode.assert_called_once()
        
        # Assert that the audio has the correct shape
        self.assertEqual(audio.shape, (44100,))
    
    def test_encode_decode(self):
        """Test encoding and decoding with the RAVE model."""
        # Create a mock audio input
        audio = np.random.randn(44100)
        
        # Mock the model's encode and decode methods
        mock_latent = torch.randn(1, 16, 100)
        mock_audio = torch.randn(1, 1, 44100)
        self.rave_model.model.encode.return_value = mock_latent
        self.rave_model.model.decode.return_value = mock_audio
        
        # Encode and decode the audio
        latent = self.rave_model.encode(audio)
        reconstructed = self.rave_model.decode(latent)
        
        # Assert that the model's encode and decode methods were called
        self.rave_model.model.encode.assert_called_once()
        self.rave_model.model.decode.assert_called_once()
        
        # Assert that the reconstructed audio has the correct shape
        self.assertEqual(reconstructed.shape, (44100,))


class TestRAVEProcessor(unittest.TestCase):
    """Tests for the RAVEProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock RAVE model
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = torch.randn(1, 16, 100)  # Mock latent representation
        self.mock_model.decode.return_value = np.random.randn(44100)  # Mock audio output
        
        # Initialize the RAVE processor
        self.processor = RAVEProcessor(model=self.mock_model)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_process_audio(self):
        """Test processing audio with the RAVE processor."""
        # Create a mock audio input
        audio = np.random.randn(44100)
        
        # Process the audio
        latent, processed = self.processor.process_audio(audio)
        
        # Assert that the model's encode and decode methods were called
        self.mock_model.encode.assert_called_once()
        self.mock_model.decode.assert_called_once()
        
        # Assert that the latent representation and processed audio have the correct shapes
        self.assertEqual(latent.shape, (1, 16, 100))
        self.assertEqual(processed.shape, (44100,))
    
    def test_save_latent(self):
        """Test saving latent representation to a file."""
        # Create a mock latent representation
        latent = torch.randn(1, 16, 100)
        
        # Save the latent representation
        output_path = os.path.join(self.output_dir, "latent.pt")
        self.processor.save_latent(latent, output_path)
        
        # Assert that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load the saved latent representation
        loaded_latent = torch.load(output_path)
        
        # Assert that the loaded latent representation has the correct shape
        self.assertEqual(loaded_latent.shape, (1, 16, 100))
    
    def test_load_latent(self):
        """Test loading latent representation from a file."""
        # Create a mock latent representation
        latent = torch.randn(1, 16, 100)
        
        # Save the latent representation
        output_path = os.path.join(self.output_dir, "latent.pt")
        torch.save(latent, output_path)
        
        # Load the latent representation
        loaded_latent = self.processor.load_latent(output_path)
        
        # Assert that the loaded latent representation has the correct shape
        self.assertEqual(loaded_latent.shape, (1, 16, 100))
    
    def test_process_and_save(self):
        """Test processing audio and saving the results."""
        # Create a mock audio input
        audio = np.random.randn(44100)
        
        # Mock the model's encode and decode methods
        mock_latent = torch.randn(1, 16, 100)
        mock_audio = np.random.randn(44100)
        self.mock_model.encode.return_value = mock_latent
        self.mock_model.decode.return_value = mock_audio
        
        # Process the audio and save the results
        latent_path = os.path.join(self.output_dir, "latent.pt")
        audio_path = os.path.join(self.output_dir, "processed.wav")
        self.processor.process_and_save(audio, latent_path, audio_path)
        
        # Assert that the model's encode and decode methods were called
        self.mock_model.encode.assert_called_once()
        self.mock_model.decode.assert_called_once()
        
        # Assert that the files were created
        self.assertTrue(os.path.exists(latent_path))
        self.assertTrue(os.path.exists(audio_path))


if __name__ == "__main__":
    unittest.main()