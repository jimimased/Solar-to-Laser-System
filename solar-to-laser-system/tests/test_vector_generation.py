"""
Tests for the vector generation module.
"""

import unittest
import numpy as np
import os
import tempfile
import torch
from unittest.mock import patch, MagicMock

from src.vector_generation.generator import VectorGenerator
from src.vector_generation.mapping import (
    DirectMapping,
    PCAMapping,
    TSNEMapping
)


class TestVectorGenerator(unittest.TestCase):
    """Tests for the VectorGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock mapping
        self.mock_mapping = MagicMock()
        self.mock_mapping.map.return_value = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        
        # Initialize the vector generator
        self.generator = VectorGenerator(mapping=self.mock_mapping)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_generate_vectors(self):
        """Test generating vectors from latent representation."""
        # Create a mock latent representation
        latent = torch.randn(1, 16, 100)
        
        # Generate vectors
        vectors = self.generator.generate_vectors(latent)
        
        # Assert that the mapping.map method was called with the correct latent representation
        self.mock_mapping.map.assert_called_once()
        np.testing.assert_array_equal(self.mock_mapping.map.call_args[0][0], latent.numpy())
        
        # Assert that the vectors have the correct shape
        self.assertEqual(vectors.shape, (5, 2))
    
    def test_smooth_vectors(self):
        """Test smoothing vectors."""
        # Create a mock vector path
        vectors = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0]
        ])
        
        # Smooth the vectors
        smoothed = self.generator.smooth_vectors(vectors, smoothing_factor=0.5)
        
        # Assert that the smoothed vectors have the correct shape
        self.assertEqual(smoothed.shape, vectors.shape)
        
        # Assert that the smoothed vectors are different from the original vectors
        self.assertFalse(np.array_equal(smoothed, vectors))
    
    def test_normalize_vectors(self):
        """Test normalizing vectors."""
        # Create a mock vector path
        vectors = np.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 0.0]
        ])
        
        # Normalize the vectors
        normalized = self.generator.normalize_vectors(vectors, range_min=-1.0, range_max=1.0)
        
        # Assert that the normalized vectors have the correct shape
        self.assertEqual(normalized.shape, vectors.shape)
        
        # Assert that the normalized vectors are within the specified range
        self.assertGreaterEqual(np.min(normalized), -1.0)
        self.assertLessEqual(np.max(normalized), 1.0)
    
    def test_save_to_svg(self):
        """Test saving vectors to SVG file."""
        # Create a mock vector path
        vectors = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0]
        ])
        
        # Save the vectors to SVG
        output_path = os.path.join(self.output_dir, "test.svg")
        self.generator.save_to_svg(vectors, output_path)
        
        # Assert that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Assert that the file is a valid SVG file
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn("<svg", content)
            self.assertIn("<path", content)
    
    def test_generate_and_save(self):
        """Test generating vectors and saving to file."""
        # Create a mock latent representation
        latent = torch.randn(1, 16, 100)
        
        # Mock the mapping to return a specific vector path
        self.mock_mapping.map.return_value = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0]
        ])
        
        # Generate vectors and save to file
        output_path = os.path.join(self.output_dir, "test.svg")
        self.generator.generate_and_save(latent, output_path)
        
        # Assert that the mapping.map method was called with the correct latent representation
        self.mock_mapping.map.assert_called_once()
        
        # Assert that the file was created
        self.assertTrue(os.path.exists(output_path))


class TestDirectMapping(unittest.TestCase):
    """Tests for the DirectMapping class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapping = DirectMapping()
    
    def test_map(self):
        """Test mapping latent representation to vectors."""
        # Create a mock latent representation
        latent = np.random.randn(1, 16, 100)
        
        # Map the latent representation to vectors
        vectors = self.mapping.map(latent)
        
        # Assert that the vectors have the correct shape
        self.assertEqual(len(vectors.shape), 2)
        self.assertEqual(vectors.shape[1], 2)  # X and Y coordinates
        
        # Assert that the number of vectors is related to the latent representation size
        self.assertGreater(vectors.shape[0], 0)


class TestPCAMapping(unittest.TestCase):
    """Tests for the PCAMapping class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapping = PCAMapping(n_components=2)
    
    def test_map(self):
        """Test mapping latent representation to vectors."""
        # Create a mock latent representation
        latent = np.random.randn(1, 16, 100)
        
        # Map the latent representation to vectors
        vectors = self.mapping.map(latent)
        
        # Assert that the vectors have the correct shape
        self.assertEqual(len(vectors.shape), 2)
        self.assertEqual(vectors.shape[1], 2)  # X and Y coordinates
        
        # Assert that the number of vectors is related to the latent representation size
        self.assertGreater(vectors.shape[0], 0)


class TestTSNEMapping(unittest.TestCase):
    """Tests for the TSNEMapping class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapping = TSNEMapping(n_components=2)
    
    def test_map(self):
        """Test mapping latent representation to vectors."""
        # Create a mock latent representation
        latent = np.random.randn(1, 16, 100)
        
        # Map the latent representation to vectors
        vectors = self.mapping.map(latent)
        
        # Assert that the vectors have the correct shape
        self.assertEqual(len(vectors.shape), 2)
        self.assertEqual(vectors.shape[1], 2)  # X and Y coordinates
        
        # Assert that the number of vectors is related to the latent representation size
        self.assertGreater(vectors.shape[0], 0)


if __name__ == "__main__":
    unittest.main()