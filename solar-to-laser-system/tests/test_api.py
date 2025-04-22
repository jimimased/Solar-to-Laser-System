"""
Tests for the API endpoints.
"""

import unittest
import json
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.data_collection.api import api as data_collection_api
from src.audio_conversion.api import api as audio_conversion_api
from src.rave_processing.api import api as rave_processing_api
from src.vector_generation.api import api as vector_generation_api
from src.laser_control.api import api as laser_control_api
from src.deployment.api import api as deployment_api


class TestDataCollectionAPI(unittest.TestCase):
    """Tests for the data collection API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(data_collection_api)
    
    def test_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_get_status(self):
        """Test the status endpoint."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
    
    @patch("src.data_collection.api.collector")
    def test_post_data(self, mock_collector):
        """Test posting solar data."""
        # Mock the collector.collect_and_store method
        mock_collector.collect_and_store.return_value = MagicMock(
            timestamp=datetime.now(),
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Post data
        response = self.client.post(
            "/api/solar/data",
            json={
                "panel_id": "panel1"
            }
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("timestamp", response.json())
        self.assertIn("voltage", response.json())
        self.assertIn("current", response.json())
        self.assertIn("power", response.json())
    
    @patch("src.data_collection.api.storage")
    def test_get_data(self, mock_storage):
        """Test getting solar data."""
        # Mock the storage.query method
        mock_storage.query.return_value = [
            MagicMock(
                timestamp=datetime.now(),
                voltage=12.5,
                current=2.1,
                power=26.25,
                temperature=25.0,
                irradiance=800.0,
                metadata={"panel_id": "panel1"}
            )
        ]
        
        # Get data
        response = self.client.get(
            "/api/solar/data/2025-04-22T00:00:00Z/2025-04-22T23:59:59Z?panel_id=panel1"
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertEqual(len(response.json()), 1)
        self.assertIn("timestamp", response.json()[0])
        self.assertIn("voltage", response.json()[0])
        self.assertIn("current", response.json()[0])
        self.assertIn("power", response.json()[0])


class TestAudioConversionAPI(unittest.TestCase):
    """Tests for the audio conversion API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(audio_conversion_api)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_get_status(self):
        """Test the status endpoint."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
    
    @patch("src.audio_conversion.api.converter")
    @patch("src.data_collection.api.storage")
    def test_convert_audio(self, mock_storage, mock_converter):
        """Test converting solar data to audio."""
        # Mock the storage.query method
        mock_storage.query.return_value = [
            MagicMock(
                timestamp=datetime.now(),
                voltage=12.5,
                current=2.1,
                power=26.25,
                temperature=25.0,
                irradiance=800.0,
                metadata={"panel_id": "panel1"}
            )
        ]
        
        # Mock the converter.convert_multiple method
        import numpy as np
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        mock_converter.convert_multiple.return_value = mock_audio
        
        # Mock the converter.save_to_file method
        mock_converter.save_to_file.return_value = "/tmp/audio.wav"
        
        # Convert audio
        response = self.client.post(
            "/api/audio/convert",
            json={
                "start_time": "2025-04-22T00:00:00Z",
                "end_time": "2025-04-22T23:59:59Z",
                "panel_id": "panel1",
                "synthesizer": "direct"
            }
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("audio_file", response.json())
    
    @patch("src.audio_conversion.api.db")
    def test_get_audio_files(self, mock_db):
        """Test getting audio files."""
        # Mock the db.query method
        mock_db.query.return_value = [
            {
                "id": 1,
                "filename": "audio1.wav",
                "sample_rate": 44100,
                "duration": 1.0,
                "channels": 1,
                "solar_data_start": "2025-04-22T00:00:00Z",
                "solar_data_end": "2025-04-22T23:59:59Z",
                "parameters": {"synthesizer": "direct"},
                "storage_path": "/tmp/audio1.wav",
                "created_at": "2025-04-22T12:00:00Z"
            }
        ]
        
        # Get audio files
        response = self.client.get("/api/audio/files")
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertEqual(len(response.json()), 1)
        self.assertIn("id", response.json()[0])
        self.assertIn("filename", response.json()[0])
        self.assertIn("sample_rate", response.json()[0])


class TestRAVEProcessingAPI(unittest.TestCase):
    """Tests for the RAVE processing API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(rave_processing_api)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_get_status(self):
        """Test the status endpoint."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
    
    @patch("src.rave_processing.api.processor")
    @patch("src.audio_conversion.api.db")
    def test_process_audio(self, mock_db, mock_processor):
        """Test processing audio with RAVE."""
        # Mock the db.query method
        mock_db.query.return_value = [
            {
                "id": 1,
                "filename": "audio1.wav",
                "sample_rate": 44100,
                "duration": 1.0,
                "channels": 1,
                "solar_data_start": "2025-04-22T00:00:00Z",
                "solar_data_end": "2025-04-22T23:59:59Z",
                "parameters": {"synthesizer": "direct"},
                "storage_path": "/tmp/audio1.wav",
                "created_at": "2025-04-22T12:00:00Z"
            }
        ]
        
        # Mock the processor.process_and_save method
        import torch
        import numpy as np
        mock_latent = torch.randn(1, 16, 100)
        mock_audio = np.sin(np.linspace(0, 2 * np.pi, 44100))
        mock_processor.process_and_save.return_value = (mock_latent, mock_audio, "/tmp/latent.pt", "/tmp/processed.wav")
        
        # Process audio
        response = self.client.post(
            "/api/rave/process",
            json={
                "audio_file_id": 1
            }
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("latent_file", response.json())
        self.assertIn("processed_audio_file", response.json())
    
    @patch("src.rave_processing.api.db")
    def test_get_latent_files(self, mock_db):
        """Test getting latent files."""
        # Mock the db.query method
        mock_db.query.return_value = [
            {
                "id": 1,
                "filename": "latent1.pt",
                "audio_file_id": 1,
                "parameters": {"model": "rave_v2"},
                "storage_path": "/tmp/latent1.pt",
                "created_at": "2025-04-22T12:00:00Z"
            }
        ]
        
        # Get latent files
        response = self.client.get("/api/rave/latent_files")
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertEqual(len(response.json()), 1)
        self.assertIn("id", response.json()[0])
        self.assertIn("filename", response.json()[0])
        self.assertIn("audio_file_id", response.json()[0])


class TestVectorGenerationAPI(unittest.TestCase):
    """Tests for the vector generation API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(vector_generation_api)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_get_status(self):
        """Test the status endpoint."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
    
    @patch("src.vector_generation.api.generator")
    @patch("src.rave_processing.api.db")
    def test_generate_vectors(self, mock_db, mock_generator):
        """Test generating vectors from latent representation."""
        # Mock the db.query method
        mock_db.query.return_value = [
            {
                "id": 1,
                "filename": "latent1.pt",
                "audio_file_id": 1,
                "parameters": {"model": "rave_v2"},
                "storage_path": "/tmp/latent1.pt",
                "created_at": "2025-04-22T12:00:00Z"
            }
        ]
        
        # Mock the generator.generate_and_save method
        import torch
        import numpy as np
        mock_latent = torch.randn(1, 16, 100)
        mock_vectors = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        mock_generator.generate_and_save.return_value = (mock_vectors, "/tmp/vectors.svg")
        
        # Generate vectors
        response = self.client.post(
            "/api/vector/generate",
            json={
                "latent_file_id": 1,
                "mapping": "direct",
                "smoothing_factor": 0.5,
                "normalization_range": [-1.0, 1.0]
            }
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("vector_file", response.json())
    
    @patch("src.vector_generation.api.db")
    def test_get_vector_files(self, mock_db):
        """Test getting vector files."""
        # Mock the db.query method
        mock_db.query.return_value = [
            {
                "id": 1,
                "filename": "vectors1.svg",
                "format": "svg",
                "point_count": 100,
                "latent_file_id": 1,
                "parameters": {"mapping": "direct"},
                "storage_path": "/tmp/vectors1.svg",
                "created_at": "2025-04-22T12:00:00Z"
            }
        ]
        
        # Get vector files
        response = self.client.get("/api/vector/files")
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertEqual(len(response.json()), 1)
        self.assertIn("id", response.json()[0])
        self.assertIn("filename", response.json()[0])
        self.assertIn("format", response.json()[0])


class TestLaserControlAPI(unittest.TestCase):
    """Tests for the laser control API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(laser_control_api)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_get_status(self):
        """Test the status endpoint."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
    
    @patch("src.laser_control.api.controller")
    @patch("src.vector_generation.api.db")
    def test_generate_laser_file(self, mock_db, mock_controller):
        """Test generating a laser file from vector graphics."""
        # Mock the db.query method
        mock_db.query.return_value = [
            {
                "id": 1,
                "filename": "vectors1.svg",
                "format": "svg",
                "point_count": 100,
                "latent_file_id": 1,
                "parameters": {"mapping": "direct"},
                "storage_path": "/tmp/vectors1.svg",
                "created_at": "2025-04-22T12:00:00Z"
            }
        ]
        
        # Mock the controller.convert_svg_to_ilda method
        mock_controller.convert_svg_to_ilda.return_value = "/tmp/laser.ild"
        
        # Generate laser file
        response = self.client.post(
            "/api/laser/generate",
            json={
                "vector_file_id": 1,
                "format": "ILDA",
                "frame_rate": 30,
                "points_per_frame": 500,
                "color_mode": "RGB",
                "intensity": 0.8
            }
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("laser_file", response.json())
    
    @patch("src.laser_control.api.db")
    def test_get_laser_files(self, mock_db):
        """Test getting laser files."""
        # Mock the db.query method
        mock_db.query.return_value = [
            {
                "id": 1,
                "filename": "laser1.ild",
                "format": "ILDA",
                "frame_count": 30,
                "vector_file_id": 1,
                "parameters": {"frame_rate": 30},
                "storage_path": "/tmp/laser1.ild",
                "created_at": "2025-04-22T12:00:00Z"
            }
        ]
        
        # Get laser files
        response = self.client.get("/api/laser/files")
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertEqual(len(response.json()), 1)
        self.assertIn("id", response.json()[0])
        self.assertIn("filename", response.json()[0])
        self.assertIn("format", response.json()[0])


class TestDeploymentAPI(unittest.TestCase):
    """Tests for the deployment API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(deployment_api)
    
    def test_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_get_status(self):
        """Test the status endpoint."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
    
    @patch("src.deployment.api.KubernetesDeployer")
    def test_deploy_to_kubernetes(self, mock_deployer):
        """Test deploying to Kubernetes."""
        # Mock the KubernetesDeployer.deploy_all method
        mock_deployer.return_value.deploy_all.return_value = True
        
        # Deploy to Kubernetes
        response = self.client.post(
            "/api/deployment/kubernetes",
            json={
                "namespace": "test-namespace",
                "registry": "ghcr.io/testuser"
            }
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "pending")
    
    @patch("src.deployment.api.DockerDeployer")
    def test_build_docker_images(self, mock_deployer):
        """Test building Docker images."""
        # Mock the DockerDeployer.build_and_push_all method
        mock_deployer.return_value.build_and_push_all.return_value = True
        
        # Build Docker images
        response = self.client.post(
            "/api/deployment/docker",
            json={
                "registry": "ghcr.io/testuser",
                "tag": "latest"
            }
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "pending")
    
    @patch("src.deployment.api.DockerComposeDeployer")
    def test_docker_compose_up(self, mock_deployer):
        """Test starting services with Docker Compose."""
        # Mock the DockerComposeDeployer.up method
        mock_deployer.return_value.up.return_value = True
        
        # Start services
        response = self.client.post("/api/deployment/docker-compose/up")
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "pending")


if __name__ == "__main__":
    unittest.main()