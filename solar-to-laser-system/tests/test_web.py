"""
Tests for the web interface.
"""

import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import express
from socket.io import Client as SocketIOClient

from src.web.server import app, io


class TestWebServer(unittest.TestCase):
    """Tests for the web server."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test client
        self.client = app.test_client()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_root_route(self):
        """Test the root route."""
        # Make a request to the root route
        response = self.client.get("/")
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Assert that the response contains the expected content
        self.assertIn(b"<!DOCTYPE html>", response.data)
        self.assertIn(b"<title>Solar-to-Laser System</title>", response.data)
    
    @patch("axios.post")
    def test_api_proxy_solar_data(self, mock_post):
        """Test the API proxy for solar data."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data = {"timestamp": "2025-04-22T12:00:00Z", "voltage": 12.5, "current": 2.1, "power": 26.25}
        mock_post.return_value = mock_response
        
        # Make a request to the API proxy
        response = self.client.post(
            "/api/solar/data",
            json={"panel_id": "panel1"}
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Assert that axios.post was called with the correct arguments
        mock_post.assert_called_once()
        args = mock_post.call_args[0]
        kwargs = mock_post.call_args[1]
        self.assertEqual(args[0], "http://localhost:8000/api/solar/data")
        self.assertEqual(kwargs["data"], {"panel_id": "panel1"})
    
    @patch("axios.post")
    def test_api_proxy_audio_convert(self, mock_post):
        """Test the API proxy for audio conversion."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data = {"audio_file": "/tmp/audio.wav"}
        mock_post.return_value = mock_response
        
        # Make a request to the API proxy
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
        
        # Assert that axios.post was called with the correct arguments
        mock_post.assert_called_once()
        args = mock_post.call_args[0]
        kwargs = mock_post.call_args[1]
        self.assertEqual(args[0], "http://localhost:8001/api/audio/convert")
        self.assertEqual(kwargs["data"]["panel_id"], "panel1")
    
    @patch("axios.post")
    def test_api_proxy_rave_process(self, mock_post):
        """Test the API proxy for RAVE processing."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data = {"latent_file": "/tmp/latent.pt", "processed_audio_file": "/tmp/processed.wav"}
        mock_post.return_value = mock_response
        
        # Make a request to the API proxy
        response = self.client.post(
            "/api/rave/process",
            json={"audio_file_id": 1}
        )
        
        # Assert that the response is successful
        self.assertEqual(response.status_code, 200)
        
        # Assert that axios.post was called with the correct arguments
        mock_post.assert_called_once()
        args = mock_post.call_args[0]
        kwargs = mock_post.call_args[1]
        self.assertEqual(args[0], "http://localhost:8002/api/rave/process")
        self.assertEqual(kwargs["data"]["audio_file_id"], 1)
    
    @patch("axios.post")
    def test_api_proxy_vector_generate(self, mock_post):
        """Test the API proxy for vector generation."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data = {"vector_file": "/tmp/vectors.svg"}
        mock_post.return_value = mock_response
        
        # Make a request to the API proxy
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
        
        # Assert that axios.post was called with the correct arguments
        mock_post.assert_called_once()
        args = mock_post.call_args[0]
        kwargs = mock_post.call_args[1]
        self.assertEqual(args[0], "http://localhost:8003/api/vector/generate")
        self.assertEqual(kwargs["data"]["latent_file_id"], 1)
    
    @patch("axios.post")
    def test_api_proxy_laser_generate(self, mock_post):
        """Test the API proxy for laser file generation."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data = {"laser_file": "/tmp/laser.ild"}
        mock_post.return_value = mock_response
        
        # Make a request to the API proxy
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
        
        # Assert that axios.post was called with the correct arguments
        mock_post.assert_called_once()
        args = mock_post.call_args[0]
        kwargs = mock_post.call_args[1]
        self.assertEqual(args[0], "http://localhost:8004/api/laser/generate")
        self.assertEqual(kwargs["data"]["vector_file_id"], 1)


class TestSocketIO(unittest.TestCase):
    """Tests for the Socket.IO functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a Socket.IO client
        self.client = SocketIOClient()
        
        # Connect to the server
        self.client.connect("http://localhost:80")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Disconnect from the server
        self.client.disconnect()
    
    @patch("axios.post")
    def test_solar_data_event(self, mock_post):
        """Test the solar_data event."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.data = {"timestamp": "2025-04-22T12:00:00Z", "voltage": 12.5, "current": 2.1, "power": 26.25}
        mock_post.return_value = mock_response
        
        # Create a mock callback
        callback = MagicMock()
        
        # Register the callback for the solar_data_response event
        self.client.on("solar_data_response", callback)
        
        # Emit the solar_data event
        self.client.emit("solar_data", {"panel_id": "panel1"})
        
        # Wait for the response
        self.client.sleep(0.1)
        
        # Assert that the callback was called with the correct data
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(args[0]["timestamp"], "2025-04-22T12:00:00Z")
        self.assertEqual(args[0]["voltage"], 12.5)
        self.assertEqual(args[0]["current"], 2.1)
        self.assertEqual(args[0]["power"], 26.25)
    
    @patch("axios.post")
    def test_convert_audio_event(self, mock_post):
        """Test the convert_audio event."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.data = {"audio_file": "/tmp/audio.wav"}
        mock_post.return_value = mock_response
        
        # Create a mock callback
        callback = MagicMock()
        
        # Register the callback for the audio_conversion_response event
        self.client.on("audio_conversion_response", callback)
        
        # Emit the convert_audio event
        self.client.emit("convert_audio", {
            "start_time": "2025-04-22T00:00:00Z",
            "end_time": "2025-04-22T23:59:59Z",
            "panel_id": "panel1",
            "synthesizer": "direct"
        })
        
        # Wait for the response
        self.client.sleep(0.1)
        
        # Assert that the callback was called with the correct data
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(args[0]["audio_file"], "/tmp/audio.wav")
    
    @patch("axios.post")
    def test_process_rave_event(self, mock_post):
        """Test the process_rave event."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.data = {"latent_file": "/tmp/latent.pt", "processed_audio_file": "/tmp/processed.wav"}
        mock_post.return_value = mock_response
        
        # Create a mock callback
        callback = MagicMock()
        
        # Register the callback for the rave_processing_response event
        self.client.on("rave_processing_response", callback)
        
        # Emit the process_rave event
        self.client.emit("process_rave", {"audio_file_id": 1})
        
        # Wait for the response
        self.client.sleep(0.1)
        
        # Assert that the callback was called with the correct data
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(args[0]["latent_file"], "/tmp/latent.pt")
        self.assertEqual(args[0]["processed_audio_file"], "/tmp/processed.wav")
    
    @patch("axios.post")
    def test_generate_vector_event(self, mock_post):
        """Test the generate_vector event."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.data = {"vector_file": "/tmp/vectors.svg"}
        mock_post.return_value = mock_response
        
        # Create a mock callback
        callback = MagicMock()
        
        # Register the callback for the vector_generation_response event
        self.client.on("vector_generation_response", callback)
        
        # Emit the generate_vector event
        self.client.emit("generate_vector", {
            "latent_file_id": 1,
            "mapping": "direct",
            "smoothing_factor": 0.5,
            "normalization_range": [-1.0, 1.0]
        })
        
        # Wait for the response
        self.client.sleep(0.1)
        
        # Assert that the callback was called with the correct data
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(args[0]["vector_file"], "/tmp/vectors.svg")
    
    @patch("axios.post")
    def test_control_laser_event(self, mock_post):
        """Test the control_laser event."""
        # Mock the axios.post method
        mock_response = MagicMock()
        mock_response.data = {"laser_file": "/tmp/laser.ild"}
        mock_post.return_value = mock_response
        
        # Create a mock callback
        callback = MagicMock()
        
        # Register the callback for the laser_control_response event
        self.client.on("laser_control_response", callback)
        
        # Emit the control_laser event
        self.client.emit("control_laser", {
            "vector_file_id": 1,
            "format": "ILDA",
            "frame_rate": 30,
            "points_per_frame": 500,
            "color_mode": "RGB",
            "intensity": 0.8
        })
        
        # Wait for the response
        self.client.sleep(0.1)
        
        # Assert that the callback was called with the correct data
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(args[0]["laser_file"], "/tmp/laser.ild")
    
    def test_error_event(self):
        """Test the error event."""
        # Create a mock callback
        callback = MagicMock()
        
        # Register the callback for the error event
        self.client.on("error", callback)
        
        # Emit an event that will cause an error
        self.client.emit("solar_data", {"invalid": "data"})
        
        # Wait for the response
        self.client.sleep(0.1)
        
        # Assert that the callback was called with an error message
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertIn("message", args[0])


if __name__ == "__main__":
    unittest.main()