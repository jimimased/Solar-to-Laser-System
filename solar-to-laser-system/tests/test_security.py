"""
Security tests for the Solar-to-Laser System.
"""

import unittest
import os
import tempfile
import json
import requests
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.data_collection.api import api as data_collection_api
from src.audio_conversion.api import api as audio_conversion_api
from src.rave_processing.api import api as rave_processing_api
from src.vector_generation.api import api as vector_generation_api
from src.laser_control.api import api as laser_control_api
from src.deployment.api import api as deployment_api


@pytest.mark.security
class TestAPIAuthentication(unittest.TestCase):
    """Tests for API authentication."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.data_collection_client = TestClient(data_collection_api)
        self.audio_conversion_client = TestClient(audio_conversion_api)
        self.rave_processing_client = TestClient(rave_processing_api)
        self.vector_generation_client = TestClient(vector_generation_api)
        self.laser_control_client = TestClient(laser_control_api)
        self.deployment_client = TestClient(deployment_api)
    
    @patch("src.data_collection.api.authenticate")
    def test_data_collection_authentication(self, mock_authenticate):
        """Test authentication for the data collection API."""
        # Mock the authenticate function to return False
        mock_authenticate.return_value = False
        
        # Make a request to a protected endpoint
        response = self.data_collection_client.post(
            "/api/solar/data",
            json={"panel_id": "panel1"},
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Assert that the response is unauthorized
        self.assertEqual(response.status_code, 401)
        
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Make a request to a protected endpoint
        response = self.data_collection_client.post(
            "/api/solar/data",
            json={"panel_id": "panel1"},
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is not unauthorized
        self.assertNotEqual(response.status_code, 401)
    
    @patch("src.audio_conversion.api.authenticate")
    def test_audio_conversion_authentication(self, mock_authenticate):
        """Test authentication for the audio conversion API."""
        # Mock the authenticate function to return False
        mock_authenticate.return_value = False
        
        # Make a request to a protected endpoint
        response = self.audio_conversion_client.post(
            "/api/audio/convert",
            json={
                "start_time": "2025-04-22T00:00:00Z",
                "end_time": "2025-04-22T23:59:59Z",
                "panel_id": "panel1",
                "synthesizer": "direct"
            },
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Assert that the response is unauthorized
        self.assertEqual(response.status_code, 401)
        
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Make a request to a protected endpoint
        response = self.audio_conversion_client.post(
            "/api/audio/convert",
            json={
                "start_time": "2025-04-22T00:00:00Z",
                "end_time": "2025-04-22T23:59:59Z",
                "panel_id": "panel1",
                "synthesizer": "direct"
            },
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is not unauthorized
        self.assertNotEqual(response.status_code, 401)
    
    @patch("src.deployment.api.authenticate")
    def test_deployment_authentication(self, mock_authenticate):
        """Test authentication for the deployment API."""
        # Mock the authenticate function to return False
        mock_authenticate.return_value = False
        
        # Make a request to a protected endpoint
        response = self.deployment_client.post(
            "/api/deployment/kubernetes",
            json={
                "namespace": "test-namespace",
                "registry": "ghcr.io/testuser"
            },
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Assert that the response is unauthorized
        self.assertEqual(response.status_code, 401)
        
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Make a request to a protected endpoint
        response = self.deployment_client.post(
            "/api/deployment/kubernetes",
            json={
                "namespace": "test-namespace",
                "registry": "ghcr.io/testuser"
            },
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is not unauthorized
        self.assertNotEqual(response.status_code, 401)


@pytest.mark.security
class TestInputValidation(unittest.TestCase):
    """Tests for input validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.data_collection_client = TestClient(data_collection_api)
        self.audio_conversion_client = TestClient(audio_conversion_api)
        self.rave_processing_client = TestClient(rave_processing_api)
        self.vector_generation_client = TestClient(vector_generation_api)
        self.laser_control_client = TestClient(laser_control_api)
        self.deployment_client = TestClient(deployment_api)
    
    @patch("src.data_collection.api.authenticate")
    def test_data_collection_input_validation(self, mock_authenticate):
        """Test input validation for the data collection API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Make a request with invalid input
        response = self.data_collection_client.post(
            "/api/solar/data",
            json={"invalid_field": "invalid_value"},
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is a validation error
        self.assertEqual(response.status_code, 422)
        
        # Make a request with valid input
        response = self.data_collection_client.post(
            "/api/solar/data",
            json={"panel_id": "panel1"},
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is not a validation error
        self.assertNotEqual(response.status_code, 422)
    
    @patch("src.audio_conversion.api.authenticate")
    def test_audio_conversion_input_validation(self, mock_authenticate):
        """Test input validation for the audio conversion API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Make a request with invalid input
        response = self.audio_conversion_client.post(
            "/api/audio/convert",
            json={
                "start_time": "invalid_time",
                "end_time": "2025-04-22T23:59:59Z",
                "panel_id": "panel1",
                "synthesizer": "direct"
            },
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is a validation error
        self.assertEqual(response.status_code, 422)
        
        # Make a request with valid input
        response = self.audio_conversion_client.post(
            "/api/audio/convert",
            json={
                "start_time": "2025-04-22T00:00:00Z",
                "end_time": "2025-04-22T23:59:59Z",
                "panel_id": "panel1",
                "synthesizer": "direct"
            },
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is not a validation error
        self.assertNotEqual(response.status_code, 422)
    
    @patch("src.deployment.api.authenticate")
    def test_deployment_input_validation(self, mock_authenticate):
        """Test input validation for the deployment API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Make a request with invalid input
        response = self.deployment_client.post(
            "/api/deployment/kubernetes",
            json={
                "namespace": 123,  # Should be a string
                "registry": "ghcr.io/testuser"
            },
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is a validation error
        self.assertEqual(response.status_code, 422)
        
        # Make a request with valid input
        response = self.deployment_client.post(
            "/api/deployment/kubernetes",
            json={
                "namespace": "test-namespace",
                "registry": "ghcr.io/testuser"
            },
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is not a validation error
        self.assertNotEqual(response.status_code, 422)


@pytest.mark.security
class TestSQLInjection(unittest.TestCase):
    """Tests for SQL injection vulnerabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.data_collection_client = TestClient(data_collection_api)
        self.audio_conversion_client = TestClient(audio_conversion_api)
    
    @patch("src.data_collection.api.authenticate")
    @patch("src.data_collection.api.storage")
    def test_data_collection_sql_injection(self, mock_storage, mock_authenticate):
        """Test SQL injection protection for the data collection API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Mock the storage.query method
        mock_storage.query.return_value = []
        
        # Make a request with a SQL injection attempt
        response = self.data_collection_client.get(
            "/api/solar/data/2025-04-22T00:00:00Z/2025-04-22T23:59:59Z?panel_id=panel1' OR '1'='1",
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is successful (the API should sanitize the input)
        self.assertEqual(response.status_code, 200)
        
        # Assert that the storage.query method was called with sanitized input
        mock_storage.query.assert_called_once()
        args = mock_storage.query.call_args[0]
        self.assertNotIn("OR '1'='1", args[2])
    
    @patch("src.audio_conversion.api.authenticate")
    @patch("src.audio_conversion.api.db")
    def test_audio_conversion_sql_injection(self, mock_db, mock_authenticate):
        """Test SQL injection protection for the audio conversion API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Mock the db.query method
        mock_db.query.return_value = []
        
        # Make a request with a SQL injection attempt
        response = self.audio_conversion_client.get(
            "/api/audio/file/1; DROP TABLE audio_files;",
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is a validation error (the API should validate the input)
        self.assertEqual(response.status_code, 422)


@pytest.mark.security
class TestXSS(unittest.TestCase):
    """Tests for cross-site scripting (XSS) vulnerabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.data_collection_client = TestClient(data_collection_api)
    
    @patch("src.data_collection.api.authenticate")
    @patch("src.data_collection.api.collector")
    def test_data_collection_xss(self, mock_collector, mock_authenticate):
        """Test XSS protection for the data collection API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Mock the collector.collect_and_store method
        mock_collector.collect_and_store.return_value = MagicMock(
            timestamp="2025-04-22T12:00:00Z",
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Make a request with an XSS attempt
        response = self.data_collection_client.post(
            "/api/solar/data",
            json={"panel_id": "<script>alert('XSS')</script>"},
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is successful (the API should sanitize the input)
        self.assertEqual(response.status_code, 200)
        
        # Assert that the response does not contain the script tag
        self.assertNotIn("<script>", response.text)


@pytest.mark.security
class TestCSRF(unittest.TestCase):
    """Tests for cross-site request forgery (CSRF) vulnerabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.data_collection_client = TestClient(data_collection_api)
    
    @patch("src.data_collection.api.authenticate")
    def test_data_collection_csrf(self, mock_authenticate):
        """Test CSRF protection for the data collection API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Make a request without a CSRF token
        response = self.data_collection_client.post(
            "/api/solar/data",
            json={"panel_id": "panel1"},
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is successful (the API should use token-based authentication)
        self.assertNotEqual(response.status_code, 403)


@pytest.mark.security
class TestRateLimiting(unittest.TestCase):
    """Tests for rate limiting."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.data_collection_client = TestClient(data_collection_api)
    
    @patch("src.data_collection.api.authenticate")
    @patch("src.data_collection.api.rate_limiter")
    def test_data_collection_rate_limiting(self, mock_rate_limiter, mock_authenticate):
        """Test rate limiting for the data collection API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Mock the rate_limiter.is_rate_limited method to return False for the first 5 requests and True for the rest
        mock_rate_limiter.is_rate_limited.side_effect = [False, False, False, False, False, True, True, True, True, True]
        
        # Make 5 requests (should be allowed)
        for i in range(5):
            response = self.data_collection_client.post(
                "/api/solar/data",
                json={"panel_id": "panel1"},
                headers={"Authorization": "Bearer valid_token"}
            )
            
            # Assert that the response is successful
            self.assertNotEqual(response.status_code, 429)
        
        # Make 5 more requests (should be rate limited)
        for i in range(5):
            response = self.data_collection_client.post(
                "/api/solar/data",
                json={"panel_id": "panel1"},
                headers={"Authorization": "Bearer valid_token"}
            )
            
            # Assert that the response is rate limited
            self.assertEqual(response.status_code, 429)


@pytest.mark.security
class TestFileUploadSecurity(unittest.TestCase):
    """Tests for file upload security."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.audio_conversion_client = TestClient(audio_conversion_api)
    
    @patch("src.audio_conversion.api.authenticate")
    def test_audio_conversion_file_upload(self, mock_authenticate):
        """Test file upload security for the audio conversion API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            # Write some data to the file
            temp_file.write(b"This is not a valid WAV file")
            temp_file.flush()
            
            # Make a request with the file
            with open(temp_file.name, "rb") as f:
                response = self.audio_conversion_client.post(
                    "/api/audio/upload",
                    files={"file": ("test.wav", f, "audio/wav")},
                    headers={"Authorization": "Bearer valid_token"}
                )
            
            # Assert that the response is a validation error (the API should validate the file)
            self.assertEqual(response.status_code, 422)


@pytest.mark.security
class TestCommandInjection(unittest.TestCase):
    """Tests for command injection vulnerabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.deployment_client = TestClient(deployment_api)
    
    @patch("src.deployment.api.authenticate")
    @patch("subprocess.run")
    def test_deployment_command_injection(self, mock_run, mock_authenticate):
        """Test command injection protection for the deployment API."""
        # Mock the authenticate function to return True
        mock_authenticate.return_value = True
        
        # Mock the subprocess.run method
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Command executed"
        mock_run.return_value = mock_process
        
        # Make a request with a command injection attempt
        response = self.deployment_client.post(
            "/api/deployment/execute",
            json={"command": "ls; rm -rf /"},
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Assert that the response is a validation error (the API should validate the command)
        self.assertEqual(response.status_code, 422)


@pytest.mark.security
class TestSecureHeaders(unittest.TestCase):
    """Tests for secure headers."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test clients
        self.data_collection_client = TestClient(data_collection_api)
    
    def test_data_collection_secure_headers(self):
        """Test secure headers for the data collection API."""
        # Make a request to the API
        response = self.data_collection_client.get("/")
        
        # Assert that the response includes secure headers
        headers = response.headers
        self.assertIn("X-Content-Type-Options", headers)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        
        self.assertIn("X-Frame-Options", headers)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        
        self.assertIn("X-XSS-Protection", headers)
        self.assertEqual(headers["X-XSS-Protection"], "1; mode=block")
        
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])


if __name__ == "__main__":
    unittest.main()