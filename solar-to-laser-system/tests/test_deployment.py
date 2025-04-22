"""
Tests for the deployment module.
"""

import unittest
import os
import tempfile
import json
import yaml
from unittest.mock import patch, MagicMock, mock_open

from src.deployment.kubernetes import KubernetesDeployer
from src.deployment.docker import DockerDeployer, DockerComposeDeployer


class TestKubernetesDeployer(unittest.TestCase):
    """Tests for the KubernetesDeployer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize the Kubernetes deployer
        self.deployer = KubernetesDeployer(
            namespace="test-namespace",
            registry="ghcr.io/testuser",
            config_path=None
        )
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    @patch("subprocess.run")
    def test_create_namespace(self, mock_run):
        """Test creating a Kubernetes namespace."""
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "namespace/test-namespace created"
        mock_run.return_value = mock_process
        
        # Create the namespace
        result = self.deployer.create_namespace()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "kubectl")
        self.assertEqual(args[1], "apply")
        self.assertEqual(args[2], "-f")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    def test_deploy_database(self, mock_run):
        """Test deploying database components."""
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "deployment.apps/influxdb created\nservice/influxdb created\ndeployment.apps/postgres created\nservice/postgres created"
        mock_run.return_value = mock_process
        
        # Deploy the database
        result = self.deployer.deploy_database()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "kubectl")
        self.assertEqual(args[1], "apply")
        self.assertEqual(args[2], "-f")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    def test_deploy_message_queue(self, mock_run):
        """Test deploying message queue components."""
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "deployment.apps/rabbitmq created\nservice/rabbitmq created"
        mock_run.return_value = mock_process
        
        # Deploy the message queue
        result = self.deployer.deploy_message_queue()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "kubectl")
        self.assertEqual(args[1], "apply")
        self.assertEqual(args[2], "-f")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    def test_deploy_services(self, mock_run):
        """Test deploying service components."""
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "deployment.apps/data-collection created\nservice/data-collection created\ndeployment.apps/audio-conversion created\nservice/audio-conversion created"
        mock_run.return_value = mock_process
        
        # Deploy the services
        result = self.deployer.deploy_services()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "kubectl")
        self.assertEqual(args[1], "apply")
        self.assertEqual(args[2], "-f")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    def test_deploy_ingress(self, mock_run):
        """Test deploying ingress components."""
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "ingress.networking.k8s.io/solar-to-laser-ingress created"
        mock_run.return_value = mock_process
        
        # Deploy the ingress
        result = self.deployer.deploy_ingress()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "kubectl")
        self.assertEqual(args[1], "apply")
        self.assertEqual(args[2], "-f")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch.object(KubernetesDeployer, "create_namespace")
    @patch.object(KubernetesDeployer, "deploy_database")
    @patch.object(KubernetesDeployer, "deploy_message_queue")
    @patch.object(KubernetesDeployer, "deploy_services")
    @patch.object(KubernetesDeployer, "deploy_ingress")
    def test_deploy_all(self, mock_ingress, mock_services, mock_mq, mock_db, mock_ns):
        """Test deploying all components."""
        # Mock the methods to return True
        mock_ns.return_value = True
        mock_db.return_value = True
        mock_mq.return_value = True
        mock_services.return_value = True
        mock_ingress.return_value = True
        
        # Deploy all components
        result = self.deployer.deploy_all()
        
        # Assert that all methods were called
        mock_ns.assert_called_once()
        mock_db.assert_called_once()
        mock_mq.assert_called_once()
        mock_services.assert_called_once()
        mock_ingress.assert_called_once()
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    def test_get_status(self, mock_run):
        """Test getting deployment status."""
        # Mock the subprocess.run method to return successful results
        mock_process1 = MagicMock()
        mock_process1.returncode = 0
        mock_process1.stdout = json.dumps({
            "items": [
                {
                    "metadata": {"name": "data-collection"},
                    "status": {"phase": "Running"}
                }
            ]
        })
        
        mock_process2 = MagicMock()
        mock_process2.returncode = 0
        mock_process2.stdout = json.dumps({
            "items": [
                {
                    "metadata": {"name": "data-collection"},
                    "spec": {"ports": [{"port": 8000}]}
                }
            ]
        })
        
        mock_run.side_effect = [mock_process1, mock_process2]
        
        # Get the status
        status = self.deployer.get_status()
        
        # Assert that subprocess.run was called twice with the correct arguments
        self.assertEqual(mock_run.call_count, 2)
        args1 = mock_run.call_args_list[0][0][0]
        self.assertEqual(args1[0], "kubectl")
        self.assertEqual(args1[1], "get")
        self.assertEqual(args1[2], "pods")
        
        args2 = mock_run.call_args_list[1][0][0]
        self.assertEqual(args2[0], "kubectl")
        self.assertEqual(args2[1], "get")
        self.assertEqual(args2[2], "services")
        
        # Assert that the status has the correct keys
        self.assertIn("status", status)
        self.assertIn("pods", status)
        self.assertIn("services", status)


class TestDockerDeployer(unittest.TestCase):
    """Tests for the DockerDeployer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize the Docker deployer
        self.deployer = DockerDeployer(
            registry="ghcr.io/testuser",
            project_dir=".",
            dockerfile_dir="docker"
        )
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_build_image(self, mock_exists, mock_run):
        """Test building a Docker image."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Successfully built 1234567890ab"
        mock_run.return_value = mock_process
        
        # Build the image
        result = self.deployer.build_image("data-collection")
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "docker")
        self.assertEqual(args[1], "build")
        self.assertEqual(args[2], "-t")
        self.assertEqual(args[3], "ghcr.io/testuser/solar-to-laser-system/data-collection:latest")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    def test_push_image(self, mock_run):
        """Test pushing a Docker image."""
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "The push refers to repository [ghcr.io/testuser/solar-to-laser-system/data-collection]"
        mock_run.return_value = mock_process
        
        # Push the image
        result = self.deployer.push_image("data-collection")
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "docker")
        self.assertEqual(args[1], "push")
        self.assertEqual(args[2], "ghcr.io/testuser/solar-to-laser-system/data-collection:latest")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch.object(DockerDeployer, "build_image")
    @patch.object(DockerDeployer, "push_image")
    def test_build_and_push_all(self, mock_push, mock_build):
        """Test building and pushing all images."""
        # Mock the methods to return True
        mock_build.return_value = True
        mock_push.return_value = True
        
        # Build and push all images
        result = self.deployer.build_and_push_all()
        
        # Assert that the methods were called for each service
        self.assertEqual(mock_build.call_count, 6)
        self.assertEqual(mock_push.call_count, 6)
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_create_default_dockerfile(self, mock_makedirs, mock_file):
        """Test creating a default Dockerfile."""
        # Create a default Dockerfile
        self.deployer._create_default_dockerfile("data-collection")
        
        # Assert that os.makedirs was called
        mock_makedirs.assert_called_once()
        
        # Assert that open was called with the correct path
        mock_file.assert_called_once()
        args = mock_file.call_args[0]
        self.assertIn("Dockerfile.data-collection", args[0])
        
        # Assert that write was called
        mock_file().write.assert_called_once()


class TestDockerComposeDeployer(unittest.TestCase):
    """Tests for the DockerComposeDeployer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize the Docker Compose deployer
        self.deployer = DockerComposeDeployer(
            project_dir=".",
            compose_file="docker-compose.yml"
        )
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    @patch("builtins.open", new_callable=mock_open)
    def test_create_compose_file(self, mock_file):
        """Test creating a Docker Compose file."""
        # Create the Docker Compose file
        result = self.deployer.create_compose_file()
        
        # Assert that open was called with the correct path
        mock_file.assert_called_once()
        args = mock_file.call_args[0]
        self.assertIn("docker-compose.yml", args[0])
        
        # Assert that write was called
        mock_file().write.assert_called_once()
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_up(self, mock_exists, mock_run):
        """Test starting services with Docker Compose."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Creating network \"solar-to-laser-system_default\" with the default driver"
        mock_run.return_value = mock_process
        
        # Start the services
        result = self.deployer.up()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "docker-compose")
        self.assertEqual(args[1], "-f")
        self.assertEqual(args[3], "up")
        self.assertEqual(args[4], "-d")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_down(self, mock_exists, mock_run):
        """Test stopping services with Docker Compose."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Stopping solar-to-laser-system_web_1 ... done"
        mock_run.return_value = mock_process
        
        # Stop the services
        result = self.deployer.down()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "docker-compose")
        self.assertEqual(args[1], "-f")
        self.assertEqual(args[3], "down")
        
        # Assert that the result is True
        self.assertTrue(result)
    
    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_logs(self, mock_exists, mock_run):
        """Test getting logs from Docker Compose services."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock the subprocess.run method to return a successful result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "solar-to-laser-system_web_1 | Server running on port 80"
        mock_run.return_value = mock_process
        
        # Get the logs
        logs = self.deployer.logs()
        
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "docker-compose")
        self.assertEqual(args[1], "-f")
        self.assertEqual(args[3], "logs")
        
        # Assert that the logs are correct
        self.assertEqual(logs, "solar-to-laser-system_web_1 | Server running on port 80")


if __name__ == "__main__":
    unittest.main()