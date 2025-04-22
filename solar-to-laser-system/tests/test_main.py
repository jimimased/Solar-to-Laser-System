"""
Tests for the main module.
"""

import unittest
import sys
from unittest.mock import patch, MagicMock

from src.main import (
    parse_args,
    run_data_collection_api,
    run_audio_conversion_api,
    run_rave_processing_api,
    run_vector_generation_api,
    run_laser_control_api,
    run_deployment_api,
    run_docker_compose,
    deploy_to_kubernetes,
    run_all_apis,
    main
)


class TestParseArgs(unittest.TestCase):
    """Tests for the parse_args function."""
    
    def test_data_command(self):
        """Test parsing the data command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "data"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command is correct
            self.assertEqual(args.command, "data")
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 8000)
    
    def test_audio_command(self):
        """Test parsing the audio command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "audio", "--port", "9000"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command and port are correct
            self.assertEqual(args.command, "audio")
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 9000)
    
    def test_rave_command(self):
        """Test parsing the rave command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "rave", "--host", "127.0.0.1"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command and host are correct
            self.assertEqual(args.command, "rave")
            self.assertEqual(args.host, "127.0.0.1")
            self.assertEqual(args.port, 8002)
    
    def test_vector_command(self):
        """Test parsing the vector command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "vector"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command is correct
            self.assertEqual(args.command, "vector")
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 8003)
    
    def test_laser_command(self):
        """Test parsing the laser command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "laser"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command is correct
            self.assertEqual(args.command, "laser")
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 8004)
    
    def test_deploy_command(self):
        """Test parsing the deploy command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "deploy"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command is correct
            self.assertEqual(args.command, "deploy")
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 8005)
    
    def test_compose_command(self):
        """Test parsing the compose command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "compose", "--up"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command and flags are correct
            self.assertEqual(args.command, "compose")
            self.assertTrue(args.up)
            self.assertFalse(args.down)
            self.assertFalse(args.logs)
            self.assertIsNone(args.service)
    
    def test_k8s_command(self):
        """Test parsing the k8s command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "k8s", "--namespace", "test-namespace"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command and namespace are correct
            self.assertEqual(args.command, "k8s")
            self.assertEqual(args.namespace, "test-namespace")
            self.assertEqual(args.registry, "ghcr.io/yourusername")
    
    def test_all_command(self):
        """Test parsing the all command."""
        # Mock sys.argv
        with patch("sys.argv", ["solar-to-laser", "all", "--base-port", "9000"]):
            # Parse arguments
            args = parse_args()
            
            # Assert that the command and base port are correct
            self.assertEqual(args.command, "all")
            self.assertEqual(args.base_port, 9000)


class TestRunAPIs(unittest.TestCase):
    """Tests for the API runner functions."""
    
    @patch("uvicorn.run")
    def test_run_data_collection_api(self, mock_run):
        """Test running the data collection API."""
        # Run the API
        run_data_collection_api("127.0.0.1", 9000)
        
        # Assert that uvicorn.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9000)
    
    @patch("uvicorn.run")
    def test_run_audio_conversion_api(self, mock_run):
        """Test running the audio conversion API."""
        # Run the API
        run_audio_conversion_api("127.0.0.1", 9001)
        
        # Assert that uvicorn.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9001)
    
    @patch("uvicorn.run")
    def test_run_rave_processing_api(self, mock_run):
        """Test running the RAVE processing API."""
        # Run the API
        run_rave_processing_api("127.0.0.1", 9002)
        
        # Assert that uvicorn.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9002)
    
    @patch("uvicorn.run")
    def test_run_vector_generation_api(self, mock_run):
        """Test running the vector generation API."""
        # Run the API
        run_vector_generation_api("127.0.0.1", 9003)
        
        # Assert that uvicorn.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9003)
    
    @patch("uvicorn.run")
    def test_run_laser_control_api(self, mock_run):
        """Test running the laser control API."""
        # Run the API
        run_laser_control_api("127.0.0.1", 9004)
        
        # Assert that uvicorn.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9004)
    
    @patch("uvicorn.run")
    def test_run_deployment_api(self, mock_run):
        """Test running the deployment API."""
        # Run the API
        run_deployment_api("127.0.0.1", 9005)
        
        # Assert that uvicorn.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9005)


class TestRunDockerCompose(unittest.TestCase):
    """Tests for the run_docker_compose function."""
    
    @patch("src.deployment.DockerComposeDeployer")
    def test_run_docker_compose_up(self, mock_deployer):
        """Test running Docker Compose up."""
        # Mock the DockerComposeDeployer
        mock_deployer.return_value.up.return_value = True
        
        # Run Docker Compose up
        run_docker_compose(up=True, down=False, logs=False)
        
        # Assert that the up method was called
        mock_deployer.return_value.up.assert_called_once()
        mock_deployer.return_value.down.assert_not_called()
        mock_deployer.return_value.logs.assert_not_called()
    
    @patch("src.deployment.DockerComposeDeployer")
    def test_run_docker_compose_down(self, mock_deployer):
        """Test running Docker Compose down."""
        # Mock the DockerComposeDeployer
        mock_deployer.return_value.down.return_value = True
        
        # Run Docker Compose down
        run_docker_compose(up=False, down=True, logs=False)
        
        # Assert that the down method was called
        mock_deployer.return_value.up.assert_not_called()
        mock_deployer.return_value.down.assert_called_once()
        mock_deployer.return_value.logs.assert_not_called()
    
    @patch("src.deployment.DockerComposeDeployer")
    def test_run_docker_compose_logs(self, mock_deployer):
        """Test running Docker Compose logs."""
        # Mock the DockerComposeDeployer
        mock_deployer.return_value.logs.return_value = "logs"
        
        # Run Docker Compose logs
        run_docker_compose(up=False, down=False, logs=True)
        
        # Assert that the logs method was called
        mock_deployer.return_value.up.assert_not_called()
        mock_deployer.return_value.down.assert_not_called()
        mock_deployer.return_value.logs.assert_called_once()


class TestDeployToKubernetes(unittest.TestCase):
    """Tests for the deploy_to_kubernetes function."""
    
    @patch("src.deployment.KubernetesDeployer")
    def test_deploy_to_kubernetes(self, mock_deployer):
        """Test deploying to Kubernetes."""
        # Mock the KubernetesDeployer
        mock_deployer.return_value.deploy_all.return_value = True
        
        # Deploy to Kubernetes
        deploy_to_kubernetes(namespace="test-namespace", registry="ghcr.io/testuser")
        
        # Assert that the deploy_all method was called
        mock_deployer.assert_called_once_with(namespace="test-namespace", registry="ghcr.io/testuser")
        mock_deployer.return_value.deploy_all.assert_called_once()


class TestRunAllAPIs(unittest.TestCase):
    """Tests for the run_all_apis function."""
    
    @patch("multiprocessing.Process")
    def test_run_all_apis(self, mock_process):
        """Test running all APIs."""
        # Mock the Process class
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        
        # Run all APIs
        run_all_apis(base_port=9000)
        
        # Assert that Process was called for each API
        self.assertEqual(mock_process.call_count, 6)
        
        # Assert that start and join were called for each process
        self.assertEqual(mock_process_instance.start.call_count, 6)
        self.assertEqual(mock_process_instance.join.call_count, 6)


class TestMain(unittest.TestCase):
    """Tests for the main function."""
    
    @patch("src.main.parse_args")
    @patch("src.main.run_data_collection_api")
    def test_main_data_command(self, mock_run_api, mock_parse_args):
        """Test running the main function with the data command."""
        # Mock parse_args to return a data command
        mock_args = MagicMock()
        mock_args.command = "data"
        mock_args.host = "127.0.0.1"
        mock_args.port = 9000
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_data_collection_api was called with the correct arguments
        mock_run_api.assert_called_once_with("127.0.0.1", 9000)
    
    @patch("src.main.parse_args")
    @patch("src.main.run_audio_conversion_api")
    def test_main_audio_command(self, mock_run_api, mock_parse_args):
        """Test running the main function with the audio command."""
        # Mock parse_args to return an audio command
        mock_args = MagicMock()
        mock_args.command = "audio"
        mock_args.host = "127.0.0.1"
        mock_args.port = 9001
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_audio_conversion_api was called with the correct arguments
        mock_run_api.assert_called_once_with("127.0.0.1", 9001)
    
    @patch("src.main.parse_args")
    @patch("src.main.run_rave_processing_api")
    def test_main_rave_command(self, mock_run_api, mock_parse_args):
        """Test running the main function with the rave command."""
        # Mock parse_args to return a rave command
        mock_args = MagicMock()
        mock_args.command = "rave"
        mock_args.host = "127.0.0.1"
        mock_args.port = 9002
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_rave_processing_api was called with the correct arguments
        mock_run_api.assert_called_once_with("127.0.0.1", 9002)
    
    @patch("src.main.parse_args")
    @patch("src.main.run_vector_generation_api")
    def test_main_vector_command(self, mock_run_api, mock_parse_args):
        """Test running the main function with the vector command."""
        # Mock parse_args to return a vector command
        mock_args = MagicMock()
        mock_args.command = "vector"
        mock_args.host = "127.0.0.1"
        mock_args.port = 9003
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_vector_generation_api was called with the correct arguments
        mock_run_api.assert_called_once_with("127.0.0.1", 9003)
    
    @patch("src.main.parse_args")
    @patch("src.main.run_laser_control_api")
    def test_main_laser_command(self, mock_run_api, mock_parse_args):
        """Test running the main function with the laser command."""
        # Mock parse_args to return a laser command
        mock_args = MagicMock()
        mock_args.command = "laser"
        mock_args.host = "127.0.0.1"
        mock_args.port = 9004
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_laser_control_api was called with the correct arguments
        mock_run_api.assert_called_once_with("127.0.0.1", 9004)
    
    @patch("src.main.parse_args")
    @patch("src.main.run_deployment_api")
    def test_main_deploy_command(self, mock_run_api, mock_parse_args):
        """Test running the main function with the deploy command."""
        # Mock parse_args to return a deploy command
        mock_args = MagicMock()
        mock_args.command = "deploy"
        mock_args.host = "127.0.0.1"
        mock_args.port = 9005
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_deployment_api was called with the correct arguments
        mock_run_api.assert_called_once_with("127.0.0.1", 9005)
    
    @patch("src.main.parse_args")
    @patch("src.main.run_docker_compose")
    def test_main_compose_command(self, mock_run_compose, mock_parse_args):
        """Test running the main function with the compose command."""
        # Mock parse_args to return a compose command
        mock_args = MagicMock()
        mock_args.command = "compose"
        mock_args.up = True
        mock_args.down = False
        mock_args.logs = False
        mock_args.service = None
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_docker_compose was called with the correct arguments
        mock_run_compose.assert_called_once_with(True, False, False, None)
    
    @patch("src.main.parse_args")
    @patch("src.main.deploy_to_kubernetes")
    def test_main_k8s_command(self, mock_deploy, mock_parse_args):
        """Test running the main function with the k8s command."""
        # Mock parse_args to return a k8s command
        mock_args = MagicMock()
        mock_args.command = "k8s"
        mock_args.namespace = "test-namespace"
        mock_args.registry = "ghcr.io/testuser"
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that deploy_to_kubernetes was called with the correct arguments
        mock_deploy.assert_called_once_with("test-namespace", "ghcr.io/testuser")
    
    @patch("src.main.parse_args")
    @patch("src.main.run_all_apis")
    def test_main_all_command(self, mock_run_all, mock_parse_args):
        """Test running the main function with the all command."""
        # Mock parse_args to return an all command
        mock_args = MagicMock()
        mock_args.command = "all"
        mock_args.base_port = 9000
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that run_all_apis was called with the correct arguments
        mock_run_all.assert_called_once_with(9000)
    
    @patch("src.main.parse_args")
    @patch("sys.exit")
    def test_main_no_command(self, mock_exit, mock_parse_args):
        """Test running the main function with no command."""
        # Mock parse_args to return no command
        mock_args = MagicMock()
        mock_args.command = None
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Assert that sys.exit was called with the correct arguments
        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()