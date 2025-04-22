"""
Main entry point for the Solar-to-Laser System.

This module provides a command-line interface for the system.
"""

import os
import sys
import argparse
import logging
import uvicorn
from typing import Dict, Any, List, Optional

# Import modules
from . import data_collection
from . import audio_conversion
from . import rave_processing
from . import vector_generation
from . import laser_control
from . import deployment

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Solar-to-Laser System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data collection command
    data_parser = subparsers.add_parser("data", help="Run data collection API")
    data_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    data_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    # Audio conversion command
    audio_parser = subparsers.add_parser("audio", help="Run audio conversion API")
    audio_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    audio_parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    
    # RAVE processing command
    rave_parser = subparsers.add_parser("rave", help="Run RAVE processing API")
    rave_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    rave_parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    
    # Vector generation command
    vector_parser = subparsers.add_parser("vector", help="Run vector generation API")
    vector_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    vector_parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    
    # Laser control command
    laser_parser = subparsers.add_parser("laser", help="Run laser control API")
    laser_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    laser_parser.add_argument("--port", type=int, default=8004, help="Port to bind to")
    
    # Deployment command
    deploy_parser = subparsers.add_parser("deploy", help="Run deployment API")
    deploy_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    deploy_parser.add_argument("--port", type=int, default=8005, help="Port to bind to")
    
    # Docker Compose command
    compose_parser = subparsers.add_parser("compose", help="Run with Docker Compose")
    compose_parser.add_argument("--up", action="store_true", help="Start services")
    compose_parser.add_argument("--down", action="store_true", help="Stop services")
    compose_parser.add_argument("--logs", action="store_true", help="Show logs")
    compose_parser.add_argument("--service", type=str, help="Service name for logs")
    
    # Kubernetes command
    k8s_parser = subparsers.add_parser("k8s", help="Deploy to Kubernetes")
    k8s_parser.add_argument("--namespace", type=str, default="solar-to-laser-system", help="Kubernetes namespace")
    k8s_parser.add_argument("--registry", type=str, default="ghcr.io/yourusername", help="Container registry")
    
    # All-in-one command
    all_parser = subparsers.add_parser("all", help="Run all APIs")
    all_parser.add_argument("--base-port", type=int, default=8000, help="Base port number")
    
    return parser.parse_args()


def run_data_collection_api(host: str, port: int):
    """Run the data collection API."""
    logger.info(f"Starting data collection API on {host}:{port}")
    uvicorn.run(data_collection.api, host=host, port=port)


def run_audio_conversion_api(host: str, port: int):
    """Run the audio conversion API."""
    logger.info(f"Starting audio conversion API on {host}:{port}")
    uvicorn.run(audio_conversion.api, host=host, port=port)


def run_rave_processing_api(host: str, port: int):
    """Run the RAVE processing API."""
    logger.info(f"Starting RAVE processing API on {host}:{port}")
    uvicorn.run(rave_processing.api, host=host, port=port)


def run_vector_generation_api(host: str, port: int):
    """Run the vector generation API."""
    logger.info(f"Starting vector generation API on {host}:{port}")
    uvicorn.run(vector_generation.api, host=host, port=port)


def run_laser_control_api(host: str, port: int):
    """Run the laser control API."""
    logger.info(f"Starting laser control API on {host}:{port}")
    uvicorn.run(laser_control.api, host=host, port=port)


def run_deployment_api(host: str, port: int):
    """Run the deployment API."""
    logger.info(f"Starting deployment API on {host}:{port}")
    uvicorn.run(deployment.api, host=host, port=port)


def run_docker_compose(up: bool, down: bool, logs: bool, service: Optional[str] = None):
    """Run Docker Compose commands."""
    deployer = deployment.DockerComposeDeployer()
    
    if up:
        logger.info("Starting services with Docker Compose")
        deployer.up()
    
    if down:
        logger.info("Stopping services with Docker Compose")
        deployer.down()
    
    if logs:
        logger.info(f"Showing logs for {'all services' if service is None else service}")
        print(deployer.logs(service))


def deploy_to_kubernetes(namespace: str, registry: str):
    """Deploy to Kubernetes."""
    deployer = deployment.KubernetesDeployer(namespace=namespace, registry=registry)
    
    logger.info(f"Deploying to Kubernetes namespace {namespace}")
    deployer.deploy_all()


def run_all_apis(base_port: int):
    """Run all APIs."""
    import multiprocessing
    
    # Define processes
    processes = [
        multiprocessing.Process(target=run_data_collection_api, args=("0.0.0.0", base_port)),
        multiprocessing.Process(target=run_audio_conversion_api, args=("0.0.0.0", base_port + 1)),
        multiprocessing.Process(target=run_rave_processing_api, args=("0.0.0.0", base_port + 2)),
        multiprocessing.Process(target=run_vector_generation_api, args=("0.0.0.0", base_port + 3)),
        multiprocessing.Process(target=run_laser_control_api, args=("0.0.0.0", base_port + 4)),
        multiprocessing.Process(target=run_deployment_api, args=("0.0.0.0", base_port + 5))
    ]
    
    # Start processes
    for process in processes:
        process.start()
    
    # Wait for processes to finish
    for process in processes:
        process.join()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "data":
        run_data_collection_api(args.host, args.port)
    elif args.command == "audio":
        run_audio_conversion_api(args.host, args.port)
    elif args.command == "rave":
        run_rave_processing_api(args.host, args.port)
    elif args.command == "vector":
        run_vector_generation_api(args.host, args.port)
    elif args.command == "laser":
        run_laser_control_api(args.host, args.port)
    elif args.command == "deploy":
        run_deployment_api(args.host, args.port)
    elif args.command == "compose":
        run_docker_compose(args.up, args.down, args.logs, args.service)
    elif args.command == "k8s":
        deploy_to_kubernetes(args.namespace, args.registry)
    elif args.command == "all":
        run_all_apis(args.base_port)
    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()