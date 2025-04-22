"""
Docker deployment implementation.

This module provides classes for building and deploying Docker images.
"""

import os
import logging
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class DockerDeployer:
    """Class for building and deploying Docker images."""
    
    def __init__(
        self,
        registry: str = "ghcr.io/yourusername",
        project_dir: str = ".",
        dockerfile_dir: str = "docker"
    ):
        """Initialize the Docker deployer.
        
        Args:
            registry: Container registry
            project_dir: Project directory
            dockerfile_dir: Directory containing Dockerfiles
        """
        self.registry = registry
        self.project_dir = project_dir
        self.dockerfile_dir = os.path.join(project_dir, dockerfile_dir)
        
        # Ensure Dockerfile directory exists
        os.makedirs(self.dockerfile_dir, exist_ok=True)
    
    def build_image(self, service: str, tag: str = "latest") -> bool:
        """Build Docker image for a service.
        
        Args:
            service: Service name
            tag: Image tag
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if Dockerfile exists
            dockerfile_path = os.path.join(self.dockerfile_dir, f"Dockerfile.{service}")
            if not os.path.exists(dockerfile_path):
                # Create default Dockerfile
                self._create_default_dockerfile(service)
                dockerfile_path = os.path.join(self.dockerfile_dir, f"Dockerfile.{service}")
            
            # Build image
            image_name = f"{self.registry}/solar-to-laser-system/{service}:{tag}"
            
            result = subprocess.run(
                ["docker", "build", "-t", image_name, "-f", dockerfile_path, self.project_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error building image for {service}: {result.stderr}")
                return False
            
            logger.info(f"Built image {image_name}")
            return True
        except Exception as e:
            logger.error(f"Error building image for {service}: {e}")
            return False
    
    def push_image(self, service: str, tag: str = "latest") -> bool:
        """Push Docker image for a service.
        
        Args:
            service: Service name
            tag: Image tag
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Push image
            image_name = f"{self.registry}/solar-to-laser-system/{service}:{tag}"
            
            result = subprocess.run(
                ["docker", "push", image_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error pushing image for {service}: {result.stderr}")
                return False
            
            logger.info(f"Pushed image {image_name}")
            return True
        except Exception as e:
            logger.error(f"Error pushing image for {service}: {e}")
            return False
    
    def build_and_push_all(self, tag: str = "latest") -> bool:
        """Build and push all service images.
        
        Args:
            tag: Image tag
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # List of services
            services = [
                "data-collection",
                "audio-conversion",
                "rave-processing",
                "vector-generation",
                "laser-control",
                "web"
            ]
            
            # Build and push each service
            for service in services:
                if not self.build_image(service, tag):
                    return False
                
                if not self.push_image(service, tag):
                    return False
            
            logger.info("Built and pushed all images")
            return True
        except Exception as e:
            logger.error(f"Error building and pushing all images: {e}")
            return False
    
    def _create_default_dockerfile(self, service: str) -> None:
        """Create default Dockerfile for a service.
        
        Args:
            service: Service name
        """
        # Create Dockerfile directory if it doesn't exist
        os.makedirs(self.dockerfile_dir, exist_ok=True)
        
        # Create default Dockerfile based on service
        dockerfile_path = os.path.join(self.dockerfile_dir, f"Dockerfile.{service}")
        
        if service == "data-collection":
            dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.data_collection.api:api", "--host", "0.0.0.0", "--port", "8000"]
"""
        elif service == "audio-conversion":
            dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "src.audio_conversion.api:api", "--host", "0.0.0.0", "--port", "8001"]
"""
        elif service == "rave-processing":
            dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["uvicorn", "src.rave_processing.api:api", "--host", "0.0.0.0", "--port", "8002"]
"""
        elif service == "vector-generation":
            dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["uvicorn", "src.vector_generation.api:api", "--host", "0.0.0.0", "--port", "8003"]
"""
        elif service == "laser-control":
            dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8004

CMD ["uvicorn", "src.laser_control.api:api", "--host", "0.0.0.0", "--port", "8004"]
"""
        elif service == "web":
            dockerfile_content = """
FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

EXPOSE 80

CMD ["npm", "start"]
"""
        else:
            dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src"]
"""
        
        # Write Dockerfile
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content.strip())
        
        logger.info(f"Created default Dockerfile for {service}")


class DockerComposeDeployer:
    """Class for deploying with Docker Compose."""
    
    def __init__(
        self,
        project_dir: str = ".",
        compose_file: str = "docker-compose.yml"
    ):
        """Initialize the Docker Compose deployer.
        
        Args:
            project_dir: Project directory
            compose_file: Docker Compose file
        """
        self.project_dir = project_dir
        self.compose_file = os.path.join(project_dir, compose_file)
    
    def create_compose_file(self) -> bool:
        """Create Docker Compose file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create Docker Compose file
            compose_content = """
version: '3'

services:
  influxdb:
    image: influxdb:2.0
    ports:
      - "8086:8086"
    volumes:
      - influxdb-data:/var/lib/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=password
      - DOCKER_INFLUXDB_INIT_ORG=solar-to-laser
      - DOCKER_INFLUXDB_INIT_BUCKET=solar_data

  postgres:
    image: postgres:13
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=solar_to_laser

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"

  data-collection:
    build:
      context: .
      dockerfile: docker/Dockerfile.data-collection
    ports:
      - "8000:8000"
    depends_on:
      - influxdb
      - rabbitmq
    environment:
      - INFLUXDB_HOST=influxdb
      - INFLUXDB_PORT=8086
      - INFLUXDB_DATABASE=solar_data
      - RABBITMQ_HOST=rabbitmq

  audio-conversion:
    build:
      context: .
      dockerfile: docker/Dockerfile.audio-conversion
    ports:
      - "8001:8001"
    depends_on:
      - postgres
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
      - POSTGRES_HOST=postgres

  rave-processing:
    build:
      context: .
      dockerfile: docker/Dockerfile.rave-processing
    ports:
      - "8002:8002"
    depends_on:
      - postgres
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
      - POSTGRES_HOST=postgres

  vector-generation:
    build:
      context: .
      dockerfile: docker/Dockerfile.vector-generation
    ports:
      - "8003:8003"
    depends_on:
      - postgres
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
      - POSTGRES_HOST=postgres

  laser-control:
    build:
      context: .
      dockerfile: docker/Dockerfile.laser-control
    ports:
      - "8004:8004"
    depends_on:
      - postgres
      - rabbitmq
    environment:
      - RABBITMQ_HOST=rabbitmq
      - POSTGRES_HOST=postgres

  web:
    build:
      context: .
      dockerfile: docker/Dockerfile.web
    ports:
      - "80:80"
    depends_on:
      - data-collection
      - audio-conversion
      - rave-processing
      - vector-generation
      - laser-control
    environment:
      - DATA_COLLECTION_URL=http://data-collection:8000
      - AUDIO_CONVERSION_URL=http://audio-conversion:8001
      - RAVE_PROCESSING_URL=http://rave-processing:8002
      - VECTOR_GENERATION_URL=http://vector-generation:8003
      - LASER_CONTROL_URL=http://laser-control:8004

volumes:
  influxdb-data:
  postgres-data:
"""
            
            # Write Docker Compose file
            with open(self.compose_file, "w") as f:
                f.write(compose_content.strip())
            
            logger.info(f"Created Docker Compose file at {self.compose_file}")
            return True
        except Exception as e:
            logger.error(f"Error creating Docker Compose file: {e}")
            return False
    
    def up(self, detach: bool = True) -> bool:
        """Start services with Docker Compose.
        
        Args:
            detach: Run in detached mode
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if Docker Compose file exists
            if not os.path.exists(self.compose_file):
                # Create Docker Compose file
                if not self.create_compose_file():
                    return False
            
            # Start services
            cmd = ["docker-compose", "-f", self.compose_file, "up"]
            if detach:
                cmd.append("-d")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error starting services: {result.stderr}")
                return False
            
            logger.info("Started services with Docker Compose")
            return True
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            return False
    
    def down(self) -> bool:
        """Stop services with Docker Compose.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if Docker Compose file exists
            if not os.path.exists(self.compose_file):
                logger.error(f"Docker Compose file not found: {self.compose_file}")
                return False
            
            # Stop services
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "down"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error stopping services: {result.stderr}")
                return False
            
            logger.info("Stopped services with Docker Compose")
            return True
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
            return False
    
    def logs(self, service: Optional[str] = None, follow: bool = False) -> str:
        """Get logs from Docker Compose services.
        
        Args:
            service: Service name (None for all)
            follow: Follow log output
        
        Returns:
            str: Log output
        """
        try:
            # Check if Docker Compose file exists
            if not os.path.exists(self.compose_file):
                logger.error(f"Docker Compose file not found: {self.compose_file}")
                return "Docker Compose file not found"
            
            # Build command
            cmd = ["docker-compose", "-f", self.compose_file, "logs"]
            if follow:
                cmd.append("-f")
            if service:
                cmd.append(service)
            
            # Get logs
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error getting logs: {result.stderr}")
                return f"Error getting logs: {result.stderr}"
            
            return result.stdout
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return f"Error getting logs: {e}"