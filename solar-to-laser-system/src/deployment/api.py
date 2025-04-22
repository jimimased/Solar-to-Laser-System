"""
API endpoints for the deployment module.

This module provides FastAPI endpoints for deploying the system.
"""

import os
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .kubernetes import KubernetesDeployer
from .docker import DockerDeployer, DockerComposeDeployer

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses

class KubernetesDeploymentParameters(BaseModel):
    """Pydantic model for Kubernetes deployment parameters."""
    
    namespace: str = "solar-to-laser-system"
    registry: str = "ghcr.io/yourusername"
    config_path: Optional[str] = None
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "namespace": "solar-to-laser-system",
                "registry": "ghcr.io/yourusername",
                "config_path": None
            }
        }


class DockerDeploymentParameters(BaseModel):
    """Pydantic model for Docker deployment parameters."""
    
    registry: str = "ghcr.io/yourusername"
    tag: str = "latest"
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "registry": "ghcr.io/yourusername",
                "tag": "latest"
            }
        }


class DeploymentStatusResponse(BaseModel):
    """Pydantic model for deployment status response."""
    
    status: str
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Deployment completed successfully",
                "timestamp": "2025-04-22T12:00:00",
                "details": {
                    "namespace": "solar-to-laser-system",
                    "services": ["data-collection", "audio-conversion", "rave-processing", "vector-generation", "laser-control", "web"]
                }
            }
        }


class LogsResponse(BaseModel):
    """Pydantic model for logs response."""
    
    service: str
    logs: str
    timestamp: str
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "service": "data-collection",
                "logs": "2025-04-22T12:00:00 INFO Starting service...",
                "timestamp": "2025-04-22T12:00:00"
            }
        }


# API application

def create_api() -> FastAPI:
    """Create a FastAPI application for the deployment API.
    
    Returns:
        FastAPI: FastAPI application
    """
    app = FastAPI(
        title="Deployment API",
        description="API for deploying the Solar-to-Laser System",
        version="1.0.0"
    )
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "message": "Deployment API",
            "version": "1.0.0",
            "documentation": "/docs"
        }
    
    @app.get("/status", response_model=DeploymentStatusResponse, tags=["General"])
    async def get_status():
        """Get API status."""
        return DeploymentStatusResponse(
            status="ok",
            message="API is running",
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/api/deployment/kubernetes", response_model=DeploymentStatusResponse, tags=["Kubernetes"])
    async def deploy_to_kubernetes(
        parameters: KubernetesDeploymentParameters,
        background_tasks: BackgroundTasks
    ):
        """Deploy to Kubernetes."""
        try:
            # Create Kubernetes deployer
            deployer = KubernetesDeployer(
                namespace=parameters.namespace,
                registry=parameters.registry,
                config_path=parameters.config_path
            )
            
            # Define deployment task
            def deployment_task():
                try:
                    # Deploy all components
                    success = deployer.deploy_all()
                    
                    if not success:
                        logger.error("Deployment to Kubernetes failed")
                except Exception as e:
                    logger.error(f"Error in deployment task: {e}")
            
            # Add task to background tasks
            background_tasks.add_task(deployment_task)
            
            return DeploymentStatusResponse(
                status="pending",
                message="Deployment to Kubernetes started",
                timestamp=datetime.now().isoformat(),
                details={
                    "namespace": parameters.namespace,
                    "registry": parameters.registry
                }
            )
        except Exception as e:
            logger.error(f"Error deploying to Kubernetes: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/deployment/kubernetes/status", response_model=DeploymentStatusResponse, tags=["Kubernetes"])
    async def get_kubernetes_status(
        namespace: str = "solar-to-laser-system",
        config_path: Optional[str] = None
    ):
        """Get Kubernetes deployment status."""
        try:
            # Create Kubernetes deployer
            deployer = KubernetesDeployer(
                namespace=namespace,
                config_path=config_path
            )
            
            # Get deployment status
            status = deployer.get_status()
            
            return DeploymentStatusResponse(
                status=status.get("status", "error"),
                message="Kubernetes deployment status",
                timestamp=datetime.now().isoformat(),
                details=status
            )
        except Exception as e:
            logger.error(f"Error getting Kubernetes status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/deployment/docker", response_model=DeploymentStatusResponse, tags=["Docker"])
    async def build_docker_images(
        parameters: DockerDeploymentParameters,
        background_tasks: BackgroundTasks
    ):
        """Build Docker images."""
        try:
            # Create Docker deployer
            deployer = DockerDeployer(
                registry=parameters.registry
            )
            
            # Define build task
            def build_task():
                try:
                    # Build and push all images
                    success = deployer.build_and_push_all(parameters.tag)
                    
                    if not success:
                        logger.error("Building Docker images failed")
                except Exception as e:
                    logger.error(f"Error in build task: {e}")
            
            # Add task to background tasks
            background_tasks.add_task(build_task)
            
            return DeploymentStatusResponse(
                status="pending",
                message="Building Docker images started",
                timestamp=datetime.now().isoformat(),
                details={
                    "registry": parameters.registry,
                    "tag": parameters.tag
                }
            )
        except Exception as e:
            logger.error(f"Error building Docker images: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/deployment/docker-compose/up", response_model=DeploymentStatusResponse, tags=["Docker Compose"])
    async def docker_compose_up(
        background_tasks: BackgroundTasks,
        detach: bool = True
    ):
        """Start services with Docker Compose."""
        try:
            # Create Docker Compose deployer
            deployer = DockerComposeDeployer()
            
            # Define up task
            def up_task():
                try:
                    # Start services
                    success = deployer.up(detach)
                    
                    if not success:
                        logger.error("Starting services with Docker Compose failed")
                except Exception as e:
                    logger.error(f"Error in up task: {e}")
            
            # Add task to background tasks
            background_tasks.add_task(up_task)
            
            return DeploymentStatusResponse(
                status="pending",
                message="Starting services with Docker Compose",
                timestamp=datetime.now().isoformat(),
                details={
                    "detach": detach
                }
            )
        except Exception as e:
            logger.error(f"Error starting services with Docker Compose: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/deployment/docker-compose/down", response_model=DeploymentStatusResponse, tags=["Docker Compose"])
    async def docker_compose_down():
        """Stop services with Docker Compose."""
        try:
            # Create Docker Compose deployer
            deployer = DockerComposeDeployer()
            
            # Stop services
            success = deployer.down()
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to stop services")
            
            return DeploymentStatusResponse(
                status="success",
                message="Stopped services with Docker Compose",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error stopping services with Docker Compose: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/deployment/docker-compose/logs", response_model=LogsResponse, tags=["Docker Compose"])
    async def docker_compose_logs(
        service: Optional[str] = None,
        follow: bool = False
    ):
        """Get logs from Docker Compose services."""
        try:
            # Create Docker Compose deployer
            deployer = DockerComposeDeployer()
            
            # Get logs
            logs = deployer.logs(service, follow)
            
            return LogsResponse(
                service=service or "all",
                logs=logs,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error getting logs from Docker Compose: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/deployment/create-compose-file", response_model=DeploymentStatusResponse, tags=["Docker Compose"])
    async def create_compose_file():
        """Create Docker Compose file."""
        try:
            # Create Docker Compose deployer
            deployer = DockerComposeDeployer()
            
            # Create Docker Compose file
            success = deployer.create_compose_file()
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to create Docker Compose file")
            
            return DeploymentStatusResponse(
                status="success",
                message="Created Docker Compose file",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error creating Docker Compose file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/deployment/create-dockerfiles", response_model=DeploymentStatusResponse, tags=["Docker"])
    async def create_dockerfiles():
        """Create Dockerfiles for all services."""
        try:
            # Create Docker deployer
            deployer = DockerDeployer()
            
            # List of services
            services = [
                "data-collection",
                "audio-conversion",
                "rave-processing",
                "vector-generation",
                "laser-control",
                "web"
            ]
            
            # Create Dockerfile for each service
            for service in services:
                deployer._create_default_dockerfile(service)
            
            return DeploymentStatusResponse(
                status="success",
                message="Created Dockerfiles for all services",
                timestamp=datetime.now().isoformat(),
                details={
                    "services": services
                }
            )
        except Exception as e:
            logger.error(f"Error creating Dockerfiles: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default API instance
api = create_api()