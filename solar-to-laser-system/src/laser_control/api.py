"""
API endpoints for the laser control module.

This module provides FastAPI endpoints for controlling laser projectors.
"""

import os
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import numpy as np

from ..common import LaserParameters
from .controller import LaserController, ILDAController, PangolinController, SimulationController
from .ilda import ILDAFile, convert_svg_to_ilda

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses

class LaserControlParameters(BaseModel):
    """Pydantic model for laser control parameters."""
    
    format: str = "ILDA"
    frame_rate: int = 30
    points_per_frame: int = 500
    color_mode: str = "RGB"
    intensity: float = 0.8
    safety_limits: Dict[str, float] = {
        "max_intensity": 1.0,
        "max_scan_rate": 30000,
        "min_blanking_time": 0.001
    }
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "format": "ILDA",
                "frame_rate": 30,
                "points_per_frame": 500,
                "color_mode": "RGB",
                "intensity": 0.8,
                "safety_limits": {
                    "max_intensity": 1.0,
                    "max_scan_rate": 30000,
                    "min_blanking_time": 0.001
                }
            }
        }


class LaserFileInfo(BaseModel):
    """Pydantic model for laser file information."""
    
    id: int
    filename: str
    format: str
    frame_count: int
    created_at: str
    original_filename: str
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "id": 1,
                "filename": "laser_2025-04-22.ild",
                "format": "ILDA",
                "frame_count": 1,
                "created_at": "2025-04-22T12:00:00",
                "original_filename": "vector.svg"
            }
        }


class LaserDeviceInfo(BaseModel):
    """Pydantic model for laser device information."""
    
    id: int
    name: str
    type: str
    status: str
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Simulation Laser",
                "type": "Simulation",
                "status": "Available"
            }
        }


class LaserGenerationRequest(BaseModel):
    """Pydantic model for laser generation request."""
    
    vectors: List[List[float]]
    parameters: LaserControlParameters = Field(default_factory=LaserControlParameters)
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "vectors": [
                    [0.1, 0.2],
                    [0.2, 0.3],
                    [0.3, 0.4]
                ],
                "parameters": {
                    "format": "ILDA",
                    "frame_rate": 30,
                    "points_per_frame": 500,
                    "color_mode": "RGB",
                    "intensity": 0.8,
                    "safety_limits": {
                        "max_intensity": 1.0,
                        "max_scan_rate": 30000,
                        "min_blanking_time": 0.001
                    }
                }
            }
        }


class LaserGenerationResponse(BaseModel):
    """Pydantic model for laser generation response."""
    
    file_id: int
    filename: str
    download_url: str
    preview_url: str
    frame_count: int
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "file_id": 1,
                "filename": "laser_2025-04-22.ild",
                "download_url": "/api/laser/file/1",
                "preview_url": "/api/laser/preview/1",
                "frame_count": 1
            }
        }


class LaserSendRequest(BaseModel):
    """Pydantic model for laser send request."""
    
    file_id: int
    device_id: int
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "file_id": 1,
                "device_id": 1
            }
        }


class StatusResponse(BaseModel):
    """Pydantic model for status response."""
    
    status: str
    message: str
    timestamp: str


# API application

def create_api(
    laser_storage_path: str = "data/laser",
    preview_storage_path: str = "data/preview",
    database_url: Optional[str] = None
) -> FastAPI:
    """Create a FastAPI application for the laser control API.
    
    Args:
        laser_storage_path: Path to store laser files
        preview_storage_path: Path to store preview images
        database_url: URL for the database
    
    Returns:
        FastAPI: FastAPI application
    """
    app = FastAPI(
        title="Laser Control API",
        description="API for controlling laser projectors",
        version="1.0.0"
    )
    
    # Ensure storage directories exist
    os.makedirs(laser_storage_path, exist_ok=True)
    os.makedirs(preview_storage_path, exist_ok=True)
    
    # In-memory database for laser files and devices (for demo purposes)
    # In a real implementation, this would be a database
    laser_files = []
    laser_devices = [
        {
            "id": 1,
            "name": "Simulation Laser",
            "type": "Simulation",
            "status": "Available",
            "controller": SimulationController()
        }
    ]
    
    # Factory function for creating controllers
    def create_controller(device_id: int, parameters: LaserControlParameters) -> LaserController:
        """Create a laser controller based on device ID and parameters.
        
        Args:
            device_id: Device ID
            parameters: Laser control parameters
        
        Returns:
            LaserController: Laser controller
        """
        # Find device
        device = next((d for d in laser_devices if d["id"] == device_id), None)
        if not device:
            raise ValueError(f"Invalid device ID: {device_id}")
        
        # Return existing controller
        return device["controller"]
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "message": "Laser Control API",
            "version": "1.0.0",
            "documentation": "/docs"
        }
    
    @app.get("/status", response_model=StatusResponse, tags=["General"])
    async def get_status():
        """Get API status."""
        return StatusResponse(
            status="ok",
            message="API is running",
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/api/laser/generate", response_model=LaserGenerationResponse, tags=["Generation"])
    async def generate_laser(request: LaserGenerationRequest):
        """Generate laser control file from vectors."""
        try:
            # Convert vectors to numpy array
            vectors = np.array(request.vectors)
            
            # Create laser parameters
            laser_params = LaserParameters(
                format=request.parameters.format,
                frame_rate=request.parameters.frame_rate,
                points_per_frame=request.parameters.points_per_frame,
                color_mode=request.parameters.color_mode,
                intensity=request.parameters.intensity,
                safety_limits=request.parameters.safety_limits
            )
            
            # Create controller
            controller = SimulationController(
                frame_rate=laser_params.frame_rate,
                points_per_frame=laser_params.points_per_frame,
                color_mode=laser_params.color_mode,
                intensity=laser_params.intensity,
                safety_limits=laser_params.safety_limits
            )
            
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            laser_filename = f"laser_{timestamp}.ild"
            preview_filename = f"preview_{timestamp}.png"
            
            laser_path = os.path.join(laser_storage_path, laser_filename)
            preview_path = os.path.join(preview_storage_path, preview_filename)
            
            # Convert to ILDA format
            controller.convert_to_ilda(vectors, laser_path, laser_params)
            
            # Generate preview
            preview_image = controller.simulate(vectors, laser_params)
            
            # Save preview image
            import cv2
            cv2.imwrite(preview_path, preview_image)
            
            # Add to database
            file_id = len(laser_files) + 1
            file_info = {
                "id": file_id,
                "filename": laser_filename,
                "format": laser_params.format,
                "frame_count": 1,
                "created_at": datetime.now().isoformat(),
                "original_filename": "vectors.npy",  # Placeholder
                "laser_path": laser_path,
                "preview_path": preview_path
            }
            laser_files.append(file_info)
            
            # Create response
            return LaserGenerationResponse(
                file_id=file_id,
                filename=laser_filename,
                download_url=f"/api/laser/file/{file_id}",
                preview_url=f"/api/laser/preview/{file_id}",
                frame_count=1
            )
        except Exception as e:
            logger.error(f"Error generating laser file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/laser/generate-from-file", response_model=LaserGenerationResponse, tags=["Generation"])
    async def generate_laser_from_file(
        file: UploadFile = File(...),
        format: str = Form("ILDA"),
        frame_rate: int = Form(30),
        points_per_frame: int = Form(500),
        color_mode: str = Form("RGB"),
        intensity: float = Form(0.8)
    ):
        """Generate laser control file from a vector file."""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            laser_filename = f"laser_{timestamp}.ild"
            preview_filename = f"preview_{timestamp}.png"
            
            laser_path = os.path.join(laser_storage_path, laser_filename)
            preview_path = os.path.join(preview_storage_path, preview_filename)
            
            # Check file type
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext == ".svg":
                # Convert SVG to ILDA
                convert_svg_to_ilda(
                    temp_path,
                    laser_path,
                    color_mode=color_mode,
                    frame_rate=frame_rate,
                    points_per_frame=points_per_frame,
                    intensity=intensity
                )
                
                # Load ILDA file to get vectors for preview
                ilda_file = ILDAFile()
                ilda_file.load(laser_path)
                vectors = ilda_file.get_vectors()
            elif file_ext == ".npy":
                # Load numpy array
                vectors = np.load(temp_path)
                
                # Create controller
                controller = SimulationController(
                    frame_rate=frame_rate,
                    points_per_frame=points_per_frame,
                    color_mode=color_mode,
                    intensity=intensity
                )
                
                # Convert to ILDA format
                controller.convert_to_ilda(
                    vectors,
                    laser_path,
                    LaserParameters(
                        format=format,
                        frame_rate=frame_rate,
                        points_per_frame=points_per_frame,
                        color_mode=color_mode,
                        intensity=intensity
                    )
                )
            else:
                # Unsupported file type
                os.unlink(temp_path)
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
            
            # Generate preview
            controller = SimulationController(
                frame_rate=frame_rate,
                points_per_frame=points_per_frame,
                color_mode=color_mode,
                intensity=intensity
            )
            
            preview_image = controller.simulate(
                vectors,
                LaserParameters(
                    format=format,
                    frame_rate=frame_rate,
                    points_per_frame=points_per_frame,
                    color_mode=color_mode,
                    intensity=intensity
                )
            )
            
            # Save preview image
            import cv2
            cv2.imwrite(preview_path, preview_image)
            
            # Add to database
            file_id = len(laser_files) + 1
            file_info = {
                "id": file_id,
                "filename": laser_filename,
                "format": format,
                "frame_count": 1,
                "created_at": datetime.now().isoformat(),
                "original_filename": file.filename,
                "laser_path": laser_path,
                "preview_path": preview_path
            }
            laser_files.append(file_info)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Create response
            return LaserGenerationResponse(
                file_id=file_id,
                filename=laser_filename,
                download_url=f"/api/laser/file/{file_id}",
                preview_url=f"/api/laser/preview/{file_id}",
                frame_count=1
            )
        except Exception as e:
            logger.error(f"Error generating laser file from file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/laser/files", response_model=List[LaserFileInfo], tags=["Files"])
    async def get_laser_files():
        """Get list of laser files."""
        return [
            LaserFileInfo(
                id=file["id"],
                filename=file["filename"],
                format=file["format"],
                frame_count=file["frame_count"],
                created_at=file["created_at"],
                original_filename=file["original_filename"]
            )
            for file in laser_files
        ]
    
    @app.get("/api/laser/file/{file_id}", tags=["Files"])
    async def get_laser_file(file_id: int):
        """Get laser file by ID."""
        try:
            # Find file in database
            file = next((f for f in laser_files if f["id"] == file_id), None)
            
            if not file:
                raise HTTPException(status_code=404, detail=f"Laser file with ID {file_id} not found")
            
            # Return file
            return FileResponse(
                file["laser_path"],
                media_type="application/octet-stream",
                filename=file["filename"]
            )
        except Exception as e:
            logger.error(f"Error getting laser file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/laser/preview/{file_id}", tags=["Files"])
    async def get_laser_preview(file_id: int):
        """Get laser preview image by ID."""
        try:
            # Find file in database
            file = next((f for f in laser_files if f["id"] == file_id), None)
            
            if not file:
                raise HTTPException(status_code=404, detail=f"Laser file with ID {file_id} not found")
            
            # Return preview image
            return FileResponse(
                file["preview_path"],
                media_type="image/png",
                filename=os.path.basename(file["preview_path"])
            )
        except Exception as e:
            logger.error(f"Error getting laser preview: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/laser/devices", response_model=List[LaserDeviceInfo], tags=["Devices"])
    async def get_laser_devices():
        """Get list of laser devices."""
        return [
            LaserDeviceInfo(
                id=device["id"],
                name=device["name"],
                type=device["type"],
                status=device["status"]
            )
            for device in laser_devices
        ]
    
    @app.post("/api/laser/send/{device_id}", response_model=StatusResponse, tags=["Control"])
    async def send_to_laser(device_id: int, file_id: int):
        """Send laser file to a device."""
        try:
            # Find device
            device = next((d for d in laser_devices if d["id"] == device_id), None)
            if not device:
                raise HTTPException(status_code=404, detail=f"Laser device with ID {device_id} not found")
            
            # Find file
            file = next((f for f in laser_files if f["id"] == file_id), None)
            if not file:
                raise HTTPException(status_code=404, detail=f"Laser file with ID {file_id} not found")
            
            # Get controller
            controller = device["controller"]
            
            # Load ILDA file
            ilda_file = ILDAFile()
            ilda_file.load(file["laser_path"])
            
            # Get vectors
            vectors = ilda_file.get_vectors()
            
            # Send to laser
            success = controller.send(vectors)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to send to laser")
            
            return StatusResponse(
                status="success",
                message=f"Sent laser file {file_id} to device {device_id}",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error sending to laser: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/laser/simulate", response_model=StatusResponse, tags=["Simulation"])
    async def simulate_laser(request: LaserGenerationRequest):
        """Simulate laser projection."""
        try:
            # Convert vectors to numpy array
            vectors = np.array(request.vectors)
            
            # Create laser parameters
            laser_params = LaserParameters(
                format=request.parameters.format,
                frame_rate=request.parameters.frame_rate,
                points_per_frame=request.parameters.points_per_frame,
                color_mode=request.parameters.color_mode,
                intensity=request.parameters.intensity,
                safety_limits=request.parameters.safety_limits
            )
            
            # Create controller
            controller = SimulationController(
                frame_rate=laser_params.frame_rate,
                points_per_frame=laser_params.points_per_frame,
                color_mode=laser_params.color_mode,
                intensity=laser_params.intensity,
                safety_limits=laser_params.safety_limits
            )
            
            # Simulate laser projection
            preview_image = controller.simulate(vectors, laser_params)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_filename = f"preview_{timestamp}.png"
            preview_path = os.path.join(preview_storage_path, preview_filename)
            
            # Save preview image
            import cv2
            cv2.imwrite(preview_path, preview_image)
            
            return StatusResponse(
                status="success",
                message=f"Simulated laser projection and saved preview to {preview_filename}",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error simulating laser: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default API instance
api = create_api()