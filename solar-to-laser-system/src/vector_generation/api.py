"""
API endpoints for the vector generation module.

This module provides FastAPI endpoints for generating vector graphics.
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

from ..common import VectorParameters
from .generator import VectorGenerator
from .mapping import DirectMappingGenerator, DimensionalityReductionGenerator
from .patterns import PatternGenerator, LissajousGenerator, SpiralGenerator, HarmonographGenerator

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses

class VectorGenerationParameters(BaseModel):
    """Pydantic model for vector generation parameters."""
    
    generator_type: str = "direct"  # direct, pca, tsne, lissajous, spiral, harmonograph
    smoothing_factor: float = 0.8
    interpolation_method: str = "cubic"
    scaling_factor: float = 100.0
    normalization_range: List[float] = [-1.0, 1.0]
    path_simplification: float = 0.02
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "generator_type": "direct",
                "smoothing_factor": 0.8,
                "interpolation_method": "cubic",
                "scaling_factor": 100.0,
                "normalization_range": [-1.0, 1.0],
                "path_simplification": 0.02
            }
        }


class VectorFileInfo(BaseModel):
    """Pydantic model for vector file information."""
    
    id: int
    filename: str
    format: str
    point_count: int
    created_at: str
    original_filename: str
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "id": 1,
                "filename": "vector_2025-04-22.svg",
                "format": "svg",
                "point_count": 1000,
                "created_at": "2025-04-22T12:00:00",
                "original_filename": "features.npy"
            }
        }


class VectorGenerationRequest(BaseModel):
    """Pydantic model for vector generation request."""
    
    features: List[List[float]]
    parameters: VectorGenerationParameters = Field(default_factory=VectorGenerationParameters)
    output_format: str = "svg"
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "features": [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5],
                    [0.3, 0.4, 0.5, 0.6]
                ],
                "parameters": {
                    "generator_type": "direct",
                    "smoothing_factor": 0.8,
                    "interpolation_method": "cubic",
                    "scaling_factor": 100.0,
                    "normalization_range": [-1.0, 1.0],
                    "path_simplification": 0.02
                },
                "output_format": "svg"
            }
        }


class VectorGenerationResponse(BaseModel):
    """Pydantic model for vector generation response."""
    
    file_id: int
    filename: str
    download_url: str
    preview_url: str
    point_count: int
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "file_id": 1,
                "filename": "vector_2025-04-22.svg",
                "download_url": "/api/vector/file/1",
                "preview_url": "/api/vector/preview/1",
                "point_count": 1000
            }
        }


class StatusResponse(BaseModel):
    """Pydantic model for status response."""
    
    status: str
    message: str
    timestamp: str


# API application

def create_api(
    vector_storage_path: str = "data/vector",
    preview_storage_path: str = "data/preview",
    database_url: Optional[str] = None
) -> FastAPI:
    """Create a FastAPI application for the vector generation API.
    
    Args:
        vector_storage_path: Path to store vector files
        preview_storage_path: Path to store preview images
        database_url: URL for the database
    
    Returns:
        FastAPI: FastAPI application
    """
    app = FastAPI(
        title="Vector Generation API",
        description="API for generating vector graphics from audio features",
        version="1.0.0"
    )
    
    # Ensure storage directories exist
    os.makedirs(vector_storage_path, exist_ok=True)
    os.makedirs(preview_storage_path, exist_ok=True)
    
    # In-memory database for vector files (for demo purposes)
    # In a real implementation, this would be a database
    vector_files = []
    
    # Factory function for creating generators
    def create_generator(parameters: VectorGenerationParameters) -> VectorGenerator:
        """Create a vector generator based on parameters.
        
        Args:
            parameters: Vector generation parameters
        
        Returns:
            VectorGenerator: Vector generator
        """
        generator_type = parameters.generator_type.lower()
        
        if generator_type == "direct":
            return DirectMappingGenerator(
                smoothing_factor=parameters.smoothing_factor,
                interpolation_method=parameters.interpolation_method
            )
        elif generator_type == "pca":
            return DimensionalityReductionGenerator(
                method="pca",
                smoothing_factor=parameters.smoothing_factor,
                interpolation_method=parameters.interpolation_method
            )
        elif generator_type == "tsne":
            return DimensionalityReductionGenerator(
                method="tsne",
                smoothing_factor=parameters.smoothing_factor,
                interpolation_method=parameters.interpolation_method
            )
        elif generator_type == "lissajous":
            return LissajousGenerator(
                smoothing_factor=parameters.smoothing_factor,
                interpolation_method=parameters.interpolation_method
            )
        elif generator_type == "spiral":
            return SpiralGenerator(
                smoothing_factor=parameters.smoothing_factor,
                interpolation_method=parameters.interpolation_method
            )
        elif generator_type == "harmonograph":
            return HarmonographGenerator(
                smoothing_factor=parameters.smoothing_factor,
                interpolation_method=parameters.interpolation_method
            )
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "message": "Vector Generation API",
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
    
    @app.post("/api/vector/generate", response_model=VectorGenerationResponse, tags=["Generation"])
    async def generate_vector(request: VectorGenerationRequest):
        """Generate vector graphics from features."""
        try:
            # Convert features to numpy array
            features = np.array(request.features)
            
            # Create vector parameters
            vector_params = VectorParameters(
                smoothing_factor=request.parameters.smoothing_factor,
                interpolation_method=request.parameters.interpolation_method,
                scaling_factor=request.parameters.scaling_factor,
                normalization_range=tuple(request.parameters.normalization_range),
                path_simplification=request.parameters.path_simplification
            )
            
            # Create generator
            generator = create_generator(request.parameters)
            
            # Generate vectors
            vectors = generator.generate(features, vector_params)
            
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vector_filename = f"vector_{timestamp}.{request.output_format}"
            preview_filename = f"preview_{timestamp}.png"
            
            vector_path = os.path.join(vector_storage_path, vector_filename)
            preview_path = os.path.join(preview_storage_path, preview_filename)
            
            # Save vector file
            if request.output_format == "svg":
                generator.save_svg(vectors, vector_path)
            else:
                # Default to SVG if format not supported
                vector_filename = f"vector_{timestamp}.svg"
                vector_path = os.path.join(vector_storage_path, vector_filename)
                generator.save_svg(vectors, vector_path)
            
            # Save preview image
            generator.save_preview(vectors, preview_path)
            
            # Add to database
            file_id = len(vector_files) + 1
            file_info = {
                "id": file_id,
                "filename": vector_filename,
                "format": request.output_format,
                "point_count": len(vectors),
                "created_at": datetime.now().isoformat(),
                "original_filename": "features.npy",  # Placeholder
                "vector_path": vector_path,
                "preview_path": preview_path
            }
            vector_files.append(file_info)
            
            # Create response
            return VectorGenerationResponse(
                file_id=file_id,
                filename=vector_filename,
                download_url=f"/api/vector/file/{file_id}",
                preview_url=f"/api/vector/preview/{file_id}",
                point_count=len(vectors)
            )
        except Exception as e:
            logger.error(f"Error generating vector: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/vector/generate-from-file", response_model=VectorGenerationResponse, tags=["Generation"])
    async def generate_vector_from_file(
        file: UploadFile = File(...),
        generator_type: str = Form("direct"),
        smoothing_factor: float = Form(0.8),
        interpolation_method: str = Form("cubic"),
        scaling_factor: float = Form(100.0),
        normalization_range_min: float = Form(-1.0),
        normalization_range_max: float = Form(1.0),
        path_simplification: float = Form(0.02),
        output_format: str = Form("svg")
    ):
        """Generate vector graphics from a features file."""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Load features from file
            features = np.load(temp_path)
            
            # Create parameters
            parameters = VectorGenerationParameters(
                generator_type=generator_type,
                smoothing_factor=smoothing_factor,
                interpolation_method=interpolation_method,
                scaling_factor=scaling_factor,
                normalization_range=[normalization_range_min, normalization_range_max],
                path_simplification=path_simplification
            )
            
            # Create vector parameters
            vector_params = VectorParameters(
                smoothing_factor=parameters.smoothing_factor,
                interpolation_method=parameters.interpolation_method,
                scaling_factor=parameters.scaling_factor,
                normalization_range=tuple(parameters.normalization_range),
                path_simplification=parameters.path_simplification
            )
            
            # Create generator
            generator = create_generator(parameters)
            
            # Generate vectors
            vectors = generator.generate(features, vector_params)
            
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vector_filename = f"vector_{timestamp}.{output_format}"
            preview_filename = f"preview_{timestamp}.png"
            
            vector_path = os.path.join(vector_storage_path, vector_filename)
            preview_path = os.path.join(preview_storage_path, preview_filename)
            
            # Save vector file
            if output_format == "svg":
                generator.save_svg(vectors, vector_path)
            else:
                # Default to SVG if format not supported
                vector_filename = f"vector_{timestamp}.svg"
                vector_path = os.path.join(vector_storage_path, vector_filename)
                generator.save_svg(vectors, vector_path)
            
            # Save preview image
            generator.save_preview(vectors, preview_path)
            
            # Add to database
            file_id = len(vector_files) + 1
            file_info = {
                "id": file_id,
                "filename": vector_filename,
                "format": output_format,
                "point_count": len(vectors),
                "created_at": datetime.now().isoformat(),
                "original_filename": file.filename,
                "vector_path": vector_path,
                "preview_path": preview_path
            }
            vector_files.append(file_info)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Create response
            return VectorGenerationResponse(
                file_id=file_id,
                filename=vector_filename,
                download_url=f"/api/vector/file/{file_id}",
                preview_url=f"/api/vector/preview/{file_id}",
                point_count=len(vectors)
            )
        except Exception as e:
            logger.error(f"Error generating vector from file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/vector/files", response_model=List[VectorFileInfo], tags=["Files"])
    async def get_vector_files():
        """Get list of vector files."""
        return [
            VectorFileInfo(
                id=file["id"],
                filename=file["filename"],
                format=file["format"],
                point_count=file["point_count"],
                created_at=file["created_at"],
                original_filename=file["original_filename"]
            )
            for file in vector_files
        ]
    
    @app.get("/api/vector/file/{file_id}", tags=["Files"])
    async def get_vector_file(file_id: int):
        """Get vector file by ID."""
        try:
            # Find file in database
            file = next((f for f in vector_files if f["id"] == file_id), None)
            
            if not file:
                raise HTTPException(status_code=404, detail=f"Vector file with ID {file_id} not found")
            
            # Return file
            return FileResponse(
                file["vector_path"],
                media_type=f"image/svg+xml" if file["format"] == "svg" else f"application/{file['format']}",
                filename=file["filename"]
            )
        except Exception as e:
            logger.error(f"Error getting vector file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/vector/preview/{file_id}", tags=["Files"])
    async def get_vector_preview(file_id: int):
        """Get vector preview image by ID."""
        try:
            # Find file in database
            file = next((f for f in vector_files if f["id"] == file_id), None)
            
            if not file:
                raise HTTPException(status_code=404, detail=f"Vector file with ID {file_id} not found")
            
            # Return preview image
            return FileResponse(
                file["preview_path"],
                media_type="image/png",
                filename=os.path.basename(file["preview_path"])
            )
        except Exception as e:
            logger.error(f"Error getting vector preview: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default API instance
api = create_api()