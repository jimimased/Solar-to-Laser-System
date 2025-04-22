"""
API endpoints for the RAVE processing module.

This module provides FastAPI endpoints for processing audio using RAVE.
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
import librosa
import torch

from ..common import AudioParameters
from .model import create_rave_model, load_rave_model
from .processor import RAVEProcessor, SpectralProcessor

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses

class RAVEProcessingParameters(BaseModel):
    """Pydantic model for RAVE processing parameters."""
    
    latent_dim: int = 128
    channels: int = 128
    n_residual_blocks: int = 4
    segment_size: int = 16384
    hop_size: int = 8192
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "latent_dim": 128,
                "channels": 128,
                "n_residual_blocks": 4,
                "segment_size": 16384,
                "hop_size": 8192
            }
        }


class ProcessedAudioInfo(BaseModel):
    """Pydantic model for processed audio information."""
    
    id: int
    filename: str
    sample_rate: int
    duration: float
    channels: int
    created_at: str
    original_filename: str
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "id": 1,
                "filename": "processed_audio_2025-04-22.wav",
                "sample_rate": 44100,
                "duration": 60.0,
                "channels": 1,
                "created_at": "2025-04-22T12:00:00",
                "original_filename": "original_audio.wav"
            }
        }


class FeatureExtractionResponse(BaseModel):
    """Pydantic model for feature extraction response."""
    
    id: int
    filename: str
    features_url: str
    feature_shape: List[int]
    created_at: str
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "id": 1,
                "filename": "features_2025-04-22.npy",
                "features_url": "/api/rave/features/1",
                "feature_shape": [100, 128],
                "created_at": "2025-04-22T12:00:00"
            }
        }


class TrainingResponse(BaseModel):
    """Pydantic model for training response."""
    
    model_id: int
    model_name: str
    epochs: int
    loss: float
    created_at: str
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "model_id": 1,
                "model_name": "rave_model_2025-04-22.pt",
                "epochs": 100,
                "loss": 0.05,
                "created_at": "2025-04-22T12:00:00"
            }
        }


class StatusResponse(BaseModel):
    """Pydantic model for status response."""
    
    status: str
    message: str
    timestamp: str


# API application

def create_api(
    audio_storage_path: str = "data/rave",
    model_storage_path: str = "models/rave",
    features_storage_path: str = "data/features",
    database_url: Optional[str] = None
) -> FastAPI:
    """Create a FastAPI application for the RAVE processing API.
    
    Args:
        audio_storage_path: Path to store processed audio files
        model_storage_path: Path to store trained models
        features_storage_path: Path to store extracted features
        database_url: URL for the database
    
    Returns:
        FastAPI: FastAPI application
    """
    app = FastAPI(
        title="RAVE Processing API",
        description="API for processing audio using RAVE",
        version="1.0.0"
    )
    
    # Ensure storage directories exist
    os.makedirs(audio_storage_path, exist_ok=True)
    os.makedirs(model_storage_path, exist_ok=True)
    os.makedirs(features_storage_path, exist_ok=True)
    
    # In-memory database for processed files (for demo purposes)
    # In a real implementation, this would be a database
    processed_files = []
    feature_files = []
    trained_models = []
    
    # Default processor
    default_processor = RAVEProcessor(
        model_path=os.path.join(model_storage_path, "default_model.pt") if os.path.exists(os.path.join(model_storage_path, "default_model.pt")) else None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Factory function for creating processors
    def create_processor(parameters: RAVEProcessingParameters) -> RAVEProcessor:
        """Create a RAVE processor based on parameters.
        
        Args:
            parameters: RAVE processing parameters
        
        Returns:
            RAVEProcessor: RAVE processor
        """
        # Find the latest trained model
        model_path = None
        if trained_models:
            latest_model = max(trained_models, key=lambda m: m["created_at"])
            model_path = latest_model["file_path"]
        
        return RAVEProcessor(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            latent_dim=parameters.latent_dim,
            channels=parameters.channels,
            n_residual_blocks=parameters.n_residual_blocks,
            segment_size=parameters.segment_size,
            hop_size=parameters.hop_size
        )
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "message": "RAVE Processing API",
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
    
    @app.post("/api/rave/process", response_model=ProcessedAudioInfo, tags=["Processing"])
    async def process_audio(
        file: UploadFile = File(...),
        parameters: Optional[RAVEProcessingParameters] = None
    ):
        """Process audio using RAVE."""
        try:
            # Use default parameters if not provided
            if parameters is None:
                parameters = RAVEProcessingParameters()
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Load audio file
            audio_data, sample_rate = librosa.load(temp_path, sr=None)
            
            # Create processor
            processor = create_processor(parameters)
            
            # Process audio
            processed_audio, processed_sr = processor.process(audio_data, sample_rate)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_audio_{timestamp}.wav"
            file_path = os.path.join(audio_storage_path, filename)
            
            # Save processed audio
            processor.save(processed_audio, processed_sr, file_path)
            
            # Get audio duration and channels
            if len(processed_audio.shape) > 1:
                # Multi-channel audio
                duration = processed_audio.shape[1] / processed_sr
                channels = processed_audio.shape[0]
            else:
                # Mono audio
                duration = len(processed_audio) / processed_sr
                channels = 1
            
            # Add to database
            file_id = len(processed_files) + 1
            file_info = {
                "id": file_id,
                "filename": filename,
                "sample_rate": processed_sr,
                "duration": duration,
                "channels": channels,
                "created_at": datetime.now().isoformat(),
                "original_filename": file.filename,
                "file_path": file_path
            }
            processed_files.append(file_info)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Create response
            return ProcessedAudioInfo(
                id=file_id,
                filename=filename,
                sample_rate=processed_sr,
                duration=duration,
                channels=channels,
                created_at=file_info["created_at"],
                original_filename=file.filename
            )
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/rave/files", response_model=List[ProcessedAudioInfo], tags=["Files"])
    async def get_processed_files():
        """Get list of processed audio files."""
        return [
            ProcessedAudioInfo(
                id=file["id"],
                filename=file["filename"],
                sample_rate=file["sample_rate"],
                duration=file["duration"],
                channels=file["channels"],
                created_at=file["created_at"],
                original_filename=file["original_filename"]
            )
            for file in processed_files
        ]
    
    @app.get("/api/rave/file/{file_id}", tags=["Files"])
    async def get_processed_file(file_id: int):
        """Get processed audio file by ID."""
        try:
            # Find file in database
            file = next((f for f in processed_files if f["id"] == file_id), None)
            
            if not file:
                raise HTTPException(status_code=404, detail=f"Processed file with ID {file_id} not found")
            
            # Return file
            return FileResponse(
                file["file_path"],
                media_type="audio/wav",
                filename=file["filename"]
            )
        except Exception as e:
            logger.error(f"Error getting processed file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/rave/extract", response_model=FeatureExtractionResponse, tags=["Features"])
    async def extract_features(
        file: UploadFile = File(...),
        parameters: Optional[RAVEProcessingParameters] = None
    ):
        """Extract features from audio using RAVE."""
        try:
            # Use default parameters if not provided
            if parameters is None:
                parameters = RAVEProcessingParameters()
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Load audio file
            audio_data, sample_rate = librosa.load(temp_path, sr=None)
            
            # Create processor
            processor = create_processor(parameters)
            
            # Extract features
            features = processor.extract_features(audio_data, sample_rate)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"features_{timestamp}.npy"
            file_path = os.path.join(features_storage_path, filename)
            
            # Save features
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, features)
            
            # Add to database
            file_id = len(feature_files) + 1
            file_info = {
                "id": file_id,
                "filename": filename,
                "feature_shape": list(features.shape),
                "created_at": datetime.now().isoformat(),
                "original_filename": file.filename,
                "file_path": file_path
            }
            feature_files.append(file_info)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Create response
            return FeatureExtractionResponse(
                id=file_id,
                filename=filename,
                features_url=f"/api/rave/features/{file_id}",
                feature_shape=file_info["feature_shape"],
                created_at=file_info["created_at"]
            )
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/rave/features/{file_id}", tags=["Features"])
    async def get_features(file_id: int):
        """Get extracted features by ID."""
        try:
            # Find file in database
            file = next((f for f in feature_files if f["id"] == file_id), None)
            
            if not file:
                raise HTTPException(status_code=404, detail=f"Feature file with ID {file_id} not found")
            
            # Load features
            features = np.load(file["file_path"])
            
            # Return features as JSON
            return JSONResponse(content={
                "features": features.tolist(),
                "shape": file["feature_shape"]
            })
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/rave/train", response_model=TrainingResponse, tags=["Training"])
    async def train_model(
        file: UploadFile = File(...),
        epochs: int = Form(100),
        batch_size: int = Form(16),
        learning_rate: float = Form(0.0001),
        kl_weight: float = Form(0.01),
        parameters: Optional[RAVEProcessingParameters] = None,
        background_tasks: BackgroundTasks = None
    ):
        """Train a RAVE model on audio data."""
        try:
            # Use default parameters if not provided
            if parameters is None:
                parameters = RAVEProcessingParameters()
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Load audio file
            audio_data, sample_rate = librosa.load(temp_path, sr=None)
            
            # Create processor
            processor = create_processor(parameters)
            
            # Generate model filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"rave_model_{timestamp}.pt"
            model_path = os.path.join(model_storage_path, model_name)
            
            # Add to database
            model_id = len(trained_models) + 1
            model_info = {
                "id": model_id,
                "model_name": model_name,
                "epochs": epochs,
                "loss": 0.0,  # Will be updated after training
                "created_at": datetime.now().isoformat(),
                "file_path": model_path
            }
            trained_models.append(model_info)
            
            # Define training callback
            def training_callback(epoch, losses):
                model_info["loss"] = losses["total_loss"]
            
            # Train model in background
            def train_model_task():
                try:
                    processor.train(
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        kl_weight=kl_weight,
                        save_path=model_path,
                        callback=training_callback
                    )
                    logger.info(f"Model training completed: {model_path}")
                except Exception as e:
                    logger.error(f"Error training model: {e}")
            
            # Start training in background
            if background_tasks:
                background_tasks.add_task(train_model_task)
            else:
                # For testing, train synchronously
                train_model_task()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Create response
            return TrainingResponse(
                model_id=model_id,
                model_name=model_name,
                epochs=epochs,
                loss=model_info["loss"],
                created_at=model_info["created_at"]
            )
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/rave/models", response_model=List[TrainingResponse], tags=["Models"])
    async def get_models():
        """Get list of trained models."""
        return [
            TrainingResponse(
                model_id=model["id"],
                model_name=model["model_name"],
                epochs=model["epochs"],
                loss=model["loss"],
                created_at=model["created_at"]
            )
            for model in trained_models
        ]
    
    @app.post("/api/rave/interpolate", response_model=ProcessedAudioInfo, tags=["Processing"])
    async def interpolate_audio(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...),
        steps: int = Form(10),
        parameters: Optional[RAVEProcessingParameters] = None
    ):
        """Interpolate between two audio files using RAVE."""
        try:
            # Use default parameters if not provided
            if parameters is None:
                parameters = RAVEProcessingParameters()
            
            # Save uploaded files to temporary locations
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file1.filename)[1]) as temp_file1:
                temp_file1.write(await file1.read())
                temp_path1 = temp_file1.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file2.filename)[1]) as temp_file2:
                temp_file2.write(await file2.read())
                temp_path2 = temp_file2.name
            
            # Load audio files
            audio_data1, sample_rate1 = librosa.load(temp_path1, sr=None)
            audio_data2, sample_rate2 = librosa.load(temp_path2, sr=None)
            
            # Create processor
            processor = create_processor(parameters)
            
            # Interpolate
            interpolated_audio = processor.interpolate(audio_data1, audio_data2, steps)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interpolated_audio_{timestamp}.wav"
            file_path = os.path.join(audio_storage_path, filename)
            
            # Save interpolated audio
            processor.save(interpolated_audio, 44100, file_path)
            
            # Get audio duration
            duration = len(interpolated_audio) / 44100
            
            # Add to database
            file_id = len(processed_files) + 1
            file_info = {
                "id": file_id,
                "filename": filename,
                "sample_rate": 44100,
                "duration": duration,
                "channels": 1,
                "created_at": datetime.now().isoformat(),
                "original_filename": f"interpolation_{file1.filename}_{file2.filename}",
                "file_path": file_path
            }
            processed_files.append(file_info)
            
            # Clean up temporary files
            os.unlink(temp_path1)
            os.unlink(temp_path2)
            
            # Create response
            return ProcessedAudioInfo(
                id=file_id,
                filename=filename,
                sample_rate=44100,
                duration=duration,
                channels=1,
                created_at=file_info["created_at"],
                original_filename=file_info["original_filename"]
            )
        except Exception as e:
            logger.error(f"Error interpolating audio: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default API instance
api = create_api()