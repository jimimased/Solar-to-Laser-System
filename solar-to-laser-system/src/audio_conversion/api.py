"""
API endpoints for the audio conversion module.

This module provides FastAPI endpoints for converting solar data to audio.
"""

import os
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import numpy as np

from ..common import SolarData, AudioParameters
from ..data_collection.api import SolarDataModel
from .converter import AudioConverter, DirectMappingConverter
from .synthesis import FMSynthesisConverter, GranularSynthesisConverter, MultiChannelConverter

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses

class AudioConversionParameters(BaseModel):
    """Pydantic model for audio conversion parameters."""
    
    sample_rate: int = 44100
    duration: float = 1.0
    converter_type: str = "direct"  # direct, fm, granular, multi
    voltage_range: List[float] = [0.0, 48.0]
    current_range: List[float] = [0.0, 10.0]
    frequency_range: List[float] = [220.0, 880.0]
    amplitude_range: List[float] = [0.0, 0.9]
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "sample_rate": 44100,
                "duration": 1.0,
                "converter_type": "direct",
                "voltage_range": [0.0, 48.0],
                "current_range": [0.0, 10.0],
                "frequency_range": [220.0, 880.0],
                "amplitude_range": [0.0, 0.9]
            }
        }


class AudioFileInfo(BaseModel):
    """Pydantic model for audio file information."""
    
    id: int
    filename: str
    sample_rate: int
    duration: float
    channels: int
    created_at: str
    solar_data_start: Optional[str] = None
    solar_data_end: Optional[str] = None
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "id": 1,
                "filename": "solar_audio_2025-04-22.wav",
                "sample_rate": 44100,
                "duration": 60.0,
                "channels": 1,
                "created_at": "2025-04-22T12:00:00",
                "solar_data_start": "2025-04-22T11:00:00",
                "solar_data_end": "2025-04-22T12:00:00"
            }
        }


class AudioConversionRequest(BaseModel):
    """Pydantic model for audio conversion request."""
    
    solar_data: List[SolarDataModel]
    parameters: AudioConversionParameters = Field(default_factory=AudioConversionParameters)
    output_format: str = "wav"
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "solar_data": [
                    {
                        "timestamp": "2025-04-22T12:00:00",
                        "voltage": 24.5,
                        "current": 3.2,
                        "power": 78.4,
                        "temperature": 25.3,
                        "irradiance": 850.0,
                        "metadata": {
                            "source": "panel_1",
                            "weather": {
                                "temperature": 28.5,
                                "humidity": 65.0,
                                "cloud_cover": 10.0
                            }
                        }
                    }
                ],
                "parameters": {
                    "sample_rate": 44100,
                    "duration": 1.0,
                    "converter_type": "direct",
                    "voltage_range": [0.0, 48.0],
                    "current_range": [0.0, 10.0],
                    "frequency_range": [220.0, 880.0],
                    "amplitude_range": [0.0, 0.9]
                },
                "output_format": "wav"
            }
        }


class AudioConversionResponse(BaseModel):
    """Pydantic model for audio conversion response."""
    
    file_id: int
    filename: str
    download_url: str
    sample_rate: int
    duration: float
    channels: int
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "file_id": 1,
                "filename": "solar_audio_2025-04-22.wav",
                "download_url": "/api/audio/file/1",
                "sample_rate": 44100,
                "duration": 60.0,
                "channels": 1
            }
        }


class StatusResponse(BaseModel):
    """Pydantic model for status response."""
    
    status: str
    message: str
    timestamp: str


# API application

def create_api(
    audio_storage_path: str = "data/audio",
    database_url: Optional[str] = None
) -> FastAPI:
    """Create a FastAPI application for the audio conversion API.
    
    Args:
        audio_storage_path: Path to store audio files
        database_url: URL for the database
    
    Returns:
        FastAPI: FastAPI application
    """
    app = FastAPI(
        title="Audio Conversion API",
        description="API for converting solar data to audio",
        version="1.0.0"
    )
    
    # Ensure audio storage directory exists
    os.makedirs(audio_storage_path, exist_ok=True)
    
    # In-memory database for audio files (for demo purposes)
    # In a real implementation, this would be a database
    audio_files = []
    
    # Factory function for creating converters
    def create_converter(parameters: AudioConversionParameters) -> AudioConverter:
        """Create an audio converter based on parameters.
        
        Args:
            parameters: Audio conversion parameters
        
        Returns:
            AudioConverter: Audio converter
        """
        converter_type = parameters.converter_type.lower()
        
        if converter_type == "direct":
            return DirectMappingConverter(
                voltage_range=tuple(parameters.voltage_range),
                current_range=tuple(parameters.current_range),
                frequency_range=tuple(parameters.frequency_range),
                amplitude_range=tuple(parameters.amplitude_range)
            )
        elif converter_type == "fm":
            return FMSynthesisConverter(
                voltage_range=tuple(parameters.voltage_range),
                current_range=tuple(parameters.current_range),
                carrier_range=tuple(parameters.frequency_range),
                amplitude_range=tuple(parameters.amplitude_range)
            )
        elif converter_type == "granular":
            return GranularSynthesisConverter(
                voltage_range=tuple(parameters.voltage_range),
                current_range=tuple(parameters.current_range),
                amplitude_range=tuple(parameters.amplitude_range)
            )
        elif converter_type == "multi":
            # Create a multi-channel converter with direct and FM synthesis
            converters = [
                DirectMappingConverter(
                    voltage_range=tuple(parameters.voltage_range),
                    current_range=tuple(parameters.current_range),
                    frequency_range=tuple(parameters.frequency_range),
                    amplitude_range=tuple(parameters.amplitude_range)
                ),
                FMSynthesisConverter(
                    voltage_range=tuple(parameters.voltage_range),
                    current_range=tuple(parameters.current_range),
                    carrier_range=tuple(parameters.frequency_range),
                    amplitude_range=tuple(parameters.amplitude_range)
                )
            ]
            return MultiChannelConverter(
                converters=converters,
                voltage_range=tuple(parameters.voltage_range),
                current_range=tuple(parameters.current_range)
            )
        else:
            raise ValueError(f"Invalid converter type: {converter_type}")
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "message": "Audio Conversion API",
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
    
    @app.post("/api/audio/convert", response_model=AudioConversionResponse, tags=["Conversion"])
    async def convert_audio(request: AudioConversionRequest):
        """Convert solar data to audio."""
        try:
            # Convert solar data models to SolarData objects
            solar_data_list = [item.to_solar_data() for item in request.solar_data]
            
            # Create converter
            converter = create_converter(request.parameters)
            
            # Create audio parameters
            audio_params = AudioParameters(
                sample_rate=request.parameters.sample_rate,
                duration=request.parameters.duration,
                frequency_mapping=converter._map_voltage_to_frequency if hasattr(converter, "_map_voltage_to_frequency") else lambda x: x,
                amplitude_mapping=converter._map_current_to_amplitude if hasattr(converter, "_map_current_to_amplitude") else lambda x: x,
                timbre_mapping=converter._map_power_to_harmonics if hasattr(converter, "_map_power_to_harmonics") else lambda x: [1.0],
                temporal_mapping=lambda x: x
            )
            
            # Convert to audio
            audio_data, sample_rate = converter.convert(solar_data_list, audio_params)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solar_audio_{timestamp}.{request.output_format}"
            file_path = os.path.join(audio_storage_path, filename)
            
            # Save audio file
            converter.save(audio_data, sample_rate, file_path, request.output_format)
            
            # Get audio duration
            if len(audio_data.shape) > 1:
                # Multi-channel audio
                duration = audio_data.shape[1] / sample_rate
                channels = audio_data.shape[0]
            else:
                # Mono audio
                duration = len(audio_data) / sample_rate
                channels = 1
            
            # Add to database
            file_id = len(audio_files) + 1
            audio_file_info = {
                "id": file_id,
                "filename": filename,
                "sample_rate": sample_rate,
                "duration": duration,
                "channels": channels,
                "created_at": datetime.now().isoformat(),
                "solar_data_start": solar_data_list[0].timestamp.isoformat() if solar_data_list else None,
                "solar_data_end": solar_data_list[-1].timestamp.isoformat() if solar_data_list else None,
                "file_path": file_path
            }
            audio_files.append(audio_file_info)
            
            # Create response
            return AudioConversionResponse(
                file_id=file_id,
                filename=filename,
                download_url=f"/api/audio/file/{file_id}",
                sample_rate=sample_rate,
                duration=duration,
                channels=channels
            )
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/audio/files", response_model=List[AudioFileInfo], tags=["Files"])
    async def get_audio_files():
        """Get list of audio files."""
        return [
            AudioFileInfo(
                id=file["id"],
                filename=file["filename"],
                sample_rate=file["sample_rate"],
                duration=file["duration"],
                channels=file["channels"],
                created_at=file["created_at"],
                solar_data_start=file["solar_data_start"],
                solar_data_end=file["solar_data_end"]
            )
            for file in audio_files
        ]
    
    @app.get("/api/audio/file/{file_id}", tags=["Files"])
    async def get_audio_file(file_id: int):
        """Get audio file by ID."""
        try:
            # Find file in database
            file = next((f for f in audio_files if f["id"] == file_id), None)
            
            if not file:
                raise HTTPException(status_code=404, detail=f"Audio file with ID {file_id} not found")
            
            # Return file
            return FileResponse(
                file["file_path"],
                media_type=f"audio/{os.path.splitext(file['filename'])[1][1:]}",
                filename=file["filename"]
            )
        except Exception as e:
            logger.error(f"Error getting audio file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/audio/process", response_model=AudioConversionResponse, tags=["Processing"])
    async def process_audio(
        file: UploadFile = File(...),
        converter_type: str = Form("direct"),
        sample_rate: int = Form(44100),
        duration: float = Form(1.0),
        output_format: str = Form("wav")
    ):
        """Process an existing audio file using RAVE."""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # TODO: Implement RAVE processing
            # For now, just copy the file
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_audio_{timestamp}.{output_format}"
            file_path = os.path.join(audio_storage_path, filename)
            
            # Copy file
            import shutil
            shutil.copy(temp_path, file_path)
            
            # Get audio info
            import librosa
            audio_data, sr = librosa.load(file_path, sr=None)
            
            # Get audio duration and channels
            if len(audio_data.shape) > 1:
                # Multi-channel audio
                duration = audio_data.shape[1] / sr
                channels = audio_data.shape[0]
            else:
                # Mono audio
                duration = len(audio_data) / sr
                channels = 1
            
            # Add to database
            file_id = len(audio_files) + 1
            audio_file_info = {
                "id": file_id,
                "filename": filename,
                "sample_rate": sr,
                "duration": duration,
                "channels": channels,
                "created_at": datetime.now().isoformat(),
                "solar_data_start": None,
                "solar_data_end": None,
                "file_path": file_path
            }
            audio_files.append(audio_file_info)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Create response
            return AudioConversionResponse(
                file_id=file_id,
                filename=filename,
                download_url=f"/api/audio/file/{file_id}",
                sample_rate=sr,
                duration=duration,
                channels=channels
            )
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/audio/preview", tags=["Conversion"])
    async def preview_audio(request: AudioConversionRequest):
        """Generate a preview of the audio conversion."""
        try:
            # Convert solar data models to SolarData objects
            solar_data_list = [item.to_solar_data() for item in request.solar_data]
            
            # Create converter
            converter = create_converter(request.parameters)
            
            # Create audio parameters
            audio_params = AudioParameters(
                sample_rate=request.parameters.sample_rate,
                duration=request.parameters.duration,
                frequency_mapping=converter._map_voltage_to_frequency if hasattr(converter, "_map_voltage_to_frequency") else lambda x: x,
                amplitude_mapping=converter._map_current_to_amplitude if hasattr(converter, "_map_current_to_amplitude") else lambda x: x,
                timbre_mapping=converter._map_power_to_harmonics if hasattr(converter, "_map_power_to_harmonics") else lambda x: [1.0],
                temporal_mapping=lambda x: x
            )
            
            # Convert to audio
            audio_data, sample_rate = converter.convert(solar_data_list, audio_params)
            
            # Generate temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.output_format}") as temp_file:
                temp_path = temp_file.name
            
            # Save audio file
            converter.save(audio_data, sample_rate, temp_path, request.output_format)
            
            # Return file
            return FileResponse(
                temp_path,
                media_type=f"audio/{request.output_format}",
                filename=f"preview.{request.output_format}",
                background=BackgroundTasks().add_task(lambda: os.unlink(temp_path))
            )
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default API instance
api = create_api()