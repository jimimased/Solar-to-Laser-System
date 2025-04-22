"""
RAVE Processing Module for the Solar-to-Laser System.

This module provides components for processing audio using RAVE (Realtime Audio Variational autoEncoder).
"""

from .model import (
    RAVE,
    Encoder,
    Decoder,
    ResidualBlock,
    ConvBlock,
    RAVELoss,
    create_rave_model,
    load_rave_model,
    save_rave_model,
)

from .processor import (
    AudioProcessor,
    RAVEProcessor,
    SpectralProcessor,
)

from .api import (
    create_api,
    api,
    RAVEProcessingParameters,
    ProcessedAudioInfo,
    FeatureExtractionResponse,
    TrainingResponse,
    StatusResponse,
)

__all__ = [
    # Model
    'RAVE',
    'Encoder',
    'Decoder',
    'ResidualBlock',
    'ConvBlock',
    'RAVELoss',
    'create_rave_model',
    'load_rave_model',
    'save_rave_model',
    
    # Processor
    'AudioProcessor',
    'RAVEProcessor',
    'SpectralProcessor',
    
    # API
    'create_api',
    'api',
    'RAVEProcessingParameters',
    'ProcessedAudioInfo',
    'FeatureExtractionResponse',
    'TrainingResponse',
    'StatusResponse',
]