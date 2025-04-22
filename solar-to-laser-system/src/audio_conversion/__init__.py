"""
Audio Conversion Module for the Solar-to-Laser System.

This module provides components for converting solar data to audio.
"""

from .converter import (
    AudioConverter,
    DirectMappingConverter,
)

from .synthesis import (
    FMSynthesisConverter,
    GranularSynthesisConverter,
    MultiChannelConverter,
)

from .api import (
    create_api,
    api,
    AudioConversionParameters,
    AudioFileInfo,
    AudioConversionRequest,
    AudioConversionResponse,
    StatusResponse,
)

__all__ = [
    # Converter
    'AudioConverter',
    'DirectMappingConverter',
    
    # Synthesis
    'FMSynthesisConverter',
    'GranularSynthesisConverter',
    'MultiChannelConverter',
    
    # API
    'create_api',
    'api',
    'AudioConversionParameters',
    'AudioFileInfo',
    'AudioConversionRequest',
    'AudioConversionResponse',
    'StatusResponse',
]