"""
Vector Generation Module for the Solar-to-Laser System.

This module provides components for generating vector graphics from audio features.
"""

from .generator import (
    VectorGenerator,
)

from .mapping import (
    DirectMappingGenerator,
    DimensionalityReductionGenerator,
)

from .patterns import (
    PatternGenerator,
    LissajousGenerator,
    SpiralGenerator,
    HarmonographGenerator,
    MultiPatternGenerator,
)

from .api import (
    create_api,
    api,
    VectorGenerationParameters,
    VectorFileInfo,
    VectorGenerationRequest,
    VectorGenerationResponse,
    StatusResponse,
)

__all__ = [
    # Generator
    'VectorGenerator',
    
    # Mapping
    'DirectMappingGenerator',
    'DimensionalityReductionGenerator',
    
    # Patterns
    'PatternGenerator',
    'LissajousGenerator',
    'SpiralGenerator',
    'HarmonographGenerator',
    'MultiPatternGenerator',
    
    # API
    'create_api',
    'api',
    'VectorGenerationParameters',
    'VectorFileInfo',
    'VectorGenerationRequest',
    'VectorGenerationResponse',
    'StatusResponse',
]