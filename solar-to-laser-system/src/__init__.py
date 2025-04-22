"""
Solar-to-Laser System.

A comprehensive system that converts solar panel data into laser imagery through audio processing.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import modules
from . import common
from . import data_collection
from . import audio_conversion
from . import rave_processing
from . import vector_generation
from . import laser_control
from . import deployment

__all__ = [
    'common',
    'data_collection',
    'audio_conversion',
    'rave_processing',
    'vector_generation',
    'laser_control',
    'deployment',
]