"""
Laser Control Module for the Solar-to-Laser System.

This module provides components for controlling laser projectors and converting vector graphics to ILDA format.
"""

from .controller import (
    LaserController,
    ILDAController,
    PangolinController,
    SimulationController,
)

from .ilda import (
    ILDAFile,
    convert_svg_to_ilda,
    parse_svg_path,
    simple_parse_svg_path,
)

from .api import (
    create_api,
    api,
    LaserControlParameters,
    LaserFileInfo,
    LaserDeviceInfo,
    LaserGenerationRequest,
    LaserGenerationResponse,
    LaserSendRequest,
    StatusResponse,
)

__all__ = [
    # Controller
    'LaserController',
    'ILDAController',
    'PangolinController',
    'SimulationController',
    
    # ILDA
    'ILDAFile',
    'convert_svg_to_ilda',
    'parse_svg_path',
    'simple_parse_svg_path',
    
    # API
    'create_api',
    'api',
    'LaserControlParameters',
    'LaserFileInfo',
    'LaserDeviceInfo',
    'LaserGenerationRequest',
    'LaserGenerationResponse',
    'LaserSendRequest',
    'StatusResponse',
]