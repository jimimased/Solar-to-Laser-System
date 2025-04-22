"""
Laser controller implementation.

This module provides classes for controlling laser projectors.
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import numpy as np

from ..common import LaserParameters

logger = logging.getLogger(__name__)


class LaserController:
    """Base class for laser controllers."""
    
    def send(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> bool:
        """Send vectors to the laser.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement send()")
    
    def convert_to_ilda(
        self,
        vectors: np.ndarray,
        file_path: str,
        parameters: Optional[LaserParameters] = None
    ) -> str:
        """Convert vectors to ILDA format and save to a file.
        
        Args:
            vectors: Vector coordinates (x, y)
            file_path: Path to save the file
            parameters: Laser parameters
        
        Returns:
            str: Path to the saved file
        """
        raise NotImplementedError("Subclasses must implement convert_to_ilda()")
    
    def simulate(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> np.ndarray:
        """Simulate laser projection.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            np.ndarray: Simulated image
        """
        raise NotImplementedError("Subclasses must implement simulate()")


class ILDAController(LaserController):
    """Controller for ILDA-compatible laser projectors."""
    
    def __init__(
        self,
        device_path: Optional[str] = None,
        frame_rate: int = 30,
        points_per_frame: int = 500,
        color_mode: str = "RGB",
        intensity: float = 0.8,
        safety_limits: Optional[Dict[str, float]] = None
    ):
        """Initialize the ILDA controller.
        
        Args:
            device_path: Path to the laser device
            frame_rate: Frame rate in frames per second
            points_per_frame: Number of points per frame
            color_mode: Color mode (RGB, etc.)
            intensity: Intensity of the laser
            safety_limits: Safety limits for the laser
        """
        self.device_path = device_path
        self.frame_rate = frame_rate
        self.points_per_frame = points_per_frame
        self.color_mode = color_mode
        self.intensity = intensity
        self.safety_limits = safety_limits or {
            "max_intensity": 1.0,
            "max_scan_rate": 30000,
            "min_blanking_time": 0.001
        }
        
        # Check if device exists
        self.device_available = False
        if device_path and os.path.exists(device_path):
            self.device_available = True
            logger.info(f"Laser device found at {device_path}")
        else:
            logger.warning("No laser device found, operating in simulation mode")
    
    def send(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> bool:
        """Send vectors to the laser.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        # Check if device is available
        if not self.device_available:
            logger.warning("No laser device available, cannot send vectors")
            return False
        
        try:
            # Convert vectors to ILDA format
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ild") as temp_file:
                temp_path = temp_file.name
            
            self.convert_to_ilda(vectors, temp_path, parameters)
            
            # Send ILDA file to laser
            # In a real implementation, this would use a library to send the ILDA file to the laser
            # For now, we'll just log that we would send it
            logger.info(f"Would send ILDA file {temp_path} to laser device {self.device_path}")
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return True
        except Exception as e:
            logger.error(f"Error sending vectors to laser: {e}")
            return False
    
    def convert_to_ilda(
        self,
        vectors: np.ndarray,
        file_path: str,
        parameters: Optional[LaserParameters] = None
    ) -> str:
        """Convert vectors to ILDA format and save to a file.
        
        Args:
            vectors: Vector coordinates (x, y)
            file_path: Path to save the file
            parameters: Laser parameters
        
        Returns:
            str: Path to the saved file
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Import ILDA module
            from .ilda import ILDAFile
            
            # Create ILDA file
            ilda_file = ILDAFile()
            
            # Add frame to ILDA file
            ilda_file.add_frame(
                vectors,
                color_mode=parameters.color_mode,
                frame_rate=parameters.frame_rate,
                points_per_frame=parameters.points_per_frame,
                intensity=parameters.intensity
            )
            
            # Save ILDA file
            ilda_file.save(file_path)
            
            logger.info(f"Saved ILDA file to {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error converting vectors to ILDA: {e}")
            raise
    
    def simulate(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> np.ndarray:
        """Simulate laser projection.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            np.ndarray: Simulated image
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        try:
            # Create a blank image
            width, height = 800, 600
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Normalize vectors to [0, 1] range
            vectors_norm = self._normalize_vectors(vectors)
            
            # Scale to image dimensions
            vectors_scaled = np.zeros_like(vectors_norm)
            vectors_scaled[:, 0] = vectors_norm[:, 0] * (width - 1)
            vectors_scaled[:, 1] = vectors_norm[:, 1] * (height - 1)
            
            # Convert to integer coordinates
            vectors_int = vectors_scaled.astype(np.int32)
            
            # Draw lines
            for i in range(len(vectors_int) - 1):
                x1, y1 = vectors_int[i]
                x2, y2 = vectors_int[i + 1]
                
                # Draw line using Bresenham's algorithm
                self._draw_line(image, x1, y1, x2, y2, parameters.intensity)
            
            return image
        except Exception as e:
            logger.error(f"Error simulating laser projection: {e}")
            raise
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to [0, 1] range.
        
        Args:
            vectors: Vector coordinates (x, y)
        
        Returns:
            np.ndarray: Normalized vector coordinates
        """
        # Find min and max values
        min_vals = np.min(vectors, axis=0)
        max_vals = np.max(vectors, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        # Normalize to [0, 1]
        vectors_norm = (vectors - min_vals) / range_vals
        
        return vectors_norm
    
    def _draw_line(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        intensity: float
    ) -> None:
        """Draw a line on the image using Bresenham's algorithm.
        
        Args:
            image: Image to draw on
            x1, y1: Start point
            x2, y2: End point
            intensity: Line intensity
        """
        import cv2
        
        # Convert intensity to color
        color = (
            int(0 * intensity * 255),  # B
            int(1 * intensity * 255),  # G
            int(0 * intensity * 255)   # R
        )
        
        # Draw line
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


class PangolinController(LaserController):
    """Controller for Pangolin-compatible laser projectors."""
    
    def __init__(
        self,
        device_path: Optional[str] = None,
        frame_rate: int = 30,
        points_per_frame: int = 500,
        color_mode: str = "RGB",
        intensity: float = 0.8,
        safety_limits: Optional[Dict[str, float]] = None
    ):
        """Initialize the Pangolin controller.
        
        Args:
            device_path: Path to the laser device
            frame_rate: Frame rate in frames per second
            points_per_frame: Number of points per frame
            color_mode: Color mode (RGB, etc.)
            intensity: Intensity of the laser
            safety_limits: Safety limits for the laser
        """
        self.device_path = device_path
        self.frame_rate = frame_rate
        self.points_per_frame = points_per_frame
        self.color_mode = color_mode
        self.intensity = intensity
        self.safety_limits = safety_limits or {
            "max_intensity": 1.0,
            "max_scan_rate": 30000,
            "min_blanking_time": 0.001
        }
        
        # Check if device exists
        self.device_available = False
        if device_path and os.path.exists(device_path):
            self.device_available = True
            logger.info(f"Laser device found at {device_path}")
        else:
            logger.warning("No laser device found, operating in simulation mode")
    
    def send(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> bool:
        """Send vectors to the laser.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        # Check if device is available
        if not self.device_available:
            logger.warning("No laser device available, cannot send vectors")
            return False
        
        try:
            # In a real implementation, this would use the Pangolin SDK to send the vectors to the laser
            # For now, we'll just log that we would send it
            logger.info(f"Would send {len(vectors)} vectors to Pangolin laser device {self.device_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error sending vectors to laser: {e}")
            return False
    
    def convert_to_ilda(
        self,
        vectors: np.ndarray,
        file_path: str,
        parameters: Optional[LaserParameters] = None
    ) -> str:
        """Convert vectors to ILDA format and save to a file.
        
        Args:
            vectors: Vector coordinates (x, y)
            file_path: Path to save the file
            parameters: Laser parameters
        
        Returns:
            str: Path to the saved file
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Import ILDA module
            from .ilda import ILDAFile
            
            # Create ILDA file
            ilda_file = ILDAFile()
            
            # Add frame to ILDA file
            ilda_file.add_frame(
                vectors,
                color_mode=parameters.color_mode,
                frame_rate=parameters.frame_rate,
                points_per_frame=parameters.points_per_frame,
                intensity=parameters.intensity
            )
            
            # Save ILDA file
            ilda_file.save(file_path)
            
            logger.info(f"Saved ILDA file to {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error converting vectors to ILDA: {e}")
            raise
    
    def simulate(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> np.ndarray:
        """Simulate laser projection.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            np.ndarray: Simulated image
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        try:
            # Create a blank image
            width, height = 800, 600
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Normalize vectors to [0, 1] range
            vectors_norm = self._normalize_vectors(vectors)
            
            # Scale to image dimensions
            vectors_scaled = np.zeros_like(vectors_norm)
            vectors_scaled[:, 0] = vectors_norm[:, 0] * (width - 1)
            vectors_scaled[:, 1] = vectors_norm[:, 1] * (height - 1)
            
            # Convert to integer coordinates
            vectors_int = vectors_scaled.astype(np.int32)
            
            # Draw lines
            for i in range(len(vectors_int) - 1):
                x1, y1 = vectors_int[i]
                x2, y2 = vectors_int[i + 1]
                
                # Draw line using Bresenham's algorithm
                self._draw_line(image, x1, y1, x2, y2, parameters.intensity)
            
            return image
        except Exception as e:
            logger.error(f"Error simulating laser projection: {e}")
            raise
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to [0, 1] range.
        
        Args:
            vectors: Vector coordinates (x, y)
        
        Returns:
            np.ndarray: Normalized vector coordinates
        """
        # Find min and max values
        min_vals = np.min(vectors, axis=0)
        max_vals = np.max(vectors, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        # Normalize to [0, 1]
        vectors_norm = (vectors - min_vals) / range_vals
        
        return vectors_norm
    
    def _draw_line(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        intensity: float
    ) -> None:
        """Draw a line on the image using Bresenham's algorithm.
        
        Args:
            image: Image to draw on
            x1, y1: Start point
            x2, y2: End point
            intensity: Line intensity
        """
        import cv2
        
        # Convert intensity to color
        color = (
            int(0 * intensity * 255),  # B
            int(0 * intensity * 255),  # G
            int(1 * intensity * 255)   # R
        )
        
        # Draw line
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


class SimulationController(LaserController):
    """Controller for simulating laser projection."""
    
    def __init__(
        self,
        frame_rate: int = 30,
        points_per_frame: int = 500,
        color_mode: str = "RGB",
        intensity: float = 0.8,
        safety_limits: Optional[Dict[str, float]] = None
    ):
        """Initialize the simulation controller.
        
        Args:
            frame_rate: Frame rate in frames per second
            points_per_frame: Number of points per frame
            color_mode: Color mode (RGB, etc.)
            intensity: Intensity of the laser
            safety_limits: Safety limits for the laser
        """
        self.frame_rate = frame_rate
        self.points_per_frame = points_per_frame
        self.color_mode = color_mode
        self.intensity = intensity
        self.safety_limits = safety_limits or {
            "max_intensity": 1.0,
            "max_scan_rate": 30000,
            "min_blanking_time": 0.001
        }
    
    def send(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> bool:
        """Send vectors to the simulated laser.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        try:
            # Simulate laser projection
            image = self.simulate(vectors, parameters)
            
            # In a real implementation, this would display the image
            # For now, we'll just log that we would display it
            logger.info(f"Would display simulated laser projection with {len(vectors)} vectors")
            
            return True
        except Exception as e:
            logger.error(f"Error sending vectors to simulated laser: {e}")
            return False
    
    def convert_to_ilda(
        self,
        vectors: np.ndarray,
        file_path: str,
        parameters: Optional[LaserParameters] = None
    ) -> str:
        """Convert vectors to ILDA format and save to a file.
        
        Args:
            vectors: Vector coordinates (x, y)
            file_path: Path to save the file
            parameters: Laser parameters
        
        Returns:
            str: Path to the saved file
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Import ILDA module
            from .ilda import ILDAFile
            
            # Create ILDA file
            ilda_file = ILDAFile()
            
            # Add frame to ILDA file
            ilda_file.add_frame(
                vectors,
                color_mode=parameters.color_mode,
                frame_rate=parameters.frame_rate,
                points_per_frame=parameters.points_per_frame,
                intensity=parameters.intensity
            )
            
            # Save ILDA file
            ilda_file.save(file_path)
            
            logger.info(f"Saved ILDA file to {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error converting vectors to ILDA: {e}")
            raise
    
    def simulate(
        self,
        vectors: np.ndarray,
        parameters: Optional[LaserParameters] = None
    ) -> np.ndarray:
        """Simulate laser projection.
        
        Args:
            vectors: Vector coordinates (x, y)
            parameters: Laser parameters
        
        Returns:
            np.ndarray: Simulated image
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = LaserParameters()
        
        try:
            # Create a blank image
            width, height = 800, 600
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Normalize vectors to [0, 1] range
            vectors_norm = self._normalize_vectors(vectors)
            
            # Scale to image dimensions
            vectors_scaled = np.zeros_like(vectors_norm)
            vectors_scaled[:, 0] = vectors_norm[:, 0] * (width - 1)
            vectors_scaled[:, 1] = vectors_norm[:, 1] * (height - 1)
            
            # Convert to integer coordinates
            vectors_int = vectors_scaled.astype(np.int32)
            
            # Draw lines
            for i in range(len(vectors_int) - 1):
                x1, y1 = vectors_int[i]
                x2, y2 = vectors_int[i + 1]
                
                # Draw line using OpenCV
                self._draw_line(image, x1, y1, x2, y2, parameters.intensity)
            
            return image
        except Exception as e:
            logger.error(f"Error simulating laser projection: {e}")
            raise
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to [0, 1] range.
        
        Args:
            vectors: Vector coordinates (x, y)
        
        Returns:
            np.ndarray: Normalized vector coordinates
        """
        # Find min and max values
        min_vals = np.min(vectors, axis=0)
        max_vals = np.max(vectors, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        # Normalize to [0, 1]
        vectors_norm = (vectors - min_vals) / range_vals
        
        return vectors_norm
    
    def _draw_line(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        intensity: float
    ) -> None:
        """Draw a line on the image.
        
        Args:
            image: Image to draw on
            x1, y1: Start point
            x2, y2: End point
            intensity: Line intensity
        """
        import cv2
        
        # Convert intensity to color
        color = (
            int(0 * intensity * 255),  # B
            int(1 * intensity * 255),  # G
            int(0 * intensity * 255)   # R
        )
        
        # Draw line
        cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)