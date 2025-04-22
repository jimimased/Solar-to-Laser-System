"""
ILDA format implementation.

This module provides classes for working with the ILDA (International Laser Display Association) format.
"""

import os
import logging
import struct
from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO

import numpy as np

logger = logging.getLogger(__name__)


class ILDAFile:
    """Class for working with ILDA files."""
    
    # ILDA format constants
    ILDA_HEADER_FORMAT = ">4s3xBBHHHB"
    ILDA_HEADER_SIZE = 16
    ILDA_3D_FORMAT = 0
    ILDA_2D_FORMAT = 1
    ILDA_COLOR_FORMAT = 2
    ILDA_3D_TRUE_COLOR_FORMAT = 4
    ILDA_2D_TRUE_COLOR_FORMAT = 5
    
    def __init__(self):
        """Initialize the ILDA file."""
        self.frames = []
        self.frame_formats = []
        self.frame_names = []
        self.frame_companies = []
        self.frame_point_counts = []
        self.frame_frame_numbers = []
        self.frame_total_frames = []
    
    def add_frame(
        self,
        vectors: np.ndarray,
        color_mode: str = "RGB",
        frame_rate: int = 30,
        points_per_frame: int = 500,
        intensity: float = 0.8,
        name: str = "SOLAR2LASER",
        company: str = "SOLAR2LASER"
    ) -> None:
        """Add a frame to the ILDA file.
        
        Args:
            vectors: Vector coordinates (x, y)
            color_mode: Color mode (RGB, etc.)
            frame_rate: Frame rate in frames per second
            points_per_frame: Number of points per frame
            intensity: Intensity of the laser
            name: Frame name
            company: Company name
        """
        # Normalize vectors to [-32768, 32767] range (16-bit signed integer)
        vectors_norm = self._normalize_vectors(vectors, (-32768, 32767))
        
        # Convert to integer coordinates
        vectors_int = vectors_norm.astype(np.int16)
        
        # Add z coordinate if needed
        if vectors_int.shape[1] == 2:
            z = np.zeros((vectors_int.shape[0], 1), dtype=np.int16)
            vectors_3d = np.hstack((vectors_int, z))
        else:
            vectors_3d = vectors_int
        
        # Add status byte (0 = normal point, 64 = last point in frame)
        status = np.zeros((vectors_3d.shape[0], 1), dtype=np.uint8)
        status[-1] = 64  # Last point
        
        # Add color (RGB)
        if color_mode == "RGB":
            # Create RGB colors (green for now)
            r = np.zeros((vectors_3d.shape[0], 1), dtype=np.uint8)
            g = np.ones((vectors_3d.shape[0], 1), dtype=np.uint8) * int(intensity * 255)
            b = np.zeros((vectors_3d.shape[0], 1), dtype=np.uint8)
            
            # Combine coordinates, status, and color
            frame_data = np.hstack((vectors_3d, status, r, g, b))
            
            # Set format to true color
            format_code = self.ILDA_3D_TRUE_COLOR_FORMAT
        else:
            # Combine coordinates and status
            frame_data = np.hstack((vectors_3d, status))
            
            # Set format to 3D
            format_code = self.ILDA_3D_FORMAT
        
        # Add frame to list
        self.frames.append(frame_data)
        self.frame_formats.append(format_code)
        self.frame_names.append(name[:8].ljust(8))
        self.frame_companies.append(company[:8].ljust(8))
        self.frame_point_counts.append(len(vectors))
        self.frame_frame_numbers.append(len(self.frames) - 1)
        self.frame_total_frames.append(len(self.frames))
    
    def save(self, file_path: str) -> None:
        """Save the ILDA file.
        
        Args:
            file_path: Path to save the file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Open file for writing
            with open(file_path, "wb") as f:
                # Write each frame
                for i, frame in enumerate(self.frames):
                    # Write header
                    f.write(struct.pack(
                        self.ILDA_HEADER_FORMAT,
                        b"ILDA",  # ILDA header
                        self.frame_formats[i],  # Format code
                        0,  # Unused
                        self.frame_point_counts[i],  # Number of points
                        self.frame_frame_numbers[i],  # Frame number
                        self.frame_total_frames[i],  # Total frames
                        0  # Unused
                    ))
                    
                    # Write frame name and company
                    f.write(self.frame_names[i].encode("ascii"))
                    f.write(self.frame_companies[i].encode("ascii"))
                    
                    # Write frame data
                    if self.frame_formats[i] == self.ILDA_3D_FORMAT:
                        # 3D format: X, Y, Z, Status
                        for point in frame:
                            f.write(struct.pack(
                                ">hhhB",
                                point[0],  # X
                                point[1],  # Y
                                point[2],  # Z
                                point[3]   # Status
                            ))
                    elif self.frame_formats[i] == self.ILDA_3D_TRUE_COLOR_FORMAT:
                        # 3D true color format: X, Y, Z, Status, R, G, B
                        for point in frame:
                            f.write(struct.pack(
                                ">hhhBBBB",
                                point[0],  # X
                                point[1],  # Y
                                point[2],  # Z
                                point[3],  # Status
                                point[4],  # R
                                point[5],  # G
                                point[6]   # B
                            ))
            
            logger.info(f"Saved ILDA file to {file_path}")
        except Exception as e:
            logger.error(f"Error saving ILDA file: {e}")
            raise
    
    def load(self, file_path: str) -> None:
        """Load an ILDA file.
        
        Args:
            file_path: Path to the file
        """
        try:
            # Open file for reading
            with open(file_path, "rb") as f:
                # Clear existing frames
                self.frames = []
                self.frame_formats = []
                self.frame_names = []
                self.frame_companies = []
                self.frame_point_counts = []
                self.frame_frame_numbers = []
                self.frame_total_frames = []
                
                # Read frames
                while True:
                    # Read header
                    header_data = f.read(self.ILDA_HEADER_SIZE)
                    if not header_data or len(header_data) < self.ILDA_HEADER_SIZE:
                        break
                    
                    # Parse header
                    ilda_header, format_code, _, point_count, frame_number, total_frames, _ = struct.unpack(
                        self.ILDA_HEADER_FORMAT,
                        header_data
                    )
                    
                    # Check ILDA header
                    if ilda_header != b"ILDA":
                        raise ValueError("Invalid ILDA header")
                    
                    # Read frame name and company
                    name = f.read(8).decode("ascii")
                    company = f.read(8).decode("ascii")
                    
                    # Read frame data
                    if format_code == self.ILDA_3D_FORMAT:
                        # 3D format: X, Y, Z, Status
                        frame_data = np.zeros((point_count, 4), dtype=np.int16)
                        for i in range(point_count):
                            x, y, z, status = struct.unpack(">hhhB", f.read(7))
                            frame_data[i] = [x, y, z, status]
                    elif format_code == self.ILDA_3D_TRUE_COLOR_FORMAT:
                        # 3D true color format: X, Y, Z, Status, R, G, B
                        frame_data = np.zeros((point_count, 7), dtype=np.int16)
                        for i in range(point_count):
                            x, y, z, status, r, g, b = struct.unpack(">hhhBBBB", f.read(10))
                            frame_data[i] = [x, y, z, status, r, g, b]
                    else:
                        # Skip unsupported format
                        f.seek(point_count * (7 if format_code in [0, 1, 2] else 10), os.SEEK_CUR)
                        continue
                    
                    # Add frame to list
                    self.frames.append(frame_data)
                    self.frame_formats.append(format_code)
                    self.frame_names.append(name)
                    self.frame_companies.append(company)
                    self.frame_point_counts.append(point_count)
                    self.frame_frame_numbers.append(frame_number)
                    self.frame_total_frames.append(total_frames)
            
            logger.info(f"Loaded ILDA file from {file_path}")
        except Exception as e:
            logger.error(f"Error loading ILDA file: {e}")
            raise
    
    def get_vectors(self, frame_index: int = 0) -> np.ndarray:
        """Get vectors from a frame.
        
        Args:
            frame_index: Frame index
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        if frame_index >= len(self.frames):
            raise ValueError(f"Invalid frame index: {frame_index}")
        
        # Get frame data
        frame = self.frames[frame_index]
        
        # Extract X and Y coordinates
        vectors = frame[:, :2]
        
        # Normalize to [-1, 1] range
        vectors_norm = self._normalize_vectors(vectors, (-1, 1))
        
        return vectors_norm
    
    def _normalize_vectors(
        self,
        vectors: np.ndarray,
        normalization_range: Tuple[float, float]
    ) -> np.ndarray:
        """Normalize vectors to a specific range.
        
        Args:
            vectors: Vector coordinates (x, y)
            normalization_range: Range for normalization (min, max)
        
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
        
        # Scale to target range
        range_min, range_max = normalization_range
        range_size = range_max - range_min
        
        vectors_scaled = vectors_norm * range_size + range_min
        
        return vectors_scaled


def convert_svg_to_ilda(
    svg_path: str,
    ilda_path: str,
    color_mode: str = "RGB",
    frame_rate: int = 30,
    points_per_frame: int = 500,
    intensity: float = 0.8
) -> str:
    """Convert an SVG file to ILDA format.
    
    Args:
        svg_path: Path to the SVG file
        ilda_path: Path to save the ILDA file
        color_mode: Color mode (RGB, etc.)
        frame_rate: Frame rate in frames per second
        points_per_frame: Number of points per frame
        intensity: Intensity of the laser
    
    Returns:
        str: Path to the saved ILDA file
    """
    try:
        # Import SVG parser
        from xml.dom import minidom
        
        # Parse SVG file
        svg = minidom.parse(svg_path)
        
        # Extract path data
        paths = svg.getElementsByTagName('path')
        if not paths:
            raise ValueError("No paths found in SVG file")
        
        # Extract path data
        path_data = paths[0].getAttribute('d')
        
        # Parse path data
        vectors = parse_svg_path(path_data)
        
        # Create ILDA file
        ilda_file = ILDAFile()
        
        # Add frame to ILDA file
        ilda_file.add_frame(
            vectors,
            color_mode=color_mode,
            frame_rate=frame_rate,
            points_per_frame=points_per_frame,
            intensity=intensity
        )
        
        # Save ILDA file
        ilda_file.save(ilda_path)
        
        logger.info(f"Converted SVG file {svg_path} to ILDA file {ilda_path}")
        
        return ilda_path
    except Exception as e:
        logger.error(f"Error converting SVG to ILDA: {e}")
        raise


def parse_svg_path(path_data: str) -> np.ndarray:
    """Parse SVG path data.
    
    Args:
        path_data: SVG path data
    
    Returns:
        np.ndarray: Vector coordinates (x, y)
    """
    try:
        # Import SVG path parser
        from svg.path import parse_path
        
        # Parse path data
        path = parse_path(path_data)
        
        # Sample points along the path
        num_points = 1000
        vectors = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            point = path.point(t)
            vectors.append((point.real, point.imag))
        
        return np.array(vectors)
    except ImportError:
        # Fallback to simple parsing
        logger.warning("svg.path not installed. Using simple path parsing.")
        return simple_parse_svg_path(path_data)


def simple_parse_svg_path(path_data: str) -> np.ndarray:
    """Simple SVG path parser.
    
    Args:
        path_data: SVG path data
    
    Returns:
        np.ndarray: Vector coordinates (x, y)
    """
    # Extract coordinates from path data
    # This is a very simple parser that only handles M and L commands
    vectors = []
    
    # Split path data into commands
    commands = path_data.replace(',', ' ').split()
    
    i = 0
    current_x, current_y = 0, 0
    
    while i < len(commands):
        command = commands[i]
        
        if command == 'M':
            # Move to
            current_x = float(commands[i + 1])
            current_y = float(commands[i + 2])
            vectors.append((current_x, current_y))
            i += 3
        elif command == 'L':
            # Line to
            current_x = float(commands[i + 1])
            current_y = float(commands[i + 2])
            vectors.append((current_x, current_y))
            i += 3
        elif command.isalpha():
            # Skip other commands
            i += 1
        else:
            # Implicit line to
            current_x = float(commands[i])
            current_y = float(commands[i + 1])
            vectors.append((current_x, current_y))
            i += 2
    
    return np.array(vectors)