"""
Vector generator implementation.

This module provides classes for generating vector graphics from audio features.
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..common import VectorParameters

logger = logging.getLogger(__name__)


class VectorGenerator:
    """Base class for vector generators."""
    
    def generate(
        self,
        features: np.ndarray,
        parameters: Optional[VectorParameters] = None
    ) -> np.ndarray:
        """Generate vector coordinates from features.
        
        Args:
            features: Input features
            parameters: Vector parameters
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def save_svg(
        self,
        vectors: np.ndarray,
        file_path: str,
        width: int = 800,
        height: int = 600,
        stroke_width: float = 1.0,
        stroke_color: str = "#000000"
    ) -> str:
        """Save vectors to an SVG file.
        
        Args:
            vectors: Vector coordinates (x, y)
            file_path: Path to save the file
            width: SVG width
            height: SVG height
            stroke_width: Stroke width
            stroke_color: Stroke color
        
        Returns:
            str: Path to the saved file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Normalize vectors to [0, 1] range
            vectors_norm = self._normalize_vectors(vectors)
            
            # Scale to SVG dimensions
            vectors_scaled = np.zeros_like(vectors_norm)
            vectors_scaled[:, 0] = vectors_norm[:, 0] * width
            vectors_scaled[:, 1] = vectors_norm[:, 1] * height
            
            # Create SVG content
            svg_content = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
            svg_content += f'  <path d="M'
            
            # Add path data
            for i, (x, y) in enumerate(vectors_scaled):
                if i == 0:
                    svg_content += f" {x:.2f},{y:.2f}"
                else:
                    svg_content += f" L{x:.2f},{y:.2f}"
            
            svg_content += f'" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}" />\n'
            svg_content += '</svg>'
            
            # Write to file
            with open(file_path, 'w') as f:
                f.write(svg_content)
            
            logger.info(f"Saved SVG file to {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error saving SVG file: {e}")
            raise
    
    def save_preview(
        self,
        vectors: np.ndarray,
        file_path: str,
        width: int = 800,
        height: int = 600,
        dpi: int = 100,
        line_width: float = 1.0,
        line_color: str = "black",
        bg_color: str = "white"
    ) -> str:
        """Save a preview image of the vectors.
        
        Args:
            vectors: Vector coordinates (x, y)
            file_path: Path to save the file
            width: Image width
            height: Image height
            dpi: Image DPI
            line_width: Line width
            line_color: Line color
            bg_color: Background color
        
        Returns:
            str: Path to the saved file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Normalize vectors to [0, 1] range
            vectors_norm = self._normalize_vectors(vectors)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            fig.patch.set_facecolor(bg_color)
            
            # Plot vectors
            ax.plot(vectors_norm[:, 0], vectors_norm[:, 1], color=line_color, linewidth=line_width)
            
            # Save figure
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            logger.info(f"Saved preview image to {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error saving preview image: {e}")
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
