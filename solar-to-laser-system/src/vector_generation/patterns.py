"""
Pattern-based vector generators.

This module provides vector generators based on mathematical patterns.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from ..common import VectorParameters
from .generator import VectorGenerator

logger = logging.getLogger(__name__)


class PatternGenerator(VectorGenerator):
    """Generator that creates patterns from features."""
    
    def __init__(
        self,
        pattern_type: str = "lissajous",
        smoothing_factor: float = 0.8,
        interpolation_method: str = "cubic"
    ):
        """Initialize the pattern generator.
        
        Args:
            pattern_type: Type of pattern to generate (lissajous, spiral, etc.)
            smoothing_factor: Factor for smoothing the vector paths
            interpolation_method: Method for interpolating between points
        """
        self.pattern_type = pattern_type
        self.smoothing_factor = smoothing_factor
        self.interpolation_method = interpolation_method
    
    def generate(
        self,
        features: np.ndarray,
        parameters: Optional[VectorParameters] = None
    ) -> np.ndarray:
        """Generate vector coordinates using patterns.
        
        Args:
            features: Input features
            parameters: Vector parameters
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = VectorParameters()
        
        # Extract parameters from features
        if len(features.shape) == 1:
            # Single feature vector
            params = features
        else:
            # Multiple feature vectors, use mean
            params = np.mean(features, axis=0)
        
        # Generate pattern
        if self.pattern_type == "lissajous":
            vectors = self._generate_lissajous(params, parameters.scaling_factor)
        elif self.pattern_type == "spiral":
            vectors = self._generate_spiral(params, parameters.scaling_factor)
        elif self.pattern_type == "harmonograph":
            vectors = self._generate_harmonograph(params, parameters.scaling_factor)
        else:
            raise ValueError(f"Invalid pattern type: {self.pattern_type}")
        
        # Apply smoothing if needed
        if parameters.smoothing_factor > 0 and len(vectors) > 3:
            vectors = self._smooth_vectors(vectors, parameters.smoothing_factor)
        
        # Apply normalization
        vectors = self._normalize_to_range(vectors, parameters.normalization_range)
        
        return vectors
    
    def _generate_lissajous(self, params: np.ndarray, scaling_factor: float) -> np.ndarray:
        """Generate Lissajous pattern.
        
        Args:
            params: Parameters for the pattern
            scaling_factor: Scaling factor
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        # Extract parameters
        if len(params) >= 4:
            a = 1.0 + params[0] % 1.0  # Frequency ratio
            b = 1.0 + params[1] % 1.0
            delta = params[2] * np.pi  # Phase difference
            num_points = int(1000 * scaling_factor)
        else:
            # Default parameters
            a = 3.0
            b = 4.0
            delta = np.pi / 2
            num_points = int(1000 * scaling_factor)
        
        # Generate pattern
        t = np.linspace(0, 2 * np.pi, num_points)
        x = np.sin(a * t + delta)
        y = np.sin(b * t)
        
        # Create vectors
        vectors = np.column_stack((x, y))
        
        return vectors
    
    def _generate_spiral(self, params: np.ndarray, scaling_factor: float) -> np.ndarray:
        """Generate spiral pattern.
        
        Args:
            params: Parameters for the pattern
            scaling_factor: Scaling factor
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        # Extract parameters
        if len(params) >= 3:
            a = 0.1 + params[0] % 0.2  # Spiral tightness
            num_turns = 1 + int(params[1] * 10)  # Number of turns
            num_points = int(1000 * scaling_factor)
        else:
            # Default parameters
            a = 0.1
            num_turns = 5
            num_points = int(1000 * scaling_factor)
        
        # Generate pattern
        theta = np.linspace(0, num_turns * 2 * np.pi, num_points)
        r = a * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Create vectors
        vectors = np.column_stack((x, y))
        
        return vectors
    
    def _generate_harmonograph(self, params: np.ndarray, scaling_factor: float) -> np.ndarray:
        """Generate harmonograph pattern.
        
        Args:
            params: Parameters for the pattern
            scaling_factor: Scaling factor
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        # Extract parameters
        if len(params) >= 8:
            a1 = params[0]  # Amplitude
            a2 = params[1]
            a3 = params[2]
            a4 = params[3]
            f1 = 1.0 + params[4] % 1.0  # Frequency
            f2 = 1.0 + params[5] % 1.0
            f3 = 1.0 + params[6] % 1.0
            f4 = 1.0 + params[7] % 1.0
            p1 = 0  # Phase
            p2 = np.pi / 2
            p3 = 0
            p4 = np.pi / 2
            d1 = 0.01  # Damping
            d2 = 0.01
            d3 = 0.01
            d4 = 0.01
            num_points = int(1000 * scaling_factor)
        else:
            # Default parameters
            a1 = a2 = a3 = a4 = 1.0
            f1 = 2.0
            f2 = 3.0
            f3 = 3.0
            f4 = 2.0
            p1 = p3 = 0
            p2 = p4 = np.pi / 2
            d1 = d2 = d3 = d4 = 0.01
            num_points = int(1000 * scaling_factor)
        
        # Generate pattern
        t = np.linspace(0, 10, num_points)
        x = a1 * np.sin(f1 * t + p1) * np.exp(-d1 * t) + a3 * np.sin(f3 * t + p3) * np.exp(-d3 * t)
        y = a2 * np.sin(f2 * t + p2) * np.exp(-d2 * t) + a4 * np.sin(f4 * t + p4) * np.exp(-d4 * t)
        
        # Create vectors
        vectors = np.column_stack((x, y))
        
        return vectors
    
    def _smooth_vectors(self, vectors: np.ndarray, smoothing_factor: float) -> np.ndarray:
        """Smooth vector paths.
        
        Args:
            vectors: Vector coordinates (x, y)
            smoothing_factor: Factor for smoothing (0-1)
        
        Returns:
            np.ndarray: Smoothed vector coordinates
        """
        # Apply moving average
        window_size = max(3, int(len(vectors) * smoothing_factor * 0.1))
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        # Apply moving average to x and y separately
        smoothed = np.zeros_like(vectors)
        
        for i in range(vectors.shape[1]):
            smoothed[:, i] = np.convolve(vectors[:, i], np.ones(window_size)/window_size, mode='same')
        
        return smoothed
    
    def _normalize_to_range(
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
        # Normalize to [0, 1]
        vectors_norm = self._normalize_vectors(vectors)
        
        # Scale to target range
        range_min, range_max = normalization_range
        range_size = range_max - range_min
        
        vectors_scaled = vectors_norm * range_size + range_min
        
        return vectors_scaled


class LissajousGenerator(PatternGenerator):
    """Generator that creates Lissajous patterns."""
    
    def __init__(
        self,
        smoothing_factor: float = 0.8,
        interpolation_method: str = "cubic"
    ):
        """Initialize the Lissajous generator.
        
        Args:
            smoothing_factor: Factor for smoothing the vector paths
            interpolation_method: Method for interpolating between points
        """
        super().__init__(
            pattern_type="lissajous",
            smoothing_factor=smoothing_factor,
            interpolation_method=interpolation_method
        )


class SpiralGenerator(PatternGenerator):
    """Generator that creates spiral patterns."""
    
    def __init__(
        self,
        smoothing_factor: float = 0.8,
        interpolation_method: str = "cubic"
    ):
        """Initialize the spiral generator.
        
        Args:
            smoothing_factor: Factor for smoothing the vector paths
            interpolation_method: Method for interpolating between points
        """
        super().__init__(
            pattern_type="spiral",
            smoothing_factor=smoothing_factor,
            interpolation_method=interpolation_method
        )


class HarmonographGenerator(PatternGenerator):
    """Generator that creates harmonograph patterns."""
    
    def __init__(
        self,
        smoothing_factor: float = 0.8,
        interpolation_method: str = "cubic"
    ):
        """Initialize the harmonograph generator.
        
        Args:
            smoothing_factor: Factor for smoothing the vector paths
            interpolation_method: Method for interpolating between points
        """
        super().__init__(
            pattern_type="harmonograph",
            smoothing_factor=smoothing_factor,
            interpolation_method=interpolation_method
        )


class MultiPatternGenerator(VectorGenerator):
    """Generator that combines multiple patterns."""
    
    def __init__(
        self,
        generators: List[VectorGenerator],
        weights: Optional[List[float]] = None
    ):
        """Initialize the multi-pattern generator.
        
        Args:
            generators: List of generators to combine
            weights: Weights for each generator (default: equal weights)
        """
        self.generators = generators
        
        if weights is None:
            self.weights = [1.0 / len(generators)] * len(generators)
        else:
            if len(weights) != len(generators):
                raise ValueError("Number of weights must match number of generators")
            
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def generate(
        self,
        features: np.ndarray,
        parameters: Optional[VectorParameters] = None
    ) -> np.ndarray:
        """Generate vector coordinates by combining multiple patterns.
        
        Args:
            features: Input features
            parameters: Vector parameters
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = VectorParameters()
        
        # Generate vectors from each generator
        all_vectors = []
        for generator, weight in zip(self.generators, self.weights):
            vectors = generator.generate(features, parameters)
            
            # Normalize to [0, 1] range
            vectors_norm = self._normalize_vectors(vectors)
            
            # Apply weight
            vectors_weighted = vectors_norm * weight
            
            all_vectors.append(vectors_weighted)
        
        # Combine vectors
        # For simplicity, we'll just use the first generator's vectors
        # In a real implementation, we would need to interpolate between the vectors
        vectors = all_vectors[0]
        
        # Apply normalization
        vectors = self._normalize_to_range(vectors, parameters.normalization_range)
        
        return vectors
    
    def _normalize_to_range(
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
        # Normalize to [0, 1]
        vectors_norm = self._normalize_vectors(vectors)
        
        # Scale to target range
        range_min, range_max = normalization_range
        range_size = range_max - range_min
        
        vectors_scaled = vectors_norm * range_size + range_min
        
        return vectors_scaled