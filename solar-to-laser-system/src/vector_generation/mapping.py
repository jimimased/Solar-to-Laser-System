"""
Vector mapping implementations.

This module provides specific implementations of vector generators.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..common import VectorParameters
from .generator import VectorGenerator

logger = logging.getLogger(__name__)


class DirectMappingGenerator(VectorGenerator):
    """Generator that directly maps features to vector coordinates."""
    
    def __init__(
        self,
        x_feature_idx: int = 0,
        y_feature_idx: int = 1,
        smoothing_factor: float = 0.8,
        interpolation_method: str = "cubic"
    ):
        """Initialize the direct mapping generator.
        
        Args:
            x_feature_idx: Index of the feature to use for x coordinates
            y_feature_idx: Index of the feature to use for y coordinates
            smoothing_factor: Factor for smoothing the vector paths
            interpolation_method: Method for interpolating between points
        """
        self.x_feature_idx = x_feature_idx
        self.y_feature_idx = y_feature_idx
        self.smoothing_factor = smoothing_factor
        self.interpolation_method = interpolation_method
    
    def generate(
        self,
        features: np.ndarray,
        parameters: Optional[VectorParameters] = None
    ) -> np.ndarray:
        """Generate vector coordinates by directly mapping features.
        
        Args:
            features: Input features
            parameters: Vector parameters
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = VectorParameters()
        
        # Extract x and y coordinates from features
        if len(features.shape) == 1:
            # 1D features, use indices
            if self.x_feature_idx >= len(features) or self.y_feature_idx >= len(features):
                raise ValueError(f"Feature indices out of range: {self.x_feature_idx}, {self.y_feature_idx}")
            
            x = np.array([features[self.x_feature_idx]])
            y = np.array([features[self.y_feature_idx]])
        elif len(features.shape) == 2:
            # 2D features, use columns
            if self.x_feature_idx >= features.shape[1] or self.y_feature_idx >= features.shape[1]:
                raise ValueError(f"Feature indices out of range: {self.x_feature_idx}, {self.y_feature_idx}")
            
            x = features[:, self.x_feature_idx]
            y = features[:, self.y_feature_idx]
        else:
            # Higher dimensional features, flatten to 2D
            features_2d = features.reshape(-1, features.shape[-1])
            if self.x_feature_idx >= features_2d.shape[1] or self.y_feature_idx >= features_2d.shape[1]:
                raise ValueError(f"Feature indices out of range: {self.x_feature_idx}, {self.y_feature_idx}")
            
            x = features_2d[:, self.x_feature_idx]
            y = features_2d[:, self.y_feature_idx]
        
        # Create initial vectors
        vectors = np.column_stack((x, y))
        
        # Apply smoothing if needed
        if parameters.smoothing_factor > 0 and len(vectors) > 3:
            vectors = self._smooth_vectors(vectors, parameters.smoothing_factor)
        
        # Apply interpolation if needed
        if parameters.interpolation_method != "none" and len(vectors) > 3:
            vectors = self._interpolate_vectors(vectors, parameters.interpolation_method, parameters.scaling_factor)
        
        # Apply normalization
        vectors = self._normalize_to_range(vectors, parameters.normalization_range)
        
        # Apply path simplification if needed
        if parameters.path_simplification > 0 and len(vectors) > 3:
            vectors = self._simplify_path(vectors, parameters.path_simplification)
        
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
    
    def _interpolate_vectors(
        self,
        vectors: np.ndarray,
        method: str,
        scaling_factor: float
    ) -> np.ndarray:
        """Interpolate between vector points.
        
        Args:
            vectors: Vector coordinates (x, y)
            method: Interpolation method (linear, cubic, etc.)
            scaling_factor: Factor for scaling the number of points
        
        Returns:
            np.ndarray: Interpolated vector coordinates
        """
        # Create parameter values (0 to 1)
        t = np.linspace(0, 1, len(vectors))
        
        # Create interpolation functions
        if method == "linear":
            fx = interpolate.interp1d(t, vectors[:, 0], kind='linear')
            fy = interpolate.interp1d(t, vectors[:, 1], kind='linear')
        elif method == "cubic":
            fx = interpolate.interp1d(t, vectors[:, 0], kind='cubic')
            fy = interpolate.interp1d(t, vectors[:, 1], kind='cubic')
        elif method == "spline":
            # Use spline with smoothing
            fx = interpolate.UnivariateSpline(t, vectors[:, 0], s=0.5)
            fy = interpolate.UnivariateSpline(t, vectors[:, 1], s=0.5)
        else:
            # No interpolation
            return vectors
        
        # Create new parameter values with more points
        num_points = int(len(vectors) * scaling_factor)
        t_new = np.linspace(0, 1, num_points)
        
        # Interpolate
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        # Create new vectors
        vectors_new = np.column_stack((x_new, y_new))
        
        return vectors_new
    
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
    
    def _simplify_path(self, vectors: np.ndarray, tolerance: float) -> np.ndarray:
        """Simplify vector path by removing redundant points.
        
        Args:
            vectors: Vector coordinates (x, y)
            tolerance: Tolerance for simplification
        
        Returns:
            np.ndarray: Simplified vector coordinates
        """
        # Implement Douglas-Peucker algorithm
        def douglas_peucker(points, epsilon):
            # Find the point with the maximum distance
            dmax = 0
            index = 0
            end = len(points) - 1
            
            for i in range(1, end):
                d = point_line_distance(points[i], points[0], points[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            # If max distance is greater than epsilon, recursively simplify
            if dmax > epsilon:
                # Recursive call
                rec_results1 = douglas_peucker(points[:index+1], epsilon)
                rec_results2 = douglas_peucker(points[index:], epsilon)
                
                # Build the result list
                result = np.vstack((rec_results1[:-1], rec_results2))
            else:
                result = np.vstack((points[0], points[end]))
            
            return result
        
        def point_line_distance(point, line_start, line_end):
            if np.array_equal(line_start, line_end):
                return np.linalg.norm(point - line_start)
            
            # Calculate the distance from point to line
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len
            point_vec_scaled = point_vec / line_len
            
            # Get the scalar projection of point_vec onto line_unitvec
            t = np.dot(line_unitvec, point_vec_scaled)
            
            # Get the closest point on the line to the point
            if t < 0:
                closest = line_start
            elif t > 1:
                closest = line_end
            else:
                closest = line_start + t * line_vec
            
            # Return the distance
            return np.linalg.norm(point - closest)
        
        # Apply Douglas-Peucker algorithm
        simplified = douglas_peucker(vectors, tolerance)
        
        return simplified


class DimensionalityReductionGenerator(VectorGenerator):
    """Generator that uses dimensionality reduction to generate vector coordinates."""
    
    def __init__(
        self,
        method: str = "pca",
        n_components: int = 2,
        smoothing_factor: float = 0.8,
        interpolation_method: str = "cubic"
    ):
        """Initialize the dimensionality reduction generator.
        
        Args:
            method: Dimensionality reduction method (pca, tsne)
            n_components: Number of components to reduce to
            smoothing_factor: Factor for smoothing the vector paths
            interpolation_method: Method for interpolating between points
        """
        self.method = method
        self.n_components = n_components
        self.smoothing_factor = smoothing_factor
        self.interpolation_method = interpolation_method
    
    def generate(
        self,
        features: np.ndarray,
        parameters: Optional[VectorParameters] = None
    ) -> np.ndarray:
        """Generate vector coordinates using dimensionality reduction.
        
        Args:
            features: Input features
            parameters: Vector parameters
        
        Returns:
            np.ndarray: Vector coordinates (x, y)
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = VectorParameters()
        
        # Reshape features if needed
        if len(features.shape) > 2:
            features = features.reshape(-1, features.shape[-1])
        
        # Apply dimensionality reduction
        if self.method == "pca":
            reducer = PCA(n_components=self.n_components)
        elif self.method == "tsne":
            reducer = TSNE(n_components=self.n_components, perplexity=min(30, max(5, len(features) // 5)))
        else:
            raise ValueError(f"Invalid dimensionality reduction method: {self.method}")
        
        # Reduce dimensionality
        vectors = reducer.fit_transform(features)
        
        # Apply smoothing if needed
        if parameters.smoothing_factor > 0 and len(vectors) > 3:
            vectors = self._smooth_vectors(vectors, parameters.smoothing_factor)
        
        # Apply interpolation if needed
        if parameters.interpolation_method != "none" and len(vectors) > 3:
            vectors = self._interpolate_vectors(vectors, parameters.interpolation_method, parameters.scaling_factor)
        
        # Apply normalization
        vectors = self._normalize_to_range(vectors, parameters.normalization_range)
        
        # Apply path simplification if needed
        if parameters.path_simplification > 0 and len(vectors) > 3:
            vectors = self._simplify_path(vectors, parameters.path_simplification)
        
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
    
    def _interpolate_vectors(
        self,
        vectors: np.ndarray,
        method: str,
        scaling_factor: float
    ) -> np.ndarray:
        """Interpolate between vector points.
        
        Args:
            vectors: Vector coordinates (x, y)
            method: Interpolation method (linear, cubic, etc.)
            scaling_factor: Factor for scaling the number of points
        
        Returns:
            np.ndarray: Interpolated vector coordinates
        """
        # Create parameter values (0 to 1)
        t = np.linspace(0, 1, len(vectors))
        
        # Create interpolation functions
        if method == "linear":
            fx = interpolate.interp1d(t, vectors[:, 0], kind='linear')
            fy = interpolate.interp1d(t, vectors[:, 1], kind='linear')
        elif method == "cubic":
            fx = interpolate.interp1d(t, vectors[:, 0], kind='cubic')
            fy = interpolate.interp1d(t, vectors[:, 1], kind='cubic')
        elif method == "spline":
            # Use spline with smoothing
            fx = interpolate.UnivariateSpline(t, vectors[:, 0], s=0.5)
            fy = interpolate.UnivariateSpline(t, vectors[:, 1], s=0.5)
        else:
            # No interpolation
            return vectors
        
        # Create new parameter values with more points
        num_points = int(len(vectors) * scaling_factor)
        t_new = np.linspace(0, 1, num_points)
        
        # Interpolate
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        # Create new vectors
        vectors_new = np.column_stack((x_new, y_new))
        
        return vectors_new
    
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
    
    def _simplify_path(self, vectors: np.ndarray, tolerance: float) -> np.ndarray:
        """Simplify vector path by removing redundant points.
        
        Args:
            vectors: Vector coordinates (x, y)
            tolerance: Tolerance for simplification
        
        Returns:
            np.ndarray: Simplified vector coordinates
        """
        # Implement Douglas-Peucker algorithm
        def douglas_peucker(points, epsilon):
            # Find the point with the maximum distance
            dmax = 0
            index = 0
            end = len(points) - 1
            
            for i in range(1, end):
                d = point_line_distance(points[i], points[0], points[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            # If max distance is greater than epsilon, recursively simplify
            if dmax > epsilon:
                # Recursive call
                rec_results1 = douglas_peucker(points[:index+1], epsilon)
                rec_results2 = douglas_peucker(points[index:], epsilon)
                
                # Build the result list
                result = np.vstack((rec_results1[:-1], rec_results2))
            else:
                result = np.vstack((points[0], points[end]))
            
            return result
        
        def point_line_distance(point, line_start, line_end):
            if np.array_equal(line_start, line_end):
                return np.linalg.norm(point - line_start)
            
            # Calculate the distance from point to line
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len
            point_vec_scaled = point_vec / line_len
            
            # Get the scalar projection of point_vec onto line_unitvec
            t = np.dot(line_unitvec, point_vec_scaled)
            
            # Get the closest point on the line to the point
            if t < 0:
                closest = line_start
            elif t > 1:
                closest = line_end
            else:
                closest = line_start + t * line_vec
            
            # Return the distance
            return np.linalg.norm(point - closest)
        
        # Apply Douglas-Peucker algorithm
        simplified = douglas_peucker(vectors, tolerance)
        
        return simplified