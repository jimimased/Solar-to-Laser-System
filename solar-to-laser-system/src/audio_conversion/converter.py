"""
Audio converter implementation.

This module provides classes for converting solar data to audio.
"""

import os
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

import numpy as np
import librosa
import soundfile as sf

from ..common import SolarData, AudioParameters

logger = logging.getLogger(__name__)


class AudioConverter:
    """Base class for audio converters."""
    
    def convert(
        self,
        solar_data: Union[SolarData, List[SolarData]],
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert solar data to audio.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        raise NotImplementedError("Subclasses must implement convert()")
    
    def save(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        file_path: str,
        format: str = "wav"
    ) -> str:
        """Save audio data to a file.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            file_path: Path to save the file
            format: Audio format
        
        Returns:
            str: Path to the saved file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save audio file
            sf.write(file_path, audio_data, sample_rate, format=format)
            
            logger.info(f"Saved audio file to {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            raise


class DirectMappingConverter(AudioConverter):
    """Converter that directly maps solar data to audio parameters."""
    
    def __init__(
        self,
        voltage_range: Tuple[float, float] = (0.0, 48.0),
        current_range: Tuple[float, float] = (0.0, 10.0),
        frequency_range: Tuple[float, float] = (220.0, 880.0),
        amplitude_range: Tuple[float, float] = (0.0, 0.9)
    ):
        """Initialize the direct mapping converter.
        
        Args:
            voltage_range: Range of voltage values (min, max)
            current_range: Range of current values (min, max)
            frequency_range: Range of frequency values (min, max)
            amplitude_range: Range of amplitude values (min, max)
        """
        self.voltage_range = voltage_range
        self.current_range = current_range
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range
    
    def _map_voltage_to_frequency(self, voltage: float) -> float:
        """Map voltage to frequency.
        
        Args:
            voltage: Voltage value
        
        Returns:
            float: Frequency value
        """
        # Normalize voltage to [0, 1]
        normalized = (voltage - self.voltage_range[0]) / (self.voltage_range[1] - self.voltage_range[0])
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Map to frequency range
        frequency = self.frequency_range[0] + normalized * (self.frequency_range[1] - self.frequency_range[0])
        
        return frequency
    
    def _map_current_to_amplitude(self, current: float) -> float:
        """Map current to amplitude.
        
        Args:
            current: Current value
        
        Returns:
            float: Amplitude value
        """
        # Normalize current to [0, 1]
        normalized = (current - self.current_range[0]) / (self.current_range[1] - self.current_range[0])
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Map to amplitude range
        amplitude = self.amplitude_range[0] + normalized * (self.amplitude_range[1] - self.amplitude_range[0])
        
        return amplitude
    
    def _map_power_to_harmonics(self, power: float, num_harmonics: int = 5) -> List[float]:
        """Map power to harmonic amplitudes.
        
        Args:
            power: Power value
            num_harmonics: Number of harmonics
        
        Returns:
            List[float]: Harmonic amplitudes
        """
        # Normalize power to [0, 1]
        power_range = (self.voltage_range[0] * self.current_range[0], self.voltage_range[1] * self.current_range[1])
        normalized = (power - power_range[0]) / (power_range[1] - power_range[0])
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Generate harmonic amplitudes
        # Higher power = more harmonics
        harmonics = []
        for i in range(1, num_harmonics + 1):
            # Amplitude decreases with harmonic number
            # Higher power = slower decrease
            amplitude = normalized * (1.0 / i) ** (1.0 - normalized * 0.5)
            harmonics.append(amplitude)
        
        # Normalize harmonics so they sum to 1
        total = sum(harmonics)
        if total > 0:
            harmonics = [h / total for h in harmonics]
        
        return harmonics
    
    def _generate_sine_wave(
        self,
        frequency: float,
        amplitude: float,
        duration: float,
        sample_rate: int,
        harmonics: Optional[List[float]] = None
    ) -> np.ndarray:
        """Generate a sine wave with optional harmonics.
        
        Args:
            frequency: Frequency in Hz
            amplitude: Amplitude
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            harmonics: Harmonic amplitudes
        
        Returns:
            np.ndarray: Audio data
        """
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Generate sine wave
        audio = np.zeros_like(t)
        
        if harmonics:
            # Add harmonics
            for i, harmonic_amp in enumerate(harmonics):
                harmonic_freq = frequency * (i + 1)
                audio += harmonic_amp * amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        else:
            # Just the fundamental
            audio = amplitude * np.sin(2 * np.pi * frequency * t)
        
        return audio
    
    def convert_single(
        self,
        solar_data: SolarData,
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert a single solar data point to audio.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = AudioParameters(
                duration=1.0,
                frequency_mapping=self._map_voltage_to_frequency,
                amplitude_mapping=self._map_current_to_amplitude,
                timbre_mapping=lambda power: self._map_power_to_harmonics(power),
                temporal_mapping=lambda values: values,
                sample_rate=44100
            )
        
        # Map solar data to audio parameters
        frequency = parameters.frequency_mapping(solar_data.voltage)
        amplitude = parameters.amplitude_mapping(solar_data.current)
        harmonics = parameters.timbre_mapping(solar_data.power)
        
        # Generate audio
        audio = self._generate_sine_wave(
            frequency=frequency,
            amplitude=amplitude,
            duration=parameters.duration,
            sample_rate=parameters.sample_rate,
            harmonics=harmonics
        )
        
        return audio, parameters.sample_rate
    
    def convert(
        self,
        solar_data: Union[SolarData, List[SolarData]],
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert solar data to audio.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Use default parameters if not provided
        if parameters is None:
            parameters = AudioParameters(
                duration=1.0,
                frequency_mapping=self._map_voltage_to_frequency,
                amplitude_mapping=self._map_current_to_amplitude,
                timbre_mapping=lambda power: self._map_power_to_harmonics(power),
                temporal_mapping=lambda values: values,
                sample_rate=44100
            )
        
        # Handle single data point
        if isinstance(solar_data, SolarData):
            return self.convert_single(solar_data, parameters)
        
        # Handle multiple data points
        if not solar_data:
            # Empty list, return silence
            return np.zeros(int(parameters.sample_rate * parameters.duration)), parameters.sample_rate
        
        # Convert each data point
        audio_segments = []
        for data_point in solar_data:
            audio, _ = self.convert_single(data_point, parameters)
            audio_segments.append(audio)
        
        # Concatenate audio segments
        audio = np.concatenate(audio_segments)
        
        return audio, parameters.sample_rate
