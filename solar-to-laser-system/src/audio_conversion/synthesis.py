"""
Audio synthesis methods for the audio conversion module.

This module provides additional audio synthesis methods for converting solar data to audio.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

from ..common import SolarData, AudioParameters
from .converter import AudioConverter

logger = logging.getLogger(__name__)


class FMSynthesisConverter(AudioConverter):
    """Converter that uses FM synthesis to convert solar data to audio."""
    
    def __init__(
        self,
        voltage_range: Tuple[float, float] = (0.0, 48.0),
        current_range: Tuple[float, float] = (0.0, 10.0),
        carrier_range: Tuple[float, float] = (220.0, 880.0),
        modulator_range: Tuple[float, float] = (50.0, 500.0),
        index_range: Tuple[float, float] = (0.5, 10.0),
        amplitude_range: Tuple[float, float] = (0.0, 0.9)
    ):
        """Initialize the FM synthesis converter.
        
        Args:
            voltage_range: Range of voltage values (min, max)
            current_range: Range of current values (min, max)
            carrier_range: Range of carrier frequency values (min, max)
            modulator_range: Range of modulator frequency values (min, max)
            index_range: Range of modulation index values (min, max)
            amplitude_range: Range of amplitude values (min, max)
        """
        self.voltage_range = voltage_range
        self.current_range = current_range
        self.carrier_range = carrier_range
        self.modulator_range = modulator_range
        self.index_range = index_range
        self.amplitude_range = amplitude_range
    
    def _map_voltage_to_carrier(self, voltage: float) -> float:
        """Map voltage to carrier frequency.
        
        Args:
            voltage: Voltage value
        
        Returns:
            float: Carrier frequency value
        """
        # Normalize voltage to [0, 1]
        normalized = (voltage - self.voltage_range[0]) / (self.voltage_range[1] - self.voltage_range[0])
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Map to carrier frequency range
        carrier = self.carrier_range[0] + normalized * (self.carrier_range[1] - self.carrier_range[0])
        
        return carrier
    
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
    
    def _map_power_to_modulation(self, power: float) -> Tuple[float, float]:
        """Map power to modulator frequency and index.
        
        Args:
            power: Power value
        
        Returns:
            Tuple[float, float]: Modulator frequency and index
        """
        # Normalize power to [0, 1]
        power_range = (self.voltage_range[0] * self.current_range[0], self.voltage_range[1] * self.current_range[1])
        normalized = (power - power_range[0]) / (power_range[1] - power_range[0])
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Map to modulator frequency range
        modulator = self.modulator_range[0] + normalized * (self.modulator_range[1] - self.modulator_range[0])
        
        # Map to index range
        index = self.index_range[0] + normalized * (self.index_range[1] - self.index_range[0])
        
        return modulator, index
    
    def _generate_fm_wave(
        self,
        carrier_freq: float,
        modulator_freq: float,
        index: float,
        amplitude: float,
        duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """Generate an FM synthesis wave.
        
        Args:
            carrier_freq: Carrier frequency in Hz
            modulator_freq: Modulator frequency in Hz
            index: Modulation index
            amplitude: Amplitude
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            np.ndarray: Audio data
        """
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Generate FM wave
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t + index * modulator)
        audio = amplitude * carrier
        
        return audio
    
    def convert_single(
        self,
        solar_data: SolarData,
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert a single solar data point to audio using FM synthesis.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Use default parameters if not provided
        if parameters is None:
            sample_rate = 44100
            duration = 1.0
        else:
            sample_rate = parameters.sample_rate
            duration = parameters.duration
        
        # Map solar data to FM parameters
        carrier_freq = self._map_voltage_to_carrier(solar_data.voltage)
        amplitude = self._map_current_to_amplitude(solar_data.current)
        modulator_freq, index = self._map_power_to_modulation(solar_data.power)
        
        # Generate audio
        audio = self._generate_fm_wave(
            carrier_freq=carrier_freq,
            modulator_freq=modulator_freq,
            index=index,
            amplitude=amplitude,
            duration=duration,
            sample_rate=sample_rate
        )
        
        return audio, sample_rate
    
    def convert(
        self,
        solar_data: Union[SolarData, List[SolarData]],
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert solar data to audio using FM synthesis.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Use default parameters if not provided
        if parameters is None:
            sample_rate = 44100
            duration = 1.0
        else:
            sample_rate = parameters.sample_rate
            duration = parameters.duration
        
        # Handle single data point
        if isinstance(solar_data, SolarData):
            return self.convert_single(solar_data, parameters)
        
        # Handle multiple data points
        if not solar_data:
            # Empty list, return silence
            return np.zeros(int(sample_rate * duration)), sample_rate
        
        # Convert each data point
        audio_segments = []
        for data_point in solar_data:
            audio, _ = self.convert_single(data_point, parameters)
            audio_segments.append(audio)
        
        # Concatenate audio segments
        audio = np.concatenate(audio_segments)
        
        return audio, sample_rate


class GranularSynthesisConverter(AudioConverter):
    """Converter that uses granular synthesis to convert solar data to audio."""
    
    def __init__(
        self,
        voltage_range: Tuple[float, float] = (0.0, 48.0),
        current_range: Tuple[float, float] = (0.0, 10.0),
        grain_size_range: Tuple[float, float] = (0.01, 0.1),
        density_range: Tuple[float, float] = (1.0, 20.0),
        pitch_range: Tuple[float, float] = (0.5, 2.0),
        amplitude_range: Tuple[float, float] = (0.0, 0.9),
        source_file: Optional[str] = None
    ):
        """Initialize the granular synthesis converter.
        
        Args:
            voltage_range: Range of voltage values (min, max)
            current_range: Range of current values (min, max)
            grain_size_range: Range of grain size values in seconds (min, max)
            density_range: Range of grain density values in grains per second (min, max)
            pitch_range: Range of pitch shift values (min, max)
            amplitude_range: Range of amplitude values (min, max)
            source_file: Path to source audio file for granulation
        """
        self.voltage_range = voltage_range
        self.current_range = current_range
        self.grain_size_range = grain_size_range
        self.density_range = density_range
        self.pitch_range = pitch_range
        self.amplitude_range = amplitude_range
        
        # Load source file or create default source
        self.source_audio = None
        self.source_sr = 44100
        
        if source_file and os.path.exists(source_file):
            self._load_source(source_file)
        else:
            self._create_default_source()
    
    def _load_source(self, source_file: str):
        """Load source audio file.
        
        Args:
            source_file: Path to source audio file
        """
        try:
            self.source_audio, self.source_sr = librosa.load(source_file, sr=None)
            logger.info(f"Loaded source file: {source_file}")
        except Exception as e:
            logger.error(f"Error loading source file: {e}")
            self._create_default_source()
    
    def _create_default_source(self):
        """Create default source audio."""
        # Create a 5-second noise source
        duration = 5.0
        self.source_sr = 44100
        self.source_audio = np.random.normal(0, 0.1, int(self.source_sr * duration))
        logger.info("Created default noise source")
    
    def _map_voltage_to_pitch(self, voltage: float) -> float:
        """Map voltage to pitch shift.
        
        Args:
            voltage: Voltage value
        
        Returns:
            float: Pitch shift value
        """
        # Normalize voltage to [0, 1]
        normalized = (voltage - self.voltage_range[0]) / (self.voltage_range[1] - self.voltage_range[0])
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Map to pitch range
        pitch = self.pitch_range[0] + normalized * (self.pitch_range[1] - self.pitch_range[0])
        
        return pitch
    
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
    
    def _map_power_to_grain_params(self, power: float) -> Tuple[float, float]:
        """Map power to grain size and density.
        
        Args:
            power: Power value
        
        Returns:
            Tuple[float, float]: Grain size and density
        """
        # Normalize power to [0, 1]
        power_range = (self.voltage_range[0] * self.current_range[0], self.voltage_range[1] * self.current_range[1])
        normalized = (power - power_range[0]) / (power_range[1] - power_range[0])
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        # Map to grain size range (inverse relationship - higher power = smaller grains)
        grain_size = self.grain_size_range[1] - normalized * (self.grain_size_range[1] - self.grain_size_range[0])
        
        # Map to density range (direct relationship - higher power = higher density)
        density = self.density_range[0] + normalized * (self.density_range[1] - self.density_range[0])
        
        return grain_size, density
    
    def _generate_grain(
        self,
        source: np.ndarray,
        source_sr: int,
        position: float,
        grain_size: float,
        pitch_shift: float
    ) -> np.ndarray:
        """Generate a single grain.
        
        Args:
            source: Source audio
            source_sr: Source sample rate
            position: Position in source (0-1)
            grain_size: Grain size in seconds
            pitch_shift: Pitch shift factor
        
        Returns:
            np.ndarray: Grain audio
        """
        # Calculate grain parameters
        grain_samples = int(grain_size * source_sr)
        position_samples = int(position * len(source))
        
        # Ensure position is valid
        position_samples = max(0, min(position_samples, len(source) - grain_samples))
        
        # Extract grain from source
        if position_samples + grain_samples <= len(source):
            grain = source[position_samples:position_samples + grain_samples].copy()
        else:
            # Handle edge case
            grain = np.zeros(grain_samples)
            available = len(source) - position_samples
            grain[:available] = source[position_samples:].copy()
        
        # Apply envelope (Hann window)
        window = np.hanning(len(grain))
        grain *= window
        
        # Apply pitch shift if needed
        if pitch_shift != 1.0:
            grain = librosa.effects.pitch_shift(
                grain,
                sr=source_sr,
                n_steps=12 * np.log2(pitch_shift)  # Convert to semitones
            )
        
        return grain
    
    def _generate_granular_synthesis(
        self,
        grain_size: float,
        density: float,
        pitch_shift: float,
        amplitude: float,
        duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """Generate granular synthesis audio.
        
        Args:
            grain_size: Grain size in seconds
            density: Grain density in grains per second
            pitch_shift: Pitch shift factor
            amplitude: Amplitude
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            np.ndarray: Audio data
        """
        # Ensure source audio is available
        if self.source_audio is None:
            self._create_default_source()
        
        # Create output buffer
        output = np.zeros(int(duration * sample_rate))
        
        # Calculate number of grains
        num_grains = int(duration * density)
        
        # Generate grains
        for i in range(num_grains):
            # Calculate grain position in output
            grain_time = np.random.uniform(0, duration)
            grain_position = int(grain_time * sample_rate)
            
            # Calculate source position (random)
            source_position = np.random.uniform(0, 1)
            
            # Generate grain
            grain = self._generate_grain(
                source=self.source_audio,
                source_sr=self.source_sr,
                position=source_position,
                grain_size=grain_size,
                pitch_shift=pitch_shift
            )
            
            # Resample grain if needed
            if self.source_sr != sample_rate:
                grain = librosa.resample(grain, orig_sr=self.source_sr, target_sr=sample_rate)
            
            # Add grain to output
            grain_end = min(grain_position + len(grain), len(output))
            output_segment = output[grain_position:grain_end]
            grain_segment = grain[:len(output_segment)]
            output_segment += grain_segment
        
        # Normalize and apply amplitude
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * amplitude
        
        return output
    
    def convert_single(
        self,
        solar_data: SolarData,
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert a single solar data point to audio using granular synthesis.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Use default parameters if not provided
        if parameters is None:
            sample_rate = 44100
            duration = 1.0
        else:
            sample_rate = parameters.sample_rate
            duration = parameters.duration
        
        # Map solar data to granular synthesis parameters
        pitch_shift = self._map_voltage_to_pitch(solar_data.voltage)
        amplitude = self._map_current_to_amplitude(solar_data.current)
        grain_size, density = self._map_power_to_grain_params(solar_data.power)
        
        # Generate audio
        audio = self._generate_granular_synthesis(
            grain_size=grain_size,
            density=density,
            pitch_shift=pitch_shift,
            amplitude=amplitude,
            duration=duration,
            sample_rate=sample_rate
        )
        
        return audio, sample_rate
    
    def convert(
        self,
        solar_data: Union[SolarData, List[SolarData]],
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert solar data to audio using granular synthesis.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Use default parameters if not provided
        if parameters is None:
            sample_rate = 44100
            duration = 1.0
        else:
            sample_rate = parameters.sample_rate
            duration = parameters.duration
        
        # Handle single data point
        if isinstance(solar_data, SolarData):
            return self.convert_single(solar_data, parameters)
        
        # Handle multiple data points
        if not solar_data:
            # Empty list, return silence
            return np.zeros(int(sample_rate * duration)), sample_rate
        
        # Convert each data point
        audio_segments = []
        for data_point in solar_data:
            audio, _ = self.convert_single(data_point, parameters)
            audio_segments.append(audio)
        
        # Concatenate audio segments
        audio = np.concatenate(audio_segments)
        
        return audio, sample_rate


class MultiChannelConverter(AudioConverter):
    """Converter that generates multi-channel audio from solar data."""
    
    def __init__(
        self,
        converters: List[AudioConverter],
        voltage_range: Tuple[float, float] = (0.0, 48.0),
        current_range: Tuple[float, float] = (0.0, 10.0)
    ):
        """Initialize the multi-channel converter.
        
        Args:
            converters: List of converters for each channel
            voltage_range: Range of voltage values (min, max)
            current_range: Range of current values (min, max)
        """
        self.converters = converters
        self.voltage_range = voltage_range
        self.current_range = current_range
    
    def convert(
        self,
        solar_data: Union[SolarData, List[SolarData]],
        parameters: Optional[AudioParameters] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert solar data to multi-channel audio.
        
        Args:
            solar_data: Solar data to convert
            parameters: Audio parameters
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Use default parameters if not provided
        if parameters is None:
            sample_rate = 44100
            duration = 1.0
        else:
            sample_rate = parameters.sample_rate
            duration = parameters.duration
        
        # Convert using each converter
        channels = []
        for converter in self.converters:
            audio, sr = converter.convert(solar_data, parameters)
            
            # Ensure all channels have the same sample rate
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            channels.append(audio)
        
        # Ensure all channels have the same length
        max_length = max(len(channel) for channel in channels)
        for i in range(len(channels)):
            if len(channels[i]) < max_length:
                channels[i] = np.pad(channels[i], (0, max_length - len(channels[i])))
        
        # Stack channels
        audio = np.stack(channels)
        
        return audio, sample_rate