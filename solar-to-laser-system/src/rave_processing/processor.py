"""
Audio processor implementation using RAVE.

This module provides classes for processing audio using the RAVE model.
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

from .model import RAVE, create_rave_model, load_rave_model

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Base class for audio processors."""
    
    def process(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Process audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        
        Returns:
            Tuple[np.ndarray, int]: Processed audio data and sample rate
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def extract_features(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract features from audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        
        Returns:
            np.ndarray: Extracted features
        """
        raise NotImplementedError("Subclasses must implement extract_features()")
    
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


class RAVEProcessor(AudioProcessor):
    """Audio processor using RAVE."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        latent_dim: int = 128,
        channels: int = 128,
        n_residual_blocks: int = 4,
        segment_size: int = 16384,
        hop_size: int = 8192
    ):
        """Initialize the RAVE processor.
        
        Args:
            model_path: Path to the RAVE model
            device: Device to use
            latent_dim: Dimension of the latent space
            channels: Number of channels in the hidden layers
            n_residual_blocks: Number of residual blocks
            segment_size: Size of audio segments for processing
            hop_size: Hop size between segments
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.latent_dim = latent_dim
        self.channels = channels
        self.n_residual_blocks = n_residual_blocks
        self.segment_size = segment_size
        self.hop_size = hop_size
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.model = load_rave_model(
                path=model_path,
                in_channels=1,
                out_channels=1,
                channels=channels,
                latent_dim=latent_dim,
                n_residual_blocks=n_residual_blocks,
                device=self.device
            )
        else:
            self.model = create_rave_model(
                in_channels=1,
                out_channels=1,
                channels=channels,
                latent_dim=latent_dim,
                n_residual_blocks=n_residual_blocks,
                device=self.device
            )
            logger.warning("No model path provided or model not found. Using untrained model.")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def preprocess_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        target_sr: int = 44100
    ) -> np.ndarray:
        """Preprocess audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            target_sr: Target sample rate
        
        Returns:
            np.ndarray: Preprocessed audio data
        """
        # Resample if needed
        if sample_rate != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        # Normalize
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        return audio_data
    
    def process_segment(self, segment: np.ndarray) -> np.ndarray:
        """Process a single audio segment.
        
        Args:
            segment: Audio segment
        
        Returns:
            np.ndarray: Processed audio segment
        """
        # Convert to tensor
        x = torch.from_numpy(segment).float().to(self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Process with model
        with torch.no_grad():
            # Encode
            mean, log_var, z = self.model.encode(x)
            
            # Decode
            x_recon = self.model.decode(z)
        
        # Convert back to numpy
        processed = x_recon.squeeze().cpu().numpy()
        
        return processed
    
    def extract_latent(self, segment: np.ndarray) -> np.ndarray:
        """Extract latent representation from a single audio segment.
        
        Args:
            segment: Audio segment
        
        Returns:
            np.ndarray: Latent representation
        """
        # Convert to tensor
        x = torch.from_numpy(segment).float().to(self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Extract latent representation
        with torch.no_grad():
            mean, log_var, z = self.model.encode(x)
        
        # Convert to numpy
        latent = z.squeeze().cpu().numpy()
        
        return latent
    
    def process(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Process audio data using RAVE.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        
        Returns:
            Tuple[np.ndarray, int]: Processed audio data and sample rate
        """
        # Preprocess audio
        audio_data = self.preprocess_audio(audio_data, sample_rate)
        
        # Pad audio to ensure it's long enough for segmentation
        if len(audio_data) < self.segment_size:
            audio_data = np.pad(audio_data, (0, self.segment_size - len(audio_data)))
        
        # Process in segments
        processed_segments = []
        
        for i in tqdm(range(0, len(audio_data) - self.segment_size + 1, self.hop_size), desc="Processing audio"):
            # Extract segment
            segment = audio_data[i:i + self.segment_size]
            
            # Process segment
            processed = self.process_segment(segment)
            
            # Add to list
            processed_segments.append(processed)
        
        # Overlap-add to reconstruct the full audio
        processed_audio = np.zeros(len(audio_data))
        window = np.hanning(self.segment_size)
        
        for i, segment in enumerate(processed_segments):
            start = i * self.hop_size
            end = start + self.segment_size
            processed_audio[start:end] += segment * window
        
        # Normalize
        processed_audio = processed_audio / (np.max(np.abs(processed_audio)) + 1e-8)
        
        return processed_audio, 44100  # Always return 44100 Hz
    
    def extract_features(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract features from audio data using RAVE.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        
        Returns:
            np.ndarray: Extracted features (latent representations)
        """
        # Preprocess audio
        audio_data = self.preprocess_audio(audio_data, sample_rate)
        
        # Pad audio to ensure it's long enough for segmentation
        if len(audio_data) < self.segment_size:
            audio_data = np.pad(audio_data, (0, self.segment_size - len(audio_data)))
        
        # Extract features in segments
        features = []
        
        for i in tqdm(range(0, len(audio_data) - self.segment_size + 1, self.hop_size), desc="Extracting features"):
            # Extract segment
            segment = audio_data[i:i + self.segment_size]
            
            # Extract latent representation
            latent = self.extract_latent(segment)
            
            # Add to list
            features.append(latent)
        
        # Stack features
        features = np.stack(features)
        
        return features
    
    def manipulate_latent(
        self,
        latent: np.ndarray,
        manipulation_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Manipulate latent representation.
        
        Args:
            latent: Latent representation
            manipulation_fn: Function to manipulate the latent representation
        
        Returns:
            np.ndarray: Manipulated latent representation
        """
        # Apply manipulation function
        manipulated = manipulation_fn(latent)
        
        return manipulated
    
    def generate_from_latent(
        self,
        latent: np.ndarray,
        segment_size: int = None
    ) -> np.ndarray:
        """Generate audio from latent representation.
        
        Args:
            latent: Latent representation
            segment_size: Size of audio segments for generation
        
        Returns:
            np.ndarray: Generated audio
        """
        segment_size = segment_size or self.segment_size
        
        # Convert to tensor
        z = torch.from_numpy(latent).float().to(self.device)
        
        # Add channel dimension if needed
        if len(z.shape) == 1:
            z = z.unsqueeze(0).unsqueeze(-1)  # [latent_dim] -> [1, latent_dim, 1]
        elif len(z.shape) == 2:
            z = z.unsqueeze(0)  # [batch, latent_dim] -> [batch, latent_dim, 1]
        
        # Generate audio
        with torch.no_grad():
            x_gen = self.model.decode(z)
        
        # Convert to numpy
        audio = x_gen.squeeze().cpu().numpy()
        
        return audio
    
    def interpolate(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        steps: int = 10
    ) -> np.ndarray:
        """Interpolate between two audio samples.
        
        Args:
            audio1: First audio sample
            audio2: Second audio sample
            steps: Number of interpolation steps
        
        Returns:
            np.ndarray: Interpolated audio
        """
        # Preprocess audio
        audio1 = self.preprocess_audio(audio1, 44100)
        audio2 = self.preprocess_audio(audio2, 44100)
        
        # Ensure both audios have the same length
        min_length = min(len(audio1), len(audio2))
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]
        
        # Pad to segment size if needed
        if min_length < self.segment_size:
            audio1 = np.pad(audio1, (0, self.segment_size - min_length))
            audio2 = np.pad(audio2, (0, self.segment_size - min_length))
            min_length = self.segment_size
        
        # Convert to tensors
        x1 = torch.from_numpy(audio1).float().to(self.device)
        x1 = x1.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        x2 = torch.from_numpy(audio2).float().to(self.device)
        x2 = x2.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Encode
        with torch.no_grad():
            mean1, log_var1, z1 = self.model.encode(x1)
            mean2, log_var2, z2 = self.model.encode(x2)
        
        # Interpolate in latent space
        interpolated_audio = []
        
        for alpha in np.linspace(0, 1, steps):
            # Interpolate latent vectors
            z = alpha * z1 + (1 - alpha) * z2
            
            # Decode
            with torch.no_grad():
                x_interp = self.model.decode(z)
            
            # Convert to numpy
            audio_interp = x_interp.squeeze().cpu().numpy()
            
            # Add to list
            interpolated_audio.append(audio_interp)
        
        # Concatenate
        interpolated_audio = np.concatenate(interpolated_audio)
        
        return interpolated_audio
    
    def train(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.0001,
        kl_weight: float = 0.01,
        save_path: Optional[str] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None
    ):
        """Train the RAVE model on audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            kl_weight: Weight for the KL divergence term
            save_path: Path to save the trained model
            callback: Callback function called after each epoch
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import Adam
        from .model import RAVELoss
        
        # Preprocess audio
        audio_data = self.preprocess_audio(audio_data, sample_rate)
        
        # Create segments
        segments = []
        
        for i in range(0, len(audio_data) - self.segment_size + 1, self.hop_size):
            segment = audio_data[i:i + self.segment_size]
            segments.append(segment)
        
        # Convert to tensor
        segments = torch.from_numpy(np.array(segments)).float().to(self.device)
        segments = segments.unsqueeze(1)  # Add channel dimension
        
        # Create dataset and dataloader
        dataset = TensorDataset(segments)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = RAVELoss(kl_weight=kl_weight)
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                x = batch[0]
                
                # Forward pass
                x_recon, mean, log_var, z = self.model(x)
                
                # Calculate loss
                loss, loss_components = loss_fn(x, x_recon, mean, log_var)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record loss
                epoch_losses.append({k: v.item() for k, v in loss_components.items()})
            
            # Calculate average losses
            avg_losses = {k: np.mean([loss[k] for loss in epoch_losses]) for k in epoch_losses[0].keys()}
            
            # Log losses
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_losses['total_loss']:.4f}")
            
            # Call callback if provided
            if callback:
                callback(epoch, avg_losses)
        
        # Set model back to evaluation mode
        self.model.eval()
        
        # Save model if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Saved trained model to {save_path}")


class SpectralProcessor(AudioProcessor):
    """Audio processor using spectral processing."""
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """Initialize the spectral processor.
        
        Args:
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of mel bands
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def process(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Process audio data using spectral processing.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        
        Returns:
            Tuple[np.ndarray, int]: Processed audio data and sample rate
        """
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        # Compute STFT
        stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Apply some processing to the STFT
        # For example, spectral filtering
        stft_processed = stft * 0.8  # Simple gain reduction
        
        # Inverse STFT
        audio_processed = librosa.istft(stft_processed, hop_length=self.hop_length)
        
        return audio_processed, sample_rate
    
    def extract_features(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract features from audio data using spectral processing.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        
        Returns:
            np.ndarray: Extracted features
        """
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db