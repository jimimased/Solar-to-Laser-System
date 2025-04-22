"""
RAVE model implementation.

This module provides the implementation of the RAVE (Realtime Audio Variational autoEncoder) model.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block for the RAVE model."""
    
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        kernel_size: int = 3,
        activation: nn.Module = nn.LeakyReLU(0.2)
    ):
        """Initialize the residual block.
        
        Args:
            channels: Number of channels
            dilation: Dilation factor
            kernel_size: Kernel size
            activation: Activation function
        """
        super().__init__()
        
        self.activation = activation
        
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.activation(x + residual)


class ConvBlock(nn.Module):
    """Convolutional block for the RAVE model."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: nn.Module = nn.LeakyReLU(0.2),
        batch_norm: bool = True
    ):
        """Initialize the convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            activation: Activation function
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.activation = activation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class Encoder(nn.Module):
    """Encoder for the RAVE model."""
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        latent_dim: int = 128,
        n_residual_blocks: int = 4
    ):
        """Initialize the encoder.
        
        Args:
            in_channels: Number of input channels
            channels: Number of channels in the hidden layers
            latent_dim: Dimension of the latent space
            n_residual_blocks: Number of residual blocks
        """
        super().__init__()
        
        # Initial convolution
        self.conv_in = ConvBlock(in_channels, channels, kernel_size=7, stride=1, padding=3)
        
        # Downsampling convolutions
        self.down1 = ConvBlock(channels, channels, kernel_size=4, stride=2, padding=1)
        self.down2 = ConvBlock(channels, channels, kernel_size=4, stride=2, padding=1)
        self.down3 = ConvBlock(channels, channels, kernel_size=4, stride=2, padding=1)
        self.down4 = ConvBlock(channels, channels, kernel_size=4, stride=2, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels, dilation=2**i)
            for i in range(n_residual_blocks)
        ])
        
        # Output convolution
        self.conv_out = ConvBlock(channels, latent_dim * 2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance of the latent distribution
        """
        # Initial convolution
        x = self.conv_in(x)
        
        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output convolution
        x = self.conv_out(x)
        
        # Split into mean and log variance
        mean, log_var = torch.chunk(x, 2, dim=1)
        
        return mean, log_var


class Decoder(nn.Module):
    """Decoder for the RAVE model."""
    
    def __init__(
        self,
        out_channels: int = 1,
        channels: int = 128,
        latent_dim: int = 128,
        n_residual_blocks: int = 4
    ):
        """Initialize the decoder.
        
        Args:
            out_channels: Number of output channels
            channels: Number of channels in the hidden layers
            latent_dim: Dimension of the latent space
            n_residual_blocks: Number of residual blocks
        """
        super().__init__()
        
        # Initial convolution
        self.conv_in = ConvBlock(latent_dim, channels, kernel_size=3, stride=1, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels, dilation=2**i)
            for i in range(n_residual_blocks)
        ])
        
        # Upsampling convolutions
        self.up1 = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        
        self.up2 = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm1d(channels)
        
        self.up3 = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm1d(channels)
        
        self.up4 = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.BatchNorm1d(channels)
        
        # Output convolution
        self.conv_out = nn.Conv1d(channels, out_channels, kernel_size=7, stride=1, padding=3)
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        # Initial convolution
        x = self.conv_in(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Upsampling
        x = self.up1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.up2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        x = self.up3(x)
        x = self.norm3(x)
        x = self.activation(x)
        
        x = self.up4(x)
        x = self.norm4(x)
        x = self.activation(x)
        
        # Output convolution
        x = self.conv_out(x)
        x = torch.tanh(x)
        
        return x


class RAVE(nn.Module):
    """RAVE (Realtime Audio Variational autoEncoder) model."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 128,
        latent_dim: int = 128,
        n_residual_blocks: int = 4
    ):
        """Initialize the RAVE model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            channels: Number of channels in the hidden layers
            latent_dim: Dimension of the latent space
            n_residual_blocks: Number of residual blocks
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(
            in_channels=in_channels,
            channels=channels,
            latent_dim=latent_dim,
            n_residual_blocks=n_residual_blocks
        )
        
        self.decoder = Decoder(
            out_channels=out_channels,
            channels=channels,
            latent_dim=latent_dim,
            n_residual_blocks=n_residual_blocks
        )
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.
        
        Args:
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent space.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean, log variance, and sampled latent vector
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return mean, log_var, z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to output.
        
        Args:
            z: Latent vector
        
        Returns:
            torch.Tensor: Output tensor
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Reconstructed output, mean, log variance, and sampled latent vector
        """
        mean, log_var, z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mean, log_var, z
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from the latent space.
        
        Args:
            n_samples: Number of samples
            device: Device to use
        
        Returns:
            torch.Tensor: Sampled output
        """
        z = torch.randn(n_samples, self.latent_dim, 1, device=device)
        return self.decode(z)
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Interpolate between two inputs.
        
        Args:
            x1: First input
            x2: Second input
            steps: Number of interpolation steps
        
        Returns:
            torch.Tensor: Interpolated outputs
        """
        # Encode inputs
        mean1, log_var1, z1 = self.encode(x1)
        mean2, log_var2, z2 = self.encode(x2)
        
        # Interpolate in latent space
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        z_interp = []
        
        for alpha in alphas:
            z = alpha * z1 + (1 - alpha) * z2
            z_interp.append(z)
        
        z_interp = torch.cat(z_interp, dim=0)
        
        # Decode interpolated latent vectors
        return self.decode(z_interp)


class RAVELoss(nn.Module):
    """Loss function for the RAVE model."""
    
    def __init__(self, kl_weight: float = 0.01):
        """Initialize the loss function.
        
        Args:
            kl_weight: Weight for the KL divergence term
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor
            x_recon: Reconstructed tensor
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and loss components
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(x_recon, x)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


def create_rave_model(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: int = 128,
    latent_dim: int = 128,
    n_residual_blocks: int = 4,
    device: Optional[torch.device] = None
) -> RAVE:
    """Create a RAVE model.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: Number of channels in the hidden layers
        latent_dim: Dimension of the latent space
        n_residual_blocks: Number of residual blocks
        device: Device to use
    
    Returns:
        RAVE: RAVE model
    """
    model = RAVE(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        latent_dim=latent_dim,
        n_residual_blocks=n_residual_blocks
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


def save_rave_model(model: RAVE, path: str):
    """Save a RAVE model.
    
    Args:
        model: RAVE model
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Saved RAVE model to {path}")


def load_rave_model(
    path: str,
    in_channels: int = 1,
    out_channels: int = 1,
    channels: int = 128,
    latent_dim: int = 128,
    n_residual_blocks: int = 4,
    device: Optional[torch.device] = None
) -> RAVE:
    """Load a RAVE model.
    
    Args:
        path: Path to load the model from
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: Number of channels in the hidden layers
        latent_dim: Dimension of the latent space
        n_residual_blocks: Number of residual blocks
        device: Device to use
    
    Returns:
        RAVE: RAVE model
    """
    model = create_rave_model(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        latent_dim=latent_dim,
        n_residual_blocks=n_residual_blocks,
        device=device
    )
    
    model.load_state_dict(torch.load(path, map_location=device))
    logger.info(f"Loaded RAVE model from {path}")
    
    return model