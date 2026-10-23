"""CNN Autoencoder for spatial crowd appearance anomaly detection."""

import torch
import torch.nn as nn


class CrowdAutoencoder(nn.Module):
    """Convolutional Autoencoder for learning normal spatial crowd representations.

    The model compresses an input image frame (e.g. 256x256x3) through 3 convolutional
    downsampling layers into a compact latent representation, and reconstructs the original
    frame via 3 transposed convolutional upsampling layers with Sigmoid activation.

    Abnormal spatial appearances yield elevated Mean Squared Reconstruction Error.

    Args:
        in_channels: Number of input and output image color channels. Defaults to 3.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatial autoencoding.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W).
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out
