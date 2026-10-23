"""ConvLSTM Cell and Spatio-Temporal Autoencoder for Crowd Anomaly Detection."""

from typing import Tuple
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """2D Convolutional Long Short-Term Memory (ConvLSTM) Cell.

    Maintains spatial dimensions while learning temporal transitions using 2D convolutions
    across input, forget, output, and cell state updates.

    Args:
        input_dim: Number of input feature channels.
        hidden_dim: Number of hidden state feature channels.
        kernel_size: 2D convolution kernel size. Defaults to 3.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward step for single timestep.

        Args:
            x: Input tensor of shape (B, input_dim, H, W).
            h_prev: Previous hidden state of shape (B, hidden_dim, H, W).
            c_prev: Previous cell state of shape (B, hidden_dim, H, W).

        Returns:
            Tuple of (h_next, c_next), both of shape (B, hidden_dim, H, W).
        """
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMAutoencoderV2(nn.Module):
    """Spatio-Temporal ConvLSTM Autoencoder for learning temporal crowd dynamics.

    Encodes consecutive frames into feature maps, maintains sequential recurrence
    with a ConvLSTM cell, and reconstructs each frame at every timestep.

    Args:
        in_channels: Number of image channels. Defaults to 3.
        hidden_dim: Latent representation channels. Defaults to 64.
        kernel_size: Convolution kernel size for ConvLSTM cell. Defaults to 3.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Spatial Feature Encoder (downsample by 4x)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # ConvLSTM Recurrent Unit
        self.convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size
        )

        # Spatial Feature Decoder (upsample by 4x)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass across full temporal sequence.

        Args:
            x_seq: Sequence tensor of shape (B, T, C, H, W).

        Returns:
            Reconstructed sequence tensor of shape (B, T, C, H, W).
        """
        B, T, C, H, W = x_seq.shape

        h = torch.zeros(B, self.hidden_dim, H // 4, W // 4, device=x_seq.device)
        c = torch.zeros_like(h)

        outputs = []
        for t in range(T):
            feat = self.enc(x_seq[:, t])
            h, c = self.convlstm(feat, h, c)
            out = self.dec(h)
            outputs.append(out)

        return torch.stack(outputs, dim=1)
