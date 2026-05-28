import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for NeRF input
    Maps input from R^d to R^(2*L*d + d) using sine and cosine functions
    """

    def __init__(self, num_freqs: int, include_input: bool = True):
        """
        Args:
            num_freqs: number of frequency bands (L)
            include_input: whether to include raw input in output
        """

        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        # frequency bands [2^0, 2^1, ..., 2^(L-1)]
        self.register_buffer("freq_bands", 2.0 ** torch.arange(num_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor (..., d)

        Returns:
            encoded tensor (..., 2*L*d + d) if include_input else (..., 2*L*d)
        """
        out = [x] if self.include_input else []

        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)

    def output_dim(self, input_dim: int = 3) -> int:
        base = input_dim if self.include_input else 0
        return base + 2 * self.num_freqs * input_dim


class NeRF(nn.Module):
    """
    Vanilla NeRF MLP

    Input:
        position (x, y, z) -> positional encoding -> MLP -> density + feature
        direction (theta, phi) - > positional encoding -> MLP -> RGB

    Output:
        rgb: (3,) color
        density: (1,) volume density (sigma)
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4,
    ):
        """
        Args:
            pos_freqs: number of frequency bands for position encoding
            dir_freqs: number of frequency bands for direction encoding
            hidden_dim: hidden layer dimension
            num_layers: number of MLP layers
            skip_layer: layer index to concatenate input again (skip connection)
        """
        super().__init__()

        self.skip_layer = skip_layer

        # positional encoding
        self.pos_enc = PositionalEncoding(num_freqs=pos_freqs, include_input=True)
        self.dir_enc = PositionalEncoding(num_freqs=dir_freqs, include_input=True)

        pos_dim = 3 + 2 * pos_freqs * 3  # 3 + 2*10*3 = 63
        dir_dim = 3 + 2 * dir_freqs * 3  # 3 + 2*4*3 = 27

        # positional MLP (density brach)
        self.pts_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.pts_layers.append(nn.Linear(pos_dim, hidden_dim))
            elif i == skip_layer:
                # skip connection: concat original input
                self.pts_layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                self.pts_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # density output head
        self.density_head = nn.Linear(hidden_dim, 1)

        # feature vector for color branch
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        # direction MLP (color brach)
        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),  # RGB in [0, 1]
        )

        self.relu = nn.ReLU()

    def forward(
        self, pts: torch.Tensor, dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pts: 3D sample points (..., 3)
            dirs: ray directions (... , 3)

        Returns:
            rgb: predicted color (..., 3)
            density: predicted density (..., 1)
        """
        # positional encoding
        pts_enc = self.pos_enc(pts)  # (..., 63)
        dirs_enc = self.dir_enc(dirs)  # (..., 27)

        # positional MLP forward
        h = pts_enc
        for i, layer in enumerate(self.pts_layers):
            if i == self.skip_layer:
                h = torch.cat([h, pts_enc], dim=-1)
            h = self.relu(layer(h))

        # density (no activation: raw sigma, clipped in renderer)
        density = self.density_head(h)  # (..., 1)

        # feature vector
        feature = self.feature_head(h)  # (..., hidden_dim)

        # color MLP
        h = torch.cat([feature, dirs_enc], dim=-1)
        rgb = self.color_layers(h)  # (..., 3)

        return rgb, density


if __name__ == "__main__":
    # for test
    model = NeRF()
    pts = torch.randn(1024, 3)
    dirs = torch.randn(1024, 3)
    rgb, density = model(pts, dirs)
    print("rgb shape:", rgb.shape)  # (1024, 3)
    print("density shape:", density.shape)  # (1024, 1)
    print("num params:", sum(p.numel() for p in model.parameters()))
