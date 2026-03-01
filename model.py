import torch.nn as nn
import tinycudann as tcnn


class GeoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(beta=10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pe):
        return self.layers(pe).squeeze(-1)


class ColorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Linear(input_dim, 3)

    def forward(self, pe):
        return self.layers(pe)


class SDFNet(nn.Module):
    def __init__(self, base_hidden_dim=16):
        super().__init__()

        print(f"Training in single region mode, hidden_dim={base_hidden_dim}")

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 10,
                "n_features_per_level": 4,
                "log2_hashmap_size": 16,
                "base_resolution": 14,
                "per_level_scale": 1.5,
            },
        )
        self.input_dim = self.encoder.n_output_dims
        self.geo_head = GeoNet(self.input_dim, base_hidden_dim)
        self.color_head = ColorNet(self.input_dim)

    def forward(self, x):
        pe = self.encoder(x).float()
        return self.geo_head(pe), self.color_head(pe)
