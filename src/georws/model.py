import torch.nn as nn
import torch.nn.functional as F

from src.zsl.utils import weights_init


class GeoRWSModel(nn.Module):
    def __init__(self, input_dim, weight_dim, latent_dim, hidden_dim, adaptive_encoder):
        super().__init__()
        self.encoder = (nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, latent_dim))
                        if adaptive_encoder else None)
        self.decoder = (nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, input_dim))
                        if adaptive_encoder else None)
        self.score_z = nn.Linear(latent_dim if adaptive_encoder else input_dim, hidden_dim)
        self.score_w = nn.Linear(weight_dim, hidden_dim, bias=False)
        self.head = nn.Linear(hidden_dim, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if layer.bias is None:
                    nn.init.normal_(layer.weight, 0.0, 0.02)
                else:
                    weights_init(layer)

    def forward(self, a, w):
        z = self.encoder(a) if self.encoder is not None else a
        recon = self.decoder(z) if self.decoder is not None else None
        logits = self.head(F.relu(self.score_z(z) + self.score_w(w)[None])).squeeze(-1)
        return logits, z, recon
