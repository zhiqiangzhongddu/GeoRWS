import torch.nn as nn


class LINEAR(nn.Module):
    def __init__(self, input_dim, nclass, bias = True):
        super(LINEAR, self).__init__()
        self.fc = nn.Linear(input_dim, nclass, bias)

    def forward(self, x):
        o = self.fc(x)
        return o


class AUTOENCODER(nn.Module):
    def __init__(
            self, cfg, input_dim, embed_dim,
            output_dim = None, num_layers = 3, vae = False, bias = True
    ):
        super(AUTOENCODER, self).__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is None:
            self.output_dim = input_dim
        if vae:
            self.embed_dim = [2 * embed_dim, embed_dim]
        else:
            self.embed_dim = [embed_dim, embed_dim]

        if num_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], self.output_dim)
            )
        elif num_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )
        elif num_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim[0], self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )
        else:
            raise NotImplementedError

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)


class JOINT_AUTOENCODER(nn.Module):
    def __init__(self, cfg, autoencoder1, autoencoder2):
        super(JOINT_AUTOENCODER, self).__init__()
        self.cfg = cfg
        self.ae1 = autoencoder1
        self.ae2 = autoencoder2

    def encode1(self, x):
        return self.ae1.encode(x)

    def encode2(self, x):
        return self.ae2.encode(x)

    def decode1(self, x):
        return self.ae1.decode(x)

    def decode2(self, x):
        return self.ae2.decode(x)

    def forward(self, x):
        attribute_in, weight_in = x
        latent_attribute = self.encode1(attribute_in)
        latent_weight = self.encode2(weight_in)

        attribute_from_attribute = self.decode1(latent_attribute)
        attribute_from_weight = self.decode1(latent_weight)
        weight_from_weight = self.decode2(latent_weight)
        weight_from_attribute = self.decode2(latent_attribute)

        return (attribute_from_attribute, attribute_from_weight, 
                weight_from_weight, weight_from_attribute, 
                latent_attribute, latent_weight)

    def predict(self, x):
        # Given attributes, predict weights
        latent_attribute = self.encode1(x)
        return self.decode2(latent_attribute)
