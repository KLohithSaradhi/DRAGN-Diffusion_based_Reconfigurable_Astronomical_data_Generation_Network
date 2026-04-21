import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        channels, kernels, strides, paddings = params["channels"], params["kernels"], params["strides"], params["paddings"]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernels[i], strides[i], paddings[i]))
            if i < len(channels) - 2: 
                layers.append(nn.BatchNorm2d(channels[i+1]))
                layers.append(self.act)
            else:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        channels, kernels, strides, paddings = params["channels"][::-1], params["kernels"][::-1], params["strides"][::-1], params["paddings"][::-1]
        out_paddings = params.get("out_paddings", [0]*len(kernels))[::-1] 
        layers = []
        for i in range(len(channels) - 1):
            if i < len(channels) - 1:
                layers.append(nn.ConvTranspose2d(channels[i], channels[i+1], kernels[i], strides[i], paddings[i], output_padding=out_paddings[i]))
                layers.append(nn.BatchNorm2d(channels[i+1]))
                layers.append(self.act)
            else:
                layers.append(nn.Conv2d(channels[i], channels[i+1], 3, 1, 1))
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
    def forward(self, x):
        latents = self.encoder(x)
        return self.decoder(latents), latents