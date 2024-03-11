from typing import Tuple, Union, Callable
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import Reshape, torch_log, FE

class ConvVAE(FE):
    def __init__(self, img_height, img_width, img_channel, hidden_dim, lr=1e-3, net_activation: nn.Module=nn.ReLU(inplace=True), device='cpu',
                 hidden_activation: Callable[[th.Tensor], th.Tensor]=F.tanh) -> None:
        super().__init__()
        channels = [img_channel, 32, 64, 128, 256, 512]
        kernel_size = 3

        modules = []
        for i in range(len(channels)-1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_size),
                    # nn.BatchNorm2d(channels[i+1]),
                    net_activation,
                )
            )
        self.encoder = nn.Sequential(*modules)
        with th.no_grad():
            shape = self.encoder(th.ones((1, img_channel, img_height, img_width), dtype=th.float)).shape
            n_flatten = shape[1] * shape[2] * shape[3]
        self.encoder.append(nn.Flatten())
        self.enc_mean = nn.Linear(n_flatten, hidden_dim)
        self.enc_var = nn.Linear(n_flatten, hidden_dim)

        channels.reverse()
        modules = [nn.Linear(hidden_dim, n_flatten), Reshape((-1, *shape[1:]))]
        for i in range(len(channels)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_size),
                    # nn.BatchNorm2d(channels[i+1]),
                    net_activation
                )
            )
        self.decoder = nn.Sequential(*modules, nn.Upsample(size=(img_height, img_width)), nn.Sigmoid())
        
        self.hidden_activation = hidden_activation
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.device = device
    
    def _encode(self, input: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        tmp = self.encoder(input)
        mu = self.enc_mean(tmp)
        std = F.softplus(self.enc_var(tmp))
        return mu, std
    
    def _decode(self, z: th.Tensor) -> th.Tensor:
        return self.decoder(z)
    
    def _reparameterize(self, mu: th.Tensor, std: th.Tensor) -> th.Tensor:
        if self.training:
            eps = th.randn(mu.shape).to(self.device)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x: th.Tensor, return_pred: bool=False) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        mu, std = self._encode(x)
        z = self._reparameterize(mu, std)
        if return_pred:
            y = self._decode(z)
            return z, y
        else:
            return self.hidden_activation(z)
    
    def loss(self, x: th.Tensor) -> th.Tensor:
        mu, std = self._encode(x)
        kl = -0.5 * th.mean(th.sum(1+torch_log(std**2) - mu**2 - std**2, dim=1))
        z = self._reparameterize(mu, std)
        y = self._decode(z)
        re = -th.mean(th.sum(x*torch_log(y) + (1-x)*torch_log(1-y), dim=1))
        return kl + re

    def to(self, device):
        super().to(device)
        self.device = device
