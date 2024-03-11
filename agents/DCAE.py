import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.utils import size_after_conv, size_after_pooling, Reshape, FE

class DCAE(FE):
    def __init__(self, img_height, img_width, img_channel, hidden_dim, lr=1e-3, net_activation=nn.ReLU(inplace=True), hidden_activation=F.tanh, loss_func=F.mse_loss) -> None:
        super().__init__()
        channels = [img_channel, 32, 64, 128, 256]
        after_height = img_height
        after_width = img_width
        ksize = 3
        pooling_size = 2
        for _ in range(len(channels)-1):
            after_height = size_after_conv(after_height, ksize=ksize)
            after_height = size_after_pooling(after_height, pooling_size)
            after_width = size_after_conv(after_width, ksize=ksize)
            after_width = size_after_pooling(after_width, pooling_size)
        after_size = after_height * after_width * channels[-1]
        features = [after_size, 1000, 150, hidden_dim]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=ksize),
            net_activation,
            nn.MaxPool2d(pooling_size),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=ksize),
            net_activation,
            nn.MaxPool2d(pooling_size),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=ksize),
            net_activation,
            nn.MaxPool2d(pooling_size),
            nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=ksize),
            net_activation,
            nn.MaxPool2d(pooling_size),
            Reshape((-1, features[0])),
            nn.Linear(in_features=features[0], out_features=features[1]),
            # nn.BatchNorm1d(num_features=features[1]),
            net_activation,
            nn.Linear(in_features=features[1], out_features=features[2]),
            # nn.BatchNorm1d(num_features=features[2]),
            net_activation,
            nn.Linear(in_features=features[2], out_features=features[3]),
            # nn.BatchNorm1d(num_features=features[3]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=features[3], out_features=features[2]),
            # nn.BatchNorm1d(num_features=features[2]),
            net_activation,
            nn.Linear(in_features=features[2], out_features=features[1]),
            # nn.BatchNorm1d(num_features=features[1]),
            net_activation,
            nn.Linear(in_features=features[1], out_features=features[0]),
            # nn.BatchNorm1d(num_features=features[0]),
            net_activation,
            Reshape((-1, channels[-1], after_height, after_width)),
            nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[3], kernel_size=ksize),
            net_activation,
            nn.Upsample(scale_factor=pooling_size),
            nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=ksize),
            net_activation,
            nn.Upsample(scale_factor=pooling_size),
            nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=ksize),
            net_activation,
            nn.Upsample(scale_factor=pooling_size),
            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=ksize),
            nn.Sigmoid(),
            nn.Upsample(size=(img_height, img_width))
        )
        self.net_activation = net_activation
        self.hidden_activation = hidden_activation
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss_func = loss_func
    
    def forward(self, x: th.Tensor, return_pred: bool = False):
        h = self.encoder(x)
        if not return_pred:
            return self.hidden_activation(h)
        else:
            x_pred = self.decoder(self.net_activation(h))
            return h, x_pred
    
    def loss(self, x: th.Tensor) -> th.Tensor:
        _, y = self.forward(x, return_pred=True)
        return self.loss_func(y, x)
    