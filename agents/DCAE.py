import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.utils import size_after_conv, size_after_pooling, Reshape

class DCAE(nn.Module):
    def __init__(self, image_size, hidden_dim, lr=1e-3, loss_func=F.mse_loss) -> None:
        super().__init__()
        img_height, img_width, img_channel = image_size
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
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=ksize),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=ksize),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=ksize),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size),
            Reshape((-1, features[0])),
            nn.Linear(in_features=features[0], out_features=features[1]),
            # nn.BatchNorm1d(num_features=features[1]),
            nn.ReLU(),
            nn.Linear(in_features=features[1], out_features=features[2]),
            # nn.BatchNorm1d(num_features=features[2]),
            nn.ReLU(),
            nn.Linear(in_features=features[2], out_features=features[3]),
            # nn.BatchNorm1d(num_features=features[3]),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=features[3], out_features=features[2]),
            # nn.BatchNorm1d(num_features=features[2]),
            nn.ReLU(),
            nn.Linear(in_features=features[2], out_features=features[1]),
            # nn.BatchNorm1d(num_features=features[1]),
            nn.ReLU(),
            nn.Linear(in_features=features[1], out_features=features[0]),
            # nn.BatchNorm1d(num_features=features[0]),
            nn.ReLU(),
            Reshape((-1, channels[-1], after_height, after_width)),
            nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[3], kernel_size=ksize),
            nn.ReLU(),
            nn.Upsample(scale_factor=pooling_size),
            nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=ksize),
            nn.ReLU(),
            nn.Upsample(scale_factor=pooling_size),
            nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=ksize),
            nn.ReLU(),
            nn.Upsample(scale_factor=pooling_size),
            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=ksize),
            nn.Sigmoid(),
            nn.Upsample(size=(img_height, img_width))
        )
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss_func = loss_func
    
    def forward(self, x, return_pred=False):
        h = self.encoder(x)
        if not return_pred:
            return h
        else:
            x_pred = self.decoder(h)
            return h, x_pred
    