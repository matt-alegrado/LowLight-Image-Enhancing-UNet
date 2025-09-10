import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, gn=False):
        super().__init__()

        num_groups = min(32, max(1, out_ch // 16))  # aim for 16 groups per channel, but no more than 32 total groups

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(num_groups, out_ch) if gn else nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(.01, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(num_groups, out_ch) if gn else nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + identity)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=None, dropout_start=1, dropout_p=0.3, gn=False):
        super(Discriminator, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        dim_in = in_channels
        layers_list = []
        for i, hidden_dim in enumerate(hidden_dims):  # conv blocks for all but last hidden layer
            if i >= dropout_start:
                dropout = nn.Dropout2d(p=dropout_p)
            else:
                dropout = nn.Identity()

            layers_list.append(ResBlock(dim_in, hidden_dim, gn=gn))
            layers_list.append(dropout)

            dim_in = hidden_dim
            
        self.model = nn.Sequential(
            *layers_list,
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, stride=1, padding=1),  # Last hidden layer translates to single channel output
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.model(x)