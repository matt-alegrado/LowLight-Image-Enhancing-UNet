import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, gn=False, downsample=True):
        super().__init__()

        num_groups = min(32, max(1, out_ch // 16))  # aim for 16 groups per channel, but no more than 32 total groups

        first_stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=first_stride, padding=1)
        self.bn1 = nn.GroupNorm(num_groups, out_ch) if gn else nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(.01, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(num_groups, out_ch) if gn else nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=first_stride)


    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + identity)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, gn=False):
        super().__init__()
        # F_g = #channels in gated signal (decoder),
        # F_l = #channels in skip (encoder),
        # F_int = # of intermediate features.
        norm_1 = nn.GroupNorm(8, F_int) if gn else nn.BatchNorm2d(F_int)
        norm_2 = nn.GroupNorm(8, F_int) if gn else nn.BatchNorm2d(F_int)
        norm_3 = nn.InstanceNorm2d(1) if gn else nn.BatchNorm2d(1)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_1
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_2
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            norm_3,
            nn.Sigmoid()
        )
        self.silu = nn.SiLU(inplace=True)

    def forward(self, g, x):
        # g = upsampled decoder features (gating signal), shape: [B, F_g, H, W]
        # x = skip connection features, shape: [B, F_l, H, W]
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.silu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, H, W] attention mask ∈ (0,1)
        return x * psi       # modulate skip features


class UNet(nn.Module):
    def __init__(self, in_channels=3, augmented_channels=2, hidden_dims=None, image_size=2048, dropout_rate=0,
                 dropout_layers=0, gn=False):
        """
        Assuming 2048x2048 center cropped images, smallest spatial resolution of the dataset, reduced to a power of 2
        :param in_channel:
        :param latent_dim:
        :param hidden_dims:
        """
        super(UNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        self.image_size = image_size
        self.augmented_channels = augmented_channels
        self.hidden_dims = hidden_dims
        self.gn = bool(gn)

        # Encoder
        dim_in = in_channels + augmented_channels
        self.enc_blocks = nn.ModuleList()

        for i, dim_out in enumerate(hidden_dims):
            block = ResBlock(dim_in, dim_out, gn=gn)
            self.enc_blocks.append(block)
            setattr(self, f"enc{i}", block)
            dim_in = dim_out

        self.final_size = self.image_size / 2 ** (
            len(hidden_dims))  # Ending image size after all the hidden layers, assuming stride 2 downscaling at each

        # Decoder
        reverse_hidden_dims = hidden_dims[::-1]
        self.dec_blocks = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        dim_in = reverse_hidden_dims[0]

        for i, dim_out in enumerate(reverse_hidden_dims[1:]):
            # Do dropout on the first n dropout_layers
            if i < dropout_layers:
                dropout = nn.Dropout2d(p=dropout_rate)
            else:
                dropout = nn.Identity()
            self.attn_gates.append(AttentionGate(dim_in, dim_in, F_int=dim_in // 2))

            # Choose either GN or BN
            if self.gn:
                num_groups = min(32, max(1, dim_out // 16))  # aim for 16 groups per channel, but no more than 32 total groups
                norm = nn.GroupNorm(num_groups, dim_out)
            else:
                norm = nn.BatchNorm2d(dim_out)

            block = nn.Sequential(
                # nn.ConvTranspose2d(dim_in*2, dim_out, kernel_size=4, stride=2, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8×8 → 16×16
                # nn.Conv2d(dim_in * 2, dim_out, kernel_size=3, stride=1, padding=1),
                ResBlock(dim_in * 2, dim_out, gn=gn, downsample=False),
                # norm,
                # nn.LeakyReLU(.01, inplace=True),
                dropout
            )
            self.dec_blocks.append(block)
            setattr(self, f"dec{i}", block)
            dim_in = dim_out

        # Choose either GN or BN
        if self.gn:
            num_groups = min(32, max(1,
                                     dim_out // 16))  # aim for 16 groups per channel, but no more than 32 total groups
            norm = nn.GroupNorm(num_groups, dim_out)
        else:
            norm = nn.BatchNorm2d(dim_out)

        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8×8 → 16×16
            ResBlock(dim_in * 2, dim_out, gn=gn, downsample=False),
            # nn.Conv2d(dim_in * 2, dim_out, kernel_size=3, stride=1, padding=1),
            # norm,
            # nn.LeakyReLU(.01, inplace=True),
        )
        self.attn_gates.append(AttentionGate(dim_in, dim_in, F_int=dim_in // 2))
        self.dec_blocks.append(block)
        setattr(self, f"dec{i + 1}", block)

        self.final_conv = nn.Conv2d(dim_out, in_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, input):
        x = input
        skips = []
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            skips.append(x)
        self.skips = skips[::-1]
        return x

    def decode(self, x):
        skip = self.attn_gates[0](self.skips[0], x)  # Do skip before first dec block
        x = torch.cat((x, skip), dim=1)
        for i, dec_block in enumerate(self.dec_blocks):
            x = dec_block(x)
            try:
                org_skip = self.skips[i+1]
                skip = self.attn_gates[i+1](org_skip, x)
                x = torch.cat((x, skip), dim=1)
            except IndexError:
                continue
        return F.sigmoid(self.final_conv(x))

    def forward(self, image):
        bottleneck = self.encode(image)  # [B,64,32,32]
        recon = self.decode(bottleneck)
        return recon

    def generate(self, image):
        return self(image)
