import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid(self.alpha * (x - self.beta))


class CrossChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.conv_mask = nn.Conv1d(channels, channels, 1)

        self.sigmoid = LearnableSigmoid()

    def forward(self, x1, x2):
        h1 = torch.tanh(self.conv1(x1))
        h2 = torch.tanh(self.conv2(x2))

        z = torch.abs(h1 * h2)
        z = self.sigmoid(z)

        mask = torch.sigmoid(self.conv_mask(z))

        a1 = x1 + mask * x1
        a2 = x2 + mask * x2

        return a1, a2


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=15, padding=7),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.main(x)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.main(x)


class CrossChannelWaveUNet(nn.Module):
    def __init__(self, n_layers=10, channels_interval=24):
        super().__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval

        # ===== Encoder (two branches) =====
        encoder_in = [1] + [i * channels_interval for i in range(1, n_layers)]
        encoder_out = [i * channels_interval for i in range(1, n_layers + 1)]

        self.encoder1 = nn.ModuleList()
        self.encoder2 = nn.ModuleList()

        for i in range(n_layers):
            self.encoder1.append(DownSamplingLayer(encoder_in[i], encoder_out[i]))
            self.encoder2.append(DownSamplingLayer(encoder_in[i], encoder_out[i]))

        # ===== Attention =====
        self.attn = nn.ModuleList([
            CrossChannelAttention(encoder_out[i])
            for i in range(n_layers)
        ])

        # ===== Bottleneck =====
        bottleneck_ch = n_layers * channels_interval
        self.middle = nn.Sequential(
            nn.Conv1d(2 * bottleneck_ch, bottleneck_ch, kernel_size=15, padding=7),
            nn.BatchNorm1d(bottleneck_ch),
            nn.LeakyReLU(0.1)
        )

        # ===== Decoder =====
        decoder_in = []
        decoder_out = []

        bottleneck_ch = n_layers * channels_interval

        for l in range(n_layers, 0, -1):
            enc_ch = l * channels_interval

            if l == n_layers:
                in_ch = bottleneck_ch + 2 * enc_ch
            else:
                prev_ch = (l + 1) * channels_interval
                in_ch = prev_ch + 2 * enc_ch

            decoder_in.append(in_ch)
            decoder_out.append(enc_ch)

        self.decoder = nn.ModuleList([
            UpSamplingLayer(decoder_in[i], decoder_out[i])
            for i in range(n_layers)
        ])

        # ===== Output =====
        self.out = nn.Sequential(
            nn.Conv1d(1 + channels_interval, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: [B, 2, T]
        """
        x1 = x[:, 0:1, :]
        x2 = x[:, 1:2, :]

        skips = []

        # ===== Encoder =====
        for i in range(self.n_layers):
            x1 = self.encoder1[i](x1)
            x2 = self.encoder2[i](x2)

            # attention
            a1, a2 = self.attn[i](x1, x2)

            # concat
            skip = torch.cat([a1, a2], dim=1)
            skips.append(skip)

            # downsample
            x1 = x1[:, :, ::2]
            x2 = x2[:, :, ::2]

        # ===== Bottleneck =====
        o = torch.cat([x1, x2], dim=1)
        o = self.middle(o)

        # ===== Decoder =====
        for i in range(self.n_layers):
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)

            skip = skips[self.n_layers - i - 1]

            # Align lengths
            if o.size(-1) != skip.size(-1):
                min_len = min(o.size(-1), skip.size(-1))
                o = o[..., :min_len]
                skip = skip[..., :min_len]

            o = torch.cat([o, skip], dim=1)
            o = self.decoder[i](o)

        # ===== Output =====
        # Use the first channel as reference (implied by the paper)
        ref = x[:, 0:1, :]
        if o.size(-1) != ref.size(-1):
            o = F.interpolate(o, size=ref.size(-1), mode="linear", align_corners=True)

        o = torch.cat([o, ref], dim=1)
        o = self.out(o)

        return o
    
if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    model = CrossChannelWaveUNet()

    x = torch.randn(4, 2, 16384)
    y = model(x)

    print(y.shape)  # [4, 1, 16384]

    macs, params = get_model_complexity_info(model,(2, 16384),print_per_layer_stat=False,as_strings=True)
    print(f"macs: {macs}, params: {params}")
    # macs: 4.43 GMac, params: 11.57 M