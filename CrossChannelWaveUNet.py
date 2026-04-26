import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSigmoid(nn.Module):
    def __init__(self, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        return torch.sigmoid(self.alpha * (x - self.beta))


class CrossChannelAttention(nn.Module):
    """
    Input:
        x1, x2: [B, C, T]
    Output:
        a1, a2: [B, C, T]
    """
    def __init__(self, channels):
        super().__init__()
        self.proj1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.proj2 = nn.Conv1d(channels, channels, kernel_size=1)

        self.learnable_sigmoid = LearnableSigmoid()

        self.mask_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x1, x2):
        h1 = torch.tanh(self.proj1(x1))
        h2 = torch.tanh(self.proj2(x2))

        z = torch.abs(h1 * h2)
        z = self.learnable_sigmoid(z)

        mask = torch.sigmoid(self.mask_proj(z))

        a1 = x1 + mask * x1
        a2 = x2 + mask * x2

        return a1, a2


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act(self.conv(x))
        skip = x
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2

        self.skip_fuse = nn.Conv1d(skip_ch, out_ch, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch + out_ch, out_ch, kernel_size, padding=pad),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)

        if x.size(-1) != skip.size(-1):
            min_len = min(x.size(-1), skip.size(-1))
            x = x[..., :min_len]
            skip = skip[..., :min_len]

        skip = self.skip_fuse(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ChannelEncoder(nn.Module):
    def __init__(self, num_layers=10, base_ch=24):
        super().__init__()

        blocks = []
        in_ch = 1

        for l in range(1, num_layers + 1):
            out_ch = base_ch * l
            blocks.append(DownBlock(in_ch, out_ch, kernel_size=15))
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)

    def forward_one_layer(self, x, idx):
        return self.blocks[idx](x)


class CrossChannelWaveUNet(nn.Module):
    """
    Proposed model approximation from Ho et al. 2020.

    Input:
        noisy: [B, 2, T]
    Output:
        enhanced: [B, 1, T]
    """
    def __init__(
        self,
        input_len=16384,
        num_layers=10,
        base_ch=24,
        bottleneck_ch=264,
    ):
        super().__init__()

        self.input_len = input_len
        self.num_layers = num_layers
        self.base_ch = base_ch

        self.encoder1 = ChannelEncoder(num_layers, base_ch)
        self.encoder2 = ChannelEncoder(num_layers, base_ch)

        self.attn_blocks = nn.ModuleList([
            CrossChannelAttention(base_ch * l)
            for l in range(1, num_layers + 1)
        ])

        last_ch = base_ch * num_layers

        self.bottleneck = nn.Sequential(
            nn.Conv1d(last_ch * 2, bottleneck_ch, kernel_size=15, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )

        up_blocks = []
        in_ch = bottleneck_ch

        for l in range(num_layers, 0, -1):
            skip_ch = base_ch * l * 2
            out_ch = base_ch * l
            up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, kernel_size=5))
            in_ch = out_ch

        self.decoder = nn.ModuleList(up_blocks)

        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        assert x.dim() == 3, "Input must be [B, 2, T]"
        assert x.size(1) == 2, "This implementation expects 2-channel input"

        x1 = x[:, 0:1, :]
        x2 = x[:, 1:2, :]

        skips = []

        for i in range(self.num_layers):
            x1, s1 = self.encoder1.forward_one_layer(x1, i)
            x2, s2 = self.encoder2.forward_one_layer(x2, i)

            a1, a2 = self.attn_blocks[i](s1, s2)

            # cross-concat: A1 concat to X2, A2 concat to X1
            skip = torch.cat([torch.cat([s1, a2], dim=1),
                              torch.cat([s2, a1], dim=1)], dim=1)

            # 上面会变成 4C，参数量偏大。
            # 更贴近论文 decoder skip 的做法是只保留双路 encoder 特征：
            skip = torch.cat([a1, a2], dim=1)

            skips.append(skip)

        z = torch.cat([x1, x2], dim=1)
        z = self.bottleneck(z)

        for up, skip in zip(self.decoder, reversed(skips)):
            z = up(z, skip)

        y = self.out_conv(z)

        if y.size(-1) != x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)

        return y


def negative_sisdr(est, ref, eps=1e-8):
    """
    est/ref: [B, 1, T]
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref
    proj = proj / (torch.sum(ref ** 2, dim=-1, keepdim=True) + eps)

    noise = est - proj

    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    sisdr = 10 * torch.log10(ratio + eps)

    return -sisdr.mean()


def wsdr_loss(est, clean, noisy_ref, eps=1e-8):
    """
    Weighted SDR loss approximation used in speech enhancement papers.

    est:       [B, 1, T]
    clean:     [B, 1, T]
    noisy_ref: [B, 1, T], usually mic-1 noisy signal
    """
    noise = noisy_ref - clean
    est_noise = noisy_ref - est

    clean_energy = torch.sum(clean ** 2, dim=-1)
    noise_energy = torch.sum(noise ** 2, dim=-1)

    alpha = clean_energy / (clean_energy + noise_energy + eps)
    alpha = alpha.mean()

    return alpha * negative_sisdr(est, clean) + (1 - alpha) * negative_sisdr(est_noise, noise)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    model = CrossChannelWaveUNet()

    x = torch.randn(4, 2, 16384)
    y = model(x)

    print(y.shape)  # [4, 1, 16384]

    macs, params = get_model_complexity_info(model,(2, 16384),print_per_layer_stat=False,as_strings=True)
    print(f"macs: {macs}, params: {params}")
    # macs: 4.1 GMac, params: 11.1 M



    clean = torch.randn(4, 1, 16384)
    noisy_ref = x[:, 0:1, :]

    loss = wsdr_loss(y, clean, noisy_ref)
    loss.backward()

    print("loss:", loss.item())