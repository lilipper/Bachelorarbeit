import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class THzToRGBHead(nn.Module):
    def __init__(self,
                 in_ch: int = 2,
                 base_ch: int = 32,
                 k_t: int = 5,
                 final_depth: int = 16,
                 cap_T_in: int = 256,
                 t_stride1: int = 8,
                 t_stride2: int = 2,
                 gn_groups: int = 8):
        super().__init__()
        self.cap_T_in = int(cap_T_in)
        self.t_stride1 = int(t_stride1)
        self.t_stride2 = int(t_stride2)

        self.down1 = nn.Conv3d(
            in_ch, base_ch,
            kernel_size=(k_t, 1, 1),
            stride=(self.t_stride1, 1, 1),
            padding=(k_t // 2, 0, 0),
            bias=True,
        )
        self.down2 = nn.Conv3d(
            base_ch, base_ch,
            kernel_size=(k_t, 1, 1),
            stride=(self.t_stride2, 1, 1),
            padding=(k_t // 2, 0, 0),
            bias=True,
        )
        self.depth_mixer = nn.Conv3d(
            base_ch, base_ch,
            kernel_size=(5, 1, 1),
            padding=(2, 0, 0),
            groups=base_ch,
            bias=False,
        )
        self.pool_to_D = nn.AdaptiveAvgPool3d((final_depth, None, None))
        self.depth_projection = nn.Conv3d(
            base_ch, base_ch,
            kernel_size=(final_depth, 1, 1),
            padding=0,
            bias=True,
        )
        g = max(1, min(gn_groups, base_ch if base_ch % gn_groups == 0 else 1))
        self.refine2d = nn.Sequential(
            nn.GroupNorm(g, base_ch),
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch // 2, max(8, base_ch // 4), 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Conv2d(max(8, base_ch // 4), 3, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)

    def forward(self, x):
        B, C, T, H, W = x.shape
        if T > self.cap_T_in:
            k = math.ceil(T / self.cap_T_in)
            x = F.avg_pool3d(x, kernel_size=(k, 1, 1), stride=(k, 1, 1), ceil_mode=True)
            T = x.size(2)
        x = F.silu(self.down1(x))
        x = F.silu(self.down2(x))
        x = self.depth_mixer(x)
        x = self.pool_to_D(x)
        x = self.depth_projection(x).squeeze(2)
        x = self.refine2d(x)
        x = self.head(x)
        return x
