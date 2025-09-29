# thz_front_rgb_head.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class THzToRGBHead(nn.Module):
    """
    Lernbare 3D->2D-Projektion für das VAE-Eingangbild.
    Input : [B, 2, T, H, W]   (T kann sehr groß sein, z.B. 1400)
    Output: [B, 3, H, W]      (in [0,1])
    Strategie:
      1) optionales dynamisches AvgPool auf T, um T <= cap_T_in zu begrenzen
      2) zwei lernende 1D-(T)-Strided Conv3d zur starken Reduktion von T
      3) depthwise Mixer in T
      4) AdaptiveAvgPool3d auf final_depth (z.B. 16)
      5) 3D->2D Projektion + 2D-Refine + RGB-Head
    """
    def __init__(self,
                 in_ch: int = 2,
                 base_ch: int = 32,          # kleiner halten als 64 für RAM
                 k_t: int = 5,
                 final_depth: int = 16,
                 cap_T_in: int = 256,        # Obergrenze für T vor allem Lernenden
                 t_stride1: int = 8,         # erste lernende Reduktion in T
                 t_stride2: int = 2,         # zweite lernende Reduktion in T
                 gn_groups: int = 8):
        super().__init__()
        self.cap_T_in = int(cap_T_in)
        self.t_stride1 = int(t_stride1)
        self.t_stride2 = int(t_stride2)

        # 1) lernende T-Reduktion (nur entlang T striden)
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

        # 2) depthwise Mischer entlang T (kleines k_t, gruppiert)
        self.depth_mixer = nn.Conv3d(
            base_ch, base_ch,
            kernel_size=(5, 1, 1),
            padding=(2, 0, 0),
            groups=base_ch,
            bias=False,
        )

        # 3) auf gewünschte Endtiefe gehen (nicht lernend, robust)
        self.pool_to_D = nn.AdaptiveAvgPool3d((final_depth, None, None))

        # 4) 3D -> 2D Projektion
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

    def forward(self, x):  # x: [B,2,T,H,W]
        B, C, T, H, W = x.shape

        # 0) Sicherheits-Pooling, falls T zu groß (nicht lernend, aber wichtig für RAM)
        if T > self.cap_T_in:
            k = math.ceil(T / self.cap_T_in)
            x = F.avg_pool3d(x, kernel_size=(k, 1, 1), stride=(k, 1, 1), ceil_mode=True)
            T = x.size(2)

        # 1) lernende T-Downsamplings
        x = F.silu(self.down1(x))   # [B, base_ch, T1, H, W]
        x = F.silu(self.down2(x))   # [B, base_ch, T2, H, W]

        # 2) Mixer
        x = self.depth_mixer(x)     # [B, base_ch, T2, H, W]

        # 3) auf final_depth
        x = self.pool_to_D(x)       # [B, base_ch, D, H, W]

        # 4) 3D->2D
        x = self.depth_projection(x).squeeze(2)  # [B, base_ch, H, W]

        # 5) 2D-Refine + RGB
        x = self.refine2d(x)        # [B, base_ch/4, H, W]
        x = self.head(x)            # [B, 3, H, W]
        return x.sigmoid()
