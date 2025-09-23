import torch
import torch.nn as nn
import torch.nn.functional as F
from adapter.ControlNet import ControlNet

class ControlNetAdapterWrapper(torch.nn.Module):
    """
    Nimmt Volumen [B,1,T,H,W] in [0,1] und gibt ein Bild [B,3,512,512] in [0,1] aus.
    Reduziert T vor dem 3D-Netz per AvgPool, um Speicher zu schonen.
    """
    def __init__(self, controlnet_cfg, in_channels=2, out_size=512, target_T=256, mid_channels=8, stride_T=3):
        super().__init__()
        self.out_size = out_size
        self.target_T = target_T
        self.in_channels = in_channels

        # Lernbarer T-Downsampler: erst Feature-Anhebung, dann stride in T
        # stride_T=3 macht aus 1400 -> ~467 (1400//3); das ist schon ~64% Speicherersparnis.
        # Du kannst stride_T=2 setzen, wenn du konservativer (mehr Info, mehr RAM) sein willst (-> 1400->700).
        self.downT = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(5,3,3), stride=(stride_T,1,1), padding=(2,1,1), bias=False),
            nn.SiLU(),
            nn.Conv3d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        num_downsamples = len(controlnet_cfg.get("num_channels", ())) - 1
        downsample_factor = 2 ** num_downsamples
        final_depth = target_T // downsample_factor

        print(f"\nAdapter konfiguriert: Eingangstiefe -> {stride_T}x Reduktion -> {target_T} -> ControlNet-Flaschenhals-Tiefe: {final_depth}")

        # Schritt 3: Aktualisiere die ControlNet-Konfiguration und instanziiere das Netz
        # Dies stellt sicher, dass das ControlNet immer korrekt gebaut wird.
        cfg = controlnet_cfg.copy()
        cfg['in_channels'] = self.in_channels
        cfg['conditioning_embedding_in_channels'] = self.in_channels
        cfg['final_depth'] = final_depth

        self.net = ControlNet(**cfg)

    def forward(self, vol):  # vol: [B,1,T,H,W]
        # 1) Lernbares Downsampling in T:
        vol = self.downT(vol)     # [B,1,T',H,W], T' ~= T/stride_T

        # (Optional) sanfte Angleichung auf ein einheitliches Ziel-T (z.B. 500)
        # Das ist linear (ohne Extra-Parameter) und sehr günstig:
        if self.target_T is not None and vol.shape[2] != self.target_T:
            vol = F.interpolate(vol, size=(self.target_T, vol.shape[-2], vol.shape[-1]),
                                mode="trilinear", align_corners=False)

        # 2) Stubs fürs 3D-ControlNet (ignoriert x/t/context intern)
        B, C, Tp, H, W = vol.shape
        x_stub = torch.zeros((B, C, Tp, H, W), device=vol.device, dtype=vol.dtype)
        t_stub = torch.zeros((B,), device=vol.device, dtype=torch.long)

        # 3) Dein ControlNet rechnet 3D->2D
        rgb = self.net(x_stub, t_stub, controlnet_cond=vol, conditioning_scale=1.0, context=None)  # [B,3,h,w]

        # 4) Final auf out_size bringen
        if (rgb.shape[-2], rgb.shape[-1]) != (self.out_size, self.out_size):
            rgb = F.interpolate(rgb, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        return torch.sigmoid(rgb)