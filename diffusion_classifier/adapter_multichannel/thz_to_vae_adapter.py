import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMultiChannelAdapter(nn.Module):
    def __init__(self, k_t=5, use_attn_pool=True, reduce_T_stride=1, hidden_channels=8):
        super().__init__()
        self.use_attn_pool = use_attn_pool
        self.reduce_T_stride = reduce_T_stride
        self.dw_t = nn.Conv3d(4, 4, kernel_size=(k_t,1,1), padding=(k_t//2,0,0), groups=4, bias=False)
        self.pw_mix = nn.Conv3d(4, 4, kernel_size=1)
        self.norm = nn.GroupNorm(4, 4)
        self.proj_t_down = None
        if reduce_T_stride and reduce_T_stride > 1:
            self.proj_t_down = nn.AvgPool3d(kernel_size=(reduce_T_stride,1,1), stride=(reduce_T_stride,1,1))
        if use_attn_pool:
            self.attn_mlp = nn.Sequential(
                nn.Conv3d(4, hidden_channels, 1, bias=True),
                nn.SiLU(),
                nn.Conv3d(hidden_channels, 1, 1, bias=True)
            )
        else:
            self.attn_mlp = None
        self.out_mix = nn.Conv2d(4, 4, kernel_size=1, bias=True)

    def forward(self, latents):
        x = latents.permute(0, 2, 1, 3, 4).contiguous()
        residual = x
        x = self.dw_t(x)
        x = self.pw_mix(x)
        x = self.norm(F.gelu(x) + residual)
        if self.proj_t_down is not None:
            x = self.proj_t_down(x)
        if self.use_attn_pool:
            logits = self.attn_mlp(x)
            weights = torch.softmax(logits, dim=2)
            x = (weights * x).sum(dim=2)
        else:
            x = F.adaptive_avg_pool3d(x, output_size=(1, None, None)).squeeze(2)
        x = self.out_mix(x)
        return x
