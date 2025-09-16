from typing import Dict, Optional
import torch
from torch import nn

class TorchvisionInputAdapter(nn.Module):
    """
    Fusioniert RGB und optionale Zusatzbilder (z.B. 'feedback_img', 'thz_img') über 1x1-Conv auf 3 Kanäle.
    Erwartet img in [-1,1]; Zusatzbilder in [0,1] werden intern nach [-1,1] skaliert.
    """
    def __init__(self, expect_feedback: bool, expect_thz_image: bool):
        super().__init__()
        in_ch = 3 + (3 if expect_feedback else 0) + (3 if expect_thz_image else 0)
        self.proj = nn.Conv2d(in_ch, 3, kernel_size=1)

        self.use_fb = expect_feedback
        self.use_thz = expect_thz_image

    def forward(self, img: torch.Tensor, extra_cond: Optional[Dict] = None) -> torch.Tensor:
        # img: [B,3,H,W] in [-1,1]
        xs = [img]
        if extra_cond is not None:
            if self.use_fb and ("feedback_img" in extra_cond) and (extra_cond["feedback_img"] is not None):
                fb = extra_cond["feedback_img"]
                # fb kommt in [0,1] → nach [-1,1]
                fb = fb * 2.0 - 1.0
                xs.append(fb)
            if self.use_thz and ("thz_img" in extra_cond) and (extra_cond["thz_img"] is not None):
                tz = extra_cond["thz_img"]
                tz = tz * 2.0 - 1.0
                xs.append(tz)
        x = torch.cat(xs, dim=1)
        return self.proj(x)
