# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
from torch import nn

from generative.networks.nets.diffusion_model_unet import get_down_block, get_mid_block, get_timestep_embedding


class ControlNetConditioningEmbedding2D(nn.Module):
    """
    Input : [B, C_in (z.B. 3), H, W]
    Output: Liste 2D-Featuremaps je Stufe: [[B,C_i,H_i,W_i] ...]
    num_channels sollte an den SD-UNet (block_out_channels[:4]) angelehnt sein.
    """
    def __init__(self, in_channels: int = 3,
                 num_channels: Sequence[int] = (320, 640, 1280, 1280)):  # <<< CHANGE HERE bei SD1.x
        super().__init__()
        self.num_channels = list(num_channels)
        self.conv_in = Convolution(
            spatial_dims=2, in_channels=in_channels, out_channels=num_channels[0],
            strides=1, kernel_size=3, padding=1, conv_only=True
        )
        self.blocks = nn.ModuleList([])
        for i in range(len(num_channels) - 1):
            cin, cout = num_channels[i], num_channels[i + 1]
            self.blocks.append(Convolution(2, cin, cin, strides=1, kernel_size=3, padding=1, conv_only=True))
            self.blocks.append(Convolution(2, cin, cout, strides=2, kernel_size=3, padding=1, conv_only=True))

    def forward(self, conditioning: torch.Tensor, return_pyramid: bool = True):
        h = F.silu(self.conv_in(conditioning))
        feats = [h]
        for i in range(0, len(self.blocks), 2):
            h = F.silu(self.blocks[i](h))
            h = F.silu(self.blocks[i + 1](h))
            feats.append(h)
        return feats


def zero_module_(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNet2DAdapter(nn.Module):
    """
    2D-Adapter:
      - cond_embed: 2D-Pyramide aus RGB-Condition (Bild vom THz-Frontmodul)
      - 2D-Injektion (1x1) für conv_in, alle ResBlöcke, Downsample, Mid
      - kein eigener UNet-forward
    """
    def __init__(self,
                 num_res_blocks: Sequence[int] | int = (2,2,2,2),         
                 num_channels: Sequence[int] = (320,640,1280,1280),      
                 cond_in_channels: int = 3,
                 zero_init_injectors: bool = True):
        super().__init__()
        if isinstance(num_res_blocks, int):
            num_res_blocks = list(ensure_tuple_rep(num_res_blocks, len(num_channels)))
        self.num_res_blocks = list(num_res_blocks)
        self.block_out_channels = list(num_channels)

        self.cond_embed = ControlNetConditioningEmbedding2D(
            in_channels=cond_in_channels, num_channels=num_channels
        )

        self.controlnet_down_blocks = nn.ModuleList()
        # (0) conv_in
        self.controlnet_down_blocks.append(
            Convolution(2, num_channels[0], num_channels[0], strides=1, kernel_size=1, padding=0, conv_only=True)
        )
        for i, C in enumerate(num_channels):
            for _ in range(self.num_res_blocks[i]):
                self.controlnet_down_blocks.append(
                    Convolution(2, C, C, strides=1, kernel_size=1, padding=0, conv_only=True)
                )
            if i < len(num_channels) - 1:
                self.controlnet_down_blocks.append(
                    Convolution(2, num_channels[i + 1], C, strides=1, kernel_size=1, padding=0, conv_only=True)
                )
        mid_C = num_channels[-1]
        self.controlnet_mid_inj = Convolution(2, mid_C, mid_C, strides=1, kernel_size=1, padding=0, conv_only=True)

        if zero_init_injectors:
            for m in self.controlnet_down_blocks: zero_module_(m)
            zero_module_(self.controlnet_mid_inj)

    def forward(self, *args, **kwargs):
        pass