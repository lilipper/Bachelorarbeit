# controlnet_adapter_inject.py

import torch
import torch.nn.functional as F

class ControlNet2DInjectionSession:
    def __init__(self, unet, cn2d, conditioning_rgb, scale=1.0):
        self.unet = unet
        self.cn2d = cn2d
        self.scale = float(scale)

        p0 = next(unet.parameters())
        self.dev = p0.device
        self.dt  = p0.dtype

        # Conditioning auf UNet-Device/Typ bringen
        self.conditioning_rgb = conditioning_rgb.to(self.dev, dtype=self.dt)

        self.handles = []
        self.cond_pyr = None

    def __enter__(self):
        # Adapter auf UNet-Device/Typ
        self._prev_ckpt_flag = (
            getattr(self.unet, "is_gradient_checkpointing", False)
            or getattr(self.unet, "gradient_checkpointing", False)
        )
        try:
            self.unet.disable_gradient_checkpointing()
        except Exception:
            try:
                self.unet.set_gradient_checkpointing(False)
            except Exception:
                if hasattr(self.unet, "gradient_checkpointing"):
                    self.unet.gradient_checkpointing = False
        self.cn2d.to(self.dev, dtype=self.dt)

        # Pyramide bauen (im Graph!)
        self.cond_pyr = self.cn2d.cond_embed(self.conditioning_rgb, return_pyramid=True)
        self.cond_pyr = [c.to(self.dev, dtype=self.dt) for c in self.cond_pyr]

        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.cond_pyr = None
        if self._prev_ckpt_flag:
            try:
                self.unet.enable_gradient_checkpointing()
            except Exception:
                try:
                    self.unet.set_gradient_checkpointing(True)
                except Exception:
                    if hasattr(self.unet, "gradient_checkpointing"):
                        self.unet.gradient_checkpointing = True

    @staticmethod
    def _resize(feat, hw):
        if feat.shape[-2:] != hw:
            feat = F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)
        return feat

    def _inject(self, out, level, proj):
        cond = self._resize(self.cond_pyr[level], out.shape[-2:])
        res  = proj(cond)
        return out + self.scale * res.to(dtype=out.dtype, device=out.device)

    def _safe_downsamplers(self, db):
        """
        Liefere Liste der Downsampler-Module oder [].
        diffusers-Varianten:
          - db.downsamplers: ModuleList oder None
          - db.downsampler : einzelnes Modul oder None
        """
        if hasattr(db, "downsamplers") and db.downsamplers is not None:
            # kann ModuleList oder list sein
            return list(db.downsamplers)
        if hasattr(db, "downsampler") and db.downsampler is not None:
            return [db.downsampler]
        return []

    def _register_hooks(self):
        cb = self.cn2d.controlnet_down_blocks
        num_res = self.cn2d.num_res_blocks  # tuple pro Stufe

        # 0) nach conv_in
        def hook_conv_in(module, inp, out):
            return self._inject(out, 0, cb[0])
        self.handles.append(self.unet.conv_in.register_forward_hook(hook_conv_in))

        # Index in cb, wir haben cb[0] schon belegt
        idx = 1

        # Down-Stufen
        for i, db in enumerate(self.unet.down_blocks):
            # Resnet-Injektionen (num_res[i] ResNets)
            for r in range(num_res[i]):
                def make_res_hook(level=i, cb_idx=idx):
                    def _h(module, inp, out):
                        return self._inject(out, level, cb[cb_idx])
                    return _h
                self.handles.append(db.resnets[r].register_forward_hook(make_res_hook()))
                idx += 1

            # Downsampler (optional/robust)
            downs = self._safe_downsamplers(db)
            if len(downs) > 0:
                def make_down_hook(level=i+1, cb_idx=idx):
                    def _h(module, inp, out):
                        return self._inject(out, level, cb[cb_idx])
                    return _h
                self.handles.append(downs[0].register_forward_hook(make_down_hook()))
                idx += 1

        # Mid-Block (Name variiert je nach Version)
        mid = getattr(self.unet, "mid_block", None)
        if mid is None:
            mid = getattr(self.unet, "mid", None)
        if mid is not None:
            def hook_mid(module, inp, out):
                return self._inject(out, len(self.cond_pyr) - 1, self.cn2d.controlnet_mid_inj)
            self.handles.append(mid.register_forward_hook(hook_mid))
