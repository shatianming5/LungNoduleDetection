#!/usr/bin/env python3
"""
Quick sanity check for the 3D ViT model in this repo.

Loads `model_config.json`, instantiates `VitDet3D`, optionally loads
`pretrained_model/pytorch_model.bin`, and runs a dummy forward pass.
All output messages are in English.
"""

import os
import torch
import numpy as np
from typing import Tuple

try:
    from transformers import ViTConfig
except Exception:
    ViTConfig = None

from model import VitDet3D


def _tuplize_image_size(v) -> Tuple[int, int, int]:
    """Return image size as (D, H, W) from config value.
    Accepts single int or sequence of length 3.
    Defaults to (40, 128, 128) if not available.
    """
    try:
        if isinstance(v, int):
            return (40, v, v)  # assume square slices if single int
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return tuple(int(x) for x in v)
    except Exception:
        pass
    return (40, 128, 128)


def test_model_loading() -> bool:
    print("=== Sanity: load config ===")
    if ViTConfig is None:
        print("transformers is not installed. Run: pip install transformers")
        return False

    try:
        config = ViTConfig.from_pretrained("model_config.json")
        print("Loaded model_config.json")
        print(f"  - image_size: {getattr(config, 'image_size', 'n/a')}")
        print(f"  - patch_size: {getattr(config, 'patch_size', 'n/a')}")
        print(f"  - hidden_size: {getattr(config, 'hidden_size', 'n/a')}")
        print(f"  - num_attention_heads: {getattr(config, 'num_attention_heads', 'n/a')}")
    except Exception as e:
        print(f"Failed to load ViTConfig: {e}")
        return False

    print("\n=== Instantiate model ===")
    try:
        model = VitDet3D(config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - parameters: {total_params:,}")
        print(f"  - trainable:  {trainable_params:,}")
    except Exception as e:
        print(f"Failed to build model: {e}")
        return False

    print("\n=== Load checkpoint (optional) ===")
    ckpt = os.path.join("pretrained_model", "pytorch_model.bin")
    if os.path.exists(ckpt):
        try:
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print("Loaded pretrained weights (strict=False)")
        except Exception as e:
            print(f"Warning: could not load weights: {e}")
    else:
        print("Checkpoint not found - skipping weight load.")

    print("\n=== Dummy forward ===")
    try:
        size = _tuplize_image_size(getattr(config, "image_size", None))
        channels = int(getattr(config, "num_channels", 1))
        batch = 2
        dummy_input = torch.randn(batch, channels, *size)
        dummy_labels = torch.zeros(batch)
        dummy_bbox = torch.randn(batch, 6)

        model.eval()
        with torch.no_grad():
            out = model(pixel_values=dummy_input, labels=dummy_labels, bbox=dummy_bbox)

        # Be flexible about output structure
        logits = out.get("logits") if isinstance(out, dict) else None
        bbox = out.get("bbox") if isinstance(out, dict) else None
        loss = out.get("loss") if isinstance(out, dict) else None
        if logits is not None:
            print(f"  - logits: {tuple(logits.shape)}")
        if bbox is not None:
            print(f"  - bbox:   {tuple(bbox.shape)}")
        if loss is not None:
            print(f"  - loss:   {float(loss):.4f}")
        print("Dummy forward OK.")
    except Exception as e:
        print(f"Failed dummy forward: {e}")
        return False

    print("\n=== Done ===")
    return True


if __name__ == "__main__":
    ok = test_model_loading()
    raise SystemExit(0 if ok else 1)
