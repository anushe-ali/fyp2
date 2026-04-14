"""
model_loader.py
────────────────
Handles PyTorch's legacy zip format where the checkpoint is saved as a
FOLDER (model_english/, model_urdu/) containing:
  data.pkl  ← the actual serialised objects
  data/     ← tensor storage files (0, 1, 2 …)
  byteorder, version

We reconstruct the zip in memory and pass it to torch.load().
"""

import os
import io
import zipfile
import torch
import torch.nn as nn
from architectures import ECAPA_TDNN, IResNet, IBasicBlock


_AUDIO_KEYS  = ["audio_model", "audio", "model_a", "model_a_state_dict"]
_VISUAL_KEYS = ["visual_model", "visual", "model_v", "model_v_state_dict"]


def _find(d: dict, candidates: list):
    for k in candidates:
        if k in d:
            return k
    return None


def _load_from_folder(folder: str, device: torch.device):
    """
    Re-zip a PyTorch legacy folder checkpoint into a BytesIO buffer
    and load it with torch.load().
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(folder):
            # Sort so storage files (0,1,2…) are added in order
            dirs.sort()
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                # Archive name must be relative to parent of folder
                arcname = os.path.relpath(full_path, start=os.path.dirname(folder))
                zf.write(full_path, arcname)
    buf.seek(0)
    return torch.load(buf, map_location=device, weights_only=False)


def load_models(checkpoint_path: str, device: torch.device):
    """
    Load (model_a, model_v) from a checkpoint file OR legacy folder.
    """
    if os.path.isdir(checkpoint_path):
        ckpt = _load_from_folder(checkpoint_path, device)
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_a = ECAPA_TDNN(C=1024, embedding_size=512).to(device)
    model_v = IResNet(block=IBasicBlock, model='res18', num_features=512).to(device)

    if isinstance(ckpt, dict):
        a_key = _find(ckpt, _AUDIO_KEYS)
        v_key = _find(ckpt, _VISUAL_KEYS)

        if a_key and v_key:
            a_obj, v_obj = ckpt[a_key], ckpt[v_key]
            if isinstance(a_obj, nn.Module):
                model_a = a_obj.to(device)
            else:
                model_a.load_state_dict(a_obj)
            if isinstance(v_obj, nn.Module):
                model_v = v_obj.to(device)
            else:
                model_v.load_state_dict(v_obj)
        else:
            raise KeyError(
                f"Keys found in checkpoint: {list(ckpt.keys())}\n\n"
                f"Expected audio key from: {_AUDIO_KEYS}\n"
                f"Expected visual key from: {_VISUAL_KEYS}\n\n"
                "Add your actual key names to _AUDIO_KEYS / _VISUAL_KEYS in model_loader.py"
            )

    elif isinstance(ckpt, (list, tuple)) and len(ckpt) == 2:
        m0, m1 = ckpt
        model_a = (m0 if isinstance(m0, nn.Module) else model_a).to(device)
        model_v = (m1 if isinstance(m1, nn.Module) else model_v).to(device)

    elif isinstance(ckpt, nn.Module):
        for attr in ["audio_encoder", "audio_model", "model_a"]:
            if hasattr(ckpt, attr):
                model_a = getattr(ckpt, attr).to(device)
                break
        for attr in ["visual_encoder", "visual_model", "model_v", "face_encoder"]:
            if hasattr(ckpt, attr):
                model_v = getattr(ckpt, attr).to(device)
                break
    else:
        raise ValueError(f"Unrecognised checkpoint type: {type(ckpt)}")

    model_a.eval()
    model_v.eval()
    return model_a, model_v