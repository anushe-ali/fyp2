"""
model_loader.py
────────────────
Supports two model families:

  Model 1 — ECAPA-TDNN + IResNet  (model_english, model_urdu)
    Checkpoint keys: "audio_model", "visual_model"
    Input: raw waveform + raw face image

  Model 2 — RFOP  (model_english2, model_urdu2)
    Checkpoint keys: "state_dict"  (with voice_branch.*, face_branch.* etc.)
    Input: pre-extracted face features (512-d) + voice features (256-d)
"""

import os
import io
import zipfile
import torch
import torch.nn as nn
from architectures import ECAPA_TDNN, IResNet, IBasicBlock
from architectures_rfop import RFOP, FACE_FEAT_DIM, VOICE_FEAT_DIM, N_CLASS, EMBED_DIM

_AUDIO_KEYS  = ["audio_model", "audio", "model_a", "model_a_state_dict"]
_VISUAL_KEYS = ["visual_model", "visual", "model_v", "model_v_state_dict"]


def _find(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def _load_from_folder(folder: str, device: torch.device):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, start=os.path.dirname(folder))
                zf.write(full_path, arcname)
    buf.seek(0)
    return torch.load(buf, map_location=device, weights_only=False)


def _is_rfop_checkpoint(ckpt: dict) -> bool:
    """Detect RFOP checkpoint by presence of state_dict with RFOP layer names."""
    if not isinstance(ckpt, dict):
        return False
    if "state_dict" not in ckpt:
        return False
    sd = ckpt["state_dict"]
    rfop_keys = ["voice_branch", "face_branch", "fusion_layer", "res_mix", "logits_layer"]
    return any(any(k.startswith(rk) for k in sd) for rk in rfop_keys)


def load_models(checkpoint_path: str, device: torch.device):
    """
    Returns (model_a, model_v, model_type)

    model_type is either "ecapa_iresnet" or "rfop"
    """
    if os.path.isdir(checkpoint_path):
        ckpt = _load_from_folder(checkpoint_path, device)
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── RFOP checkpoint ───────────────────────────────────────────────────────
    if _is_rfop_checkpoint(ckpt):
        model = RFOP(
            face_feat_dim  = FACE_FEAT_DIM,
            voice_feat_dim = VOICE_FEAT_DIM,
            n_class        = N_CLASS,
            embed_dim      = EMBED_DIM,
            use_cuda       = False,
        ).to(device)
        sd = ckpt["state_dict"]
        model.load_state_dict(sd)
        model.eval()
        return model, None, "rfop"

    # ── ECAPA-TDNN + IResNet checkpoint ───────────────────────────────────────
    if isinstance(ckpt, dict):
        model_a = ECAPA_TDNN(C=1024, embedding_size=512).to(device)
        model_v = IResNet(block=IBasicBlock, model='res18', num_features=512).to(device)

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
            model_a.eval()
            model_v.eval()
            return model_a, model_v, "ecapa_iresnet"

        raise KeyError(
            f"Keys found: {list(ckpt.keys())}\n"
            "Could not identify model type. "
            "Expected 'audio_model'/'visual_model' (Model 1) or "
            "'state_dict' with RFOP layers (Model 2)."
        )

    elif isinstance(ckpt, (list, tuple)) and len(ckpt) == 2:
        model_a = ECAPA_TDNN(C=1024, embedding_size=512).to(device)
        model_v = IResNet(block=IBasicBlock, model='res18', num_features=512).to(device)
        m0, m1 = ckpt
        model_a = (m0 if isinstance(m0, nn.Module) else model_a).to(device)
        model_v = (m1 if isinstance(m1, nn.Module) else model_v).to(device)
        model_a.eval()
        model_v.eval()
        return model_a, model_v, "ecapa_iresnet"

    raise ValueError(f"Unrecognised checkpoint type: {type(ckpt)}")