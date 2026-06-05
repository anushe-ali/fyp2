"""
model_loader.py
────────────────
Supports two model families:

  Model 1 — ECAPA-TDNN + IResNet  (model_english, model_urdu)
    Checkpoint keys: "audio_model", "visual_model"
    Input: raw waveform + raw face image

  Model 2 — RFOP  (model_english2, model_urdu2)
    Checkpoint keys: "state_dict"  (with voice_branch.*, face_branch.* etc.)
    Input: pre-extracted face features (512-d) + voice features (512-d)
"""

import os
import io
import zipfile
import torch
import torch.nn as nn
from architectures import ECAPA_TDNN, IResNet, IBasicBlock
from architectures_rfop import RFOP, FACE_FEAT_DIM, VOICE_FEAT_DIM, N_CLASS, EMBED_DIM
from architecture_dino_hubert import FaceDINOEncoder, VoiceHuBERTEncoder

_AUDIO_KEYS  = ["audio_model", "audio", "model_a", "model_a_state_dict"]
_VISUAL_KEYS = ["visual_model", "visual", "model_v", "model_v_state_dict"]
_DINO_AUDIO_KEYS = ["audio_model"]
_DINO_FACE_KEYS  = ["face_model"]

def _find(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def _zip_folder_to_buffer(folder: str) -> io.BytesIO:
    buf = io.BytesIO()
    base = os.path.basename(folder.rstrip("/\\"))
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(folder):
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, start=folder)
                arcname = os.path.join(base, rel_path).replace("\\", "/")
                info = zipfile.ZipInfo(arcname)
                info.date_time = (2020, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_STORED
                with open(full_path, "rb") as f:
                    zf.writestr(info, f.read())
    buf.seek(0)
    return buf


def _find_torch_checkpoint_folder(path: str) -> str | None:
    if not os.path.isdir(path):
        return None

    entries = sorted(os.listdir(path))
    if len(entries) == 1:
        child = os.path.join(path, entries[0])
        if os.path.isdir(child):
            return _find_torch_checkpoint_folder(child)

    if {"data.pkl", ".format_version", "version"}.issubset(entries):
        return path

    for entry in entries:
        candidate = os.path.join(path, entry)
        if os.path.isfile(candidate) and entry.lower().endswith((".pt", ".pth")):
            return candidate
        if os.path.isdir(candidate):
            subentries = set(os.listdir(candidate))
            if {"data.pkl", ".format_version", "version"}.issubset(subentries):
                return candidate

    return None


def _load_checkpoint_path(checkpoint_path: str, device: torch.device):
    if os.path.isdir(checkpoint_path):
        archive_folder = _find_torch_checkpoint_folder(checkpoint_path)
        if archive_folder is None:
            raise ValueError(
                f"Checkpoint path '{checkpoint_path}' is a directory, but no loadable Torch archive was found. "
                "If this checkpoint was extracted from a .pt/.pth file, please use the original archive file instead. "
                "The supported directory format must contain 'data.pkl', '.format_version', and 'version'."
            )
        if os.path.isdir(archive_folder):
            buffer = _zip_folder_to_buffer(archive_folder)
            return torch.load(buffer, map_location=device, weights_only=False)
        return torch.load(archive_folder, map_location=device, weights_only=False)

    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _is_rfop_checkpoint(ckpt: dict) -> bool:
    """Detect RFOP checkpoint by presence of state_dict with RFOP layer names."""
    if not isinstance(ckpt, dict):
        return False
    if "state_dict" not in ckpt:
        return False
    sd = ckpt["state_dict"]
    rfop_keys = ["voice_branch", "face_branch", "fusion_layer", "res_mix", "logits_layer"]
    return any(any(k.startswith(rk) for k in sd) for rk in rfop_keys)


def _sanitize_batchnorm_stats(model: nn.Module) -> None:
    """Fix BatchNorm running stats that are NaN or invalid in loaded checkpoints."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            if torch.isnan(module.running_mean).any():
                module.running_mean[torch.isnan(module.running_mean)] = 0.0
            if torch.isnan(module.running_var).any():
                module.running_var[torch.isnan(module.running_var)] = 1.0
            invalid_var = module.running_var <= 0
            if invalid_var.any():
                module.running_var[invalid_var] = 1.0


def _is_dino_hubert_checkpoint(ckpt: dict) -> bool:
    """Detect DINO+HuBERT checkpoint by presence of face_model and audio_model keys."""
    if not isinstance(ckpt, dict):
        return False
    return "face_model" in ckpt and "audio_model" in ckpt

def load_models(checkpoint_path: str, device: torch.device):
    """
    Returns (model_a, model_v, model_type)

    model_type is either "ecapa_iresnet" or "rfop"
    """
    if os.path.isdir(checkpoint_path):
        ckpt = _load_checkpoint_path(checkpoint_path, device)
    else:
        ckpt = _load_checkpoint_path(checkpoint_path, device)

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

    if _is_dino_hubert_checkpoint(ckpt):
        model_v = FaceDINOEncoder(embed_dim=512, pretrained=False).to(device)
        model_a = VoiceHuBERTEncoder(embed_dim=512).to(device)

        model_v.load_state_dict(ckpt["face_model"])
        model_a.load_state_dict(ckpt["audio_model"])

        model_v.eval()
        model_a.eval()
        return model_a, model_v, "dino_hubert"
    
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
            _sanitize_batchnorm_stats(model_a)
            model_a.eval()
            model_v.eval()
            return model_a, model_v, "ecapa_iresnet"

        # raise KeyError(
        #     f"Keys found: {list(ckpt.keys())}\n"
        #     "Could not identify model type. "
        #     "Expected 'audio_model'/'visual_model' (Model 1) or "
        #     "'state_dict' with RFOP layers (Model 2)."
        # )
    
    
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