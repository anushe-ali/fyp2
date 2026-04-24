"""
architectures_rfop.py
─────────────────────
RFOP model from rfop-fixed.ipynb.

Pipeline for inference:
  1. Face image  → FaceFeatureExtractor  → face_feat  [B, face_feat_dim]
  2. Audio file  → VoiceFeatureExtractor → voice_feat [B, voice_feat_dim]
  3. RFOP.forward(face_feat, voice_feat)
     → face_embed, voice_embed           [B, 256]
  4. score = L2_distance(face_embed, voice_embed)  lower = more similar

Feature extractors used at training time (from CSV):
  Face  : InceptionResnetV1 (facenet-pytorch) → 512-d
  Voice : resemblyzer VoiceEncoder            → 256-d

We replicate those here for live inference.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RFOP constants (must match training) ─────────────────────────────────────
EMBED_DIM      = 256
FACE_FEAT_DIM  = 512    # InceptionResnetV1 output
VOICE_FEAT_DIM = 256    # resemblyzer VoiceEncoder output
N_CLASS        = 64     # v1 dataset


# ═══════════════════════════════════════════════════════════════
#  RFOP CORE MODEL
# ═══════════════════════════════════════════════════════════════

def make_fc_1d(f_in: int, f_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(f_in, f_out),
        nn.BatchNorm1d(f_out),
        nn.ReLU(),
        nn.Linear(f_out, f_out),
    )


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim: int, embedding_dim: int):
        super().__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class FusionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)

    def forward(self, face: torch.Tensor, voice: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ff = F.gelu(face)
        vf = F.gelu(voice)
        w  = ff + vf
        x  = torch.cat([(face * w).unsqueeze(1),
                         (voice * w).unsqueeze(1)], dim=1)  # (B,2,dim)
        x  = self.conv(x).squeeze(1)                        # (B,dim)
        x  = F.gelu(x)
        return x, ff, vf


class RFOP(nn.Module):
    def __init__(
        self,
        face_feat_dim:  int = FACE_FEAT_DIM,
        voice_feat_dim: int = VOICE_FEAT_DIM,
        n_class:        int = N_CLASS,
        embed_dim:      int = EMBED_DIM,
        use_cuda:       bool = False,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.voice_branch = EmbedBranch(voice_feat_dim, embed_dim)
        self.face_branch  = EmbedBranch(face_feat_dim,  embed_dim)
        self.fusion_layer = FusionBlock(dim=embed_dim)
        self.res_mix      = nn.Linear(embed_dim, embed_dim)
        self.logits_layer = nn.Linear(embed_dim, n_class)

    def forward(self, faces: torch.Tensor, voices: torch.Tensor):
        voices_feats = self.voice_branch(voices)   # (B, embed_dim)
        faces_feats  = self.face_branch(faces)     # (B, embed_dim)

        comb, ff, vf = self.fusion_layer(faces_feats, voices_feats)
        comb_res     = F.normalize(self.res_mix(comb) + comb, p=2, dim=1)
        logits       = self.logits_layer(comb_res)

        return (
            [comb_res, logits],   # fused
            faces_feats,          # face branch embed
            voices_feats,         # voice branch embed
            [faces_feats],        # face_f  (for loss)
            [voices_feats],       # voice_f (for loss)
        )


# ═══════════════════════════════════════════════════════════════
#  FACE FEATURE EXTRACTOR  (InceptionResnetV1 via facenet-pytorch)
# ═══════════════════════════════════════════════════════════════

class FaceFeatureExtractor:
    """
    Extracts 512-d face embeddings using InceptionResnetV1.
    Install: pip install facenet-pytorch
    """
    def __init__(self, device: torch.device):
        try:
            from facenet_pytorch import InceptionResnetV1
            import torchvision.transforms as T
        except ImportError:
            raise ImportError(
                "facenet-pytorch is required for model_english2/model_urdu2.\n"
                "Run: pip install facenet-pytorch"
            )
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def extract(self, pil_image) -> np.ndarray:
        """PIL RGB image → numpy [512]"""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        return emb.cpu().numpy().squeeze(0)   # [512]


# ═══════════════════════════════════════════════════════════════
#  VOICE FEATURE EXTRACTOR  (resemblyzer)
# ═══════════════════════════════════════════════════════════════

class VoiceFeatureExtractor:
    """
    Extracts 256-d voice embeddings using resemblyzer.
    Install: pip install resemblyzer
    """
    def __init__(self):
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            from pathlib import Path
        except ImportError:
            raise ImportError(
                "resemblyzer is required for model_english2/model_urdu2.\n"
                "Run: pip install resemblyzer"
            )
        self.encoder        = VoiceEncoder()
        self.preprocess_wav = preprocess_wav

    def extract(self, audio_path: str) -> np.ndarray:
        """Audio file path → numpy [256]"""
        wav = self.preprocess_wav(audio_path)
        emb = self.encoder.embed_utterance(wav)
        return emb   # [256]
