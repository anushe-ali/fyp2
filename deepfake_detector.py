"""
deepfake_detector.py
─────────────────────
Face deepfake  : Xception fine-tuned on DFDC  (xception_finetuned_dfdc.pt)
Audio anti-spoof: 1D-CNN on raw waveform       (audio_deepfake.pt, optional)

Both detect_face_deepfake() and detect_audio_deepfake() always return a dict:
    {
        "is_fake":    bool,
        "confidence": float  0-1   (probability that input IS fake),
        "method":     str,
        "details":    str,
    }
"""

from __future__ import annotations
import os, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2, soundfile as sf
from PIL import Image
import torchvision.transforms as T

# ── Xception import ────────────────────────────────────────────────────────
try:
    from network.xception import xception
    _XCEPTION_AVAILABLE = True
except ImportError:
    _XCEPTION_AVAILABLE = False

# ── Weight paths ───────────────────────────────────────────────────────────
_FACE_WEIGHTS  = os.path.join("deepfake_weights", "xception_finetuned_dfdc.pt")
_AUDIO_WEIGHTS = os.path.join("deepfake_weights", "audio_deepfake.pt")

# ── Image transform (Xception expects 299×299, normalised to [-1,1]) ───────
_face_transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ── Model caches (loaded once per session) ─────────────────────────────────
_face_model_cache  = None
_audio_model_cache = None


# ═══════════════════════════════════════════════════════
# Architecture — audio anti-spoof 1D-CNN
# ═══════════════════════════════════════════════════════
class _AudioDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 64, 16), nn.ReLU(),
            nn.Conv1d(32, 64, 32,  8), nn.ReLU(),
            nn.Conv1d(64, 128, 16, 4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
    def forward(self, x):          # x: [B, T]
        return self.classifier(self.encoder(x.unsqueeze(1)))


# ═══════════════════════════════════════════════════════
# State-dict cleaner
# ═══════════════════════════════════════════════════════
def _clean_state_dict(sd: dict) -> dict:
    """Strip DataParallel / common wrapper prefixes."""
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("module.", "")
        k = k.replace("model.",  "")
        k = k.replace("net.",    "")
        cleaned[k] = v
    return cleaned


# ═══════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════
def _load_face_model(device: torch.device):
    global _face_model_cache
    if _face_model_cache is not None:
        return _face_model_cache
    if not _XCEPTION_AVAILABLE:
        return None
    if not os.path.exists(_FACE_WEIGHTS):
        return None

    # Build Xception with 2-class head (real / fake)
    model = xception(num_classes=1000, pretrained=False)
    model.last_linear = nn.Linear(model.last_linear.in_features, 2)
    model = model.to(device).eval()

    ckpt = torch.load(_FACE_WEIGHTS, map_location=device, weights_only=False)

    # unwrap checkpoint container if needed
    if isinstance(ckpt, dict):
        sd = (ckpt.get("state_dict")
              or ckpt.get("model_state_dict")
              or ckpt.get("model_state")
              or ckpt)
    else:
        sd = ckpt

    sd = _clean_state_dict(sd)

    # strict=False: tolerates any minor layer name differences
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[FaceDetector] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[FaceDetector] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    _face_model_cache = model
    return model


def _load_audio_model(device: torch.device):
    global _audio_model_cache
    if _audio_model_cache is not None:
        return _audio_model_cache
    if not os.path.exists(_AUDIO_WEIGHTS):
        return None

    model = _AudioDetector().to(device).eval()
    ckpt  = torch.load(_AUDIO_WEIGHTS, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        sd = (ckpt.get("state_dict")
              or ckpt.get("model_state_dict")
              or ckpt.get("model_state")
              or ckpt)
    else:
        sd = ckpt

    sd = _clean_state_dict(sd)
    model.load_state_dict(sd, strict=False)

    _audio_model_cache = model
    return model


# ═══════════════════════════════════════════════════════
# Public API — face
# ═══════════════════════════════════════════════════════
def detect_face_deepfake(
    img_bytes: bytes,
    device: torch.device = None,
    threshold: float = 0.5,
) -> dict:
    """
    Returns dict with keys: is_fake, confidence, method, details
    confidence = probability the image IS a deepfake (0→real, 1→fake)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Decode image ──────────────────────────────────────────────────────
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"is_fake": False, "confidence": 0.0,
                "method": "error", "details": "Could not decode image."}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # ── Neural model ──────────────────────────────────────────────────────
    model = _load_face_model(device)
    if model is not None:
        x = _face_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)                        # [1, 2]
            probs  = F.softmax(logits, dim=1)[0]
            # class 0 = real, class 1 = fake  (standard DFDC fine-tune convention)
            fake_prob = float(probs[1])

        return {
            "is_fake":    fake_prob > threshold,
            "confidence": fake_prob,
            "method":     "Xception (DFDC fine-tuned)",
            "details":    f"fake_prob={fake_prob:.4f}, threshold={threshold}",
        }

    # ── DCT frequency heuristic fallback ─────────────────────────────────
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray  = cv2.resize(gray, (256, 256))
    dct   = cv2.dct(gray)
    hf    = np.abs(dct[128:, 128:]).mean()
    lf    = np.abs(dct[:128, :128]).mean() + 1e-8
    ratio = hf / lf
    fake_prob = float(min(ratio / 0.15, 1.0) * 0.6)

    return {
        "is_fake":    fake_prob > threshold,
        "confidence": fake_prob,
        "method":     "frequency heuristic (no model weights found)",
        "details":    (f"DCT HF/LF ratio={ratio:.4f}. "
                       f"Place xception_finetuned_dfdc.pt in deepfake_weights/ for neural detection."),
    }


# ═══════════════════════════════════════════════════════
# Public API — audio
# ═══════════════════════════════════════════════════════
def detect_audio_deepfake(
    audio_bytes: bytes,
    suffix: str = ".wav",
    device: torch.device = None,
    threshold: float = 0.5,
) -> dict:
    """
    Returns dict with keys: is_fake, confidence, method, details
    confidence = probability the audio IS a spoof (0→real, 1→fake)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load waveform ─────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        wav, sr = sf.read(tmp, dtype="float32")
    finally:
        os.unlink(tmp)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    TARGET = 32000
    wav = wav[:TARGET] if len(wav) >= TARGET else np.pad(wav, (0, TARGET - len(wav)))

    # ── Neural model ──────────────────────────────────────────────────────
    model = _load_audio_model(device)
    if model is not None:
        x = torch.tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            probs     = F.softmax(model(x), dim=1)[0]
            fake_prob = float(probs[1])

        return {
            "is_fake":    fake_prob > threshold,
            "confidence": fake_prob,
            "method":     "1D-CNN audio detector",
            "details":    f"fake_prob={fake_prob:.4f}, threshold={threshold}",
        }

    # ── Statistical heuristic fallback ───────────────────────────────────
    frame_len = 512
    n_frames  = len(wav) // frame_len
    frames    = wav[:n_frames * frame_len].reshape(n_frames, frame_len)

    # ZCR variance — TTS tends to be unnaturally regular
    zcr      = np.array([np.mean(np.abs(np.diff(np.sign(f)))) / 2 for f in frames])
    zcr_var  = float(np.var(zcr))

    # Spectral flatness — TTS often sounds too clean
    spec      = np.abs(np.fft.rfft(wav))
    eps       = 1e-10
    flatness  = float(np.exp(np.mean(np.log(spec + eps))) / (np.mean(spec) + eps))

    # High zcr_var + high flatness → likely real; low → suspicious
    real_score = min(zcr_var / 0.01, 1.0) * 0.5 + min(flatness / 0.3, 1.0) * 0.5
    fake_prob  = float(max(0.0, min(1.0, 1.0 - real_score)))

    return {
        "is_fake":    fake_prob > threshold,
        "confidence": fake_prob,
        "method":     "statistical heuristic (no model weights found)",
        "details":    (f"ZCR_var={zcr_var:.5f}, spectral_flatness={flatness:.4f}. "
                       f"Place audio_deepfake.pt in deepfake_weights/ for neural detection."),
    }