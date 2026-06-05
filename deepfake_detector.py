"""
deepfake_detector.py
─────────────────────
Face deepfake  : HuggingFace AutoModelForImageClassification
                 default: Wvolf/ViT_Deepfake_Detection         (id2label: {0:'Real', 1:'Fake'})
Audio deepfake : HuggingFace AutoModelForAudioClassification
                 default: mo-thecreator/Deepfake-audio-detection

Both detect_face_deepfake() and detect_audio_deepfake() always return:
    {
        "is_fake":    bool,
        "confidence": float  0-1   (probability that input IS fake),
        "method":     str,
        "details":    str,
    }
"""

from __future__ import annotations
import io, os, tempfile
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import soundfile as sf
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

# ── Model names (override via env vars if you want different checkpoints) ──
_FACE_MODEL_NAME  = os.environ.get(
    "FACE_DEEPFAKE_MODEL",
    "Wvolf/ViT_Deepfake_Detection",
)
_AUDIO_MODEL_NAME = os.environ.get(
    "AUDIO_DEEPFAKE_MODEL",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
)

# ── Caches (loaded once per session) ───────────────────────────────────────
_face_processor_cache  = None
_face_model_cache      = None
_audio_extractor_cache = None
_audio_model_cache     = None


# ═══════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════
def _load_face_model(device: torch.device):
    global _face_processor_cache, _face_model_cache
    if _face_model_cache is not None:
        return _face_processor_cache, _face_model_cache
    _face_processor_cache = AutoImageProcessor.from_pretrained(_FACE_MODEL_NAME)
    _face_model_cache = (
        AutoModelForImageClassification
        .from_pretrained(_FACE_MODEL_NAME)
        .to(device)
        .eval()
    )
    return _face_processor_cache, _face_model_cache


def _load_audio_model(device: torch.device):
    global _audio_extractor_cache, _audio_model_cache
    if _audio_model_cache is not None:
        return _audio_extractor_cache, _audio_model_cache
    _audio_extractor_cache = AutoFeatureExtractor.from_pretrained(_AUDIO_MODEL_NAME)
    _audio_model_cache = (
        AutoModelForAudioClassification
        .from_pretrained(_AUDIO_MODEL_NAME)
        .to(device)
        .eval()
    )
    return _audio_extractor_cache, _audio_model_cache


# ═══════════════════════════════════════════════════════
# Robust label resolution
# ═══════════════════════════════════════════════════════
_FAKE_LABELS = {
    "fake", "deepfake", "spoof", "spoofed", "synthetic",
    "ai", "ai-generated", "ai_generated", "generated",
    "manipulated", "tampered",
}
_REAL_LABELS = {
    "real", "realism", "bonafide", "bona-fide", "bona_fide",
    "genuine", "authentic", "human", "original",
}


def _fake_index(id2label: dict) -> int:
    """
    Return the index whose label means 'fake'.
    Strategy:
      1. Exact match (case-insensitive) against known fake keywords.
      2. Exact match against known real keywords -> return the *other* index.
      3. Substring fallback against fake keywords.
      4. Last resort: assume index 1 is fake (most common convention).
    """
    norm = {int(i): str(l).strip().lower() for i, l in id2label.items()}

    # 1. Exact fake match
    for i, label in norm.items():
        if label in _FAKE_LABELS:
            return i

    # 2. Exact real match -> pick the *other* class (binary case)
    if len(norm) == 2:
        for i, label in norm.items():
            if label in _REAL_LABELS:
                return [j for j in norm if j != i][0]

    # 3. Substring fallback
    for i, label in norm.items():
        if any(k in label for k in _FAKE_LABELS):
            return i

    # 4. Default
    return 1 if len(norm) > 1 else 0


# ═══════════════════════════════════════════════════════
# Public API — face
# ═══════════════════════════════════════════════════════
def detect_face_deepfake(
    img_bytes: bytes,
    device: torch.device = None,
    threshold: float = 0.5,
) -> dict:
    """
    confidence = P(image is deepfake), 0 -> real, 1 -> fake.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return {"is_fake": False, "confidence": 0.0,
                "method": "error", "details": f"Could not decode image: {e}"}

    try:
        processor, model = _load_face_model(device)
    except Exception as e:
        return _face_fallback(img_bytes, threshold, reason=str(e))

    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1)[0]

    id2label  = model.config.id2label
    fake_idx  = _fake_index(id2label)
    fake_prob = float(probs[fake_idx])
    all_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "is_fake":    fake_prob > threshold,
        "confidence": fake_prob,
        "method":     f"HF AutoModelForImageClassification ({_FACE_MODEL_NAME})",
        "details":    (f"fake_class='{id2label[fake_idx]}' (idx {fake_idx}), "
                       f"fake_prob={fake_prob:.4f}, threshold={threshold}, "
                       f"all_probs={all_probs}"),
    }


def _face_fallback(img_bytes: bytes, threshold: float, reason: str) -> dict:
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"is_fake": False, "confidence": 0.0,
                "method": "error", "details": "Could not decode image."}

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
        "method":     "frequency heuristic (HF model unavailable)",
        "details":    f"DCT HF/LF ratio={ratio:.4f}. HF load failed: {reason}",
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
    confidence = P(audio is spoof), 0 -> real, 1 -> fake.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        wav, sr = sf.read(tmp, dtype="float32")
    finally:
        os.unlink(tmp)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    try:
        extractor, model = _load_audio_model(device)
    except Exception as e:
        return _audio_fallback(wav, threshold, reason=str(e))

    target_sr = getattr(extractor, "sampling_rate", 16000)
    if sr != target_sr:
        new_len = int(round(len(wav) * target_sr / sr))
        wav = np.interp(
            np.linspace(0, len(wav) - 1, new_len, dtype=np.float32),
            np.arange(len(wav), dtype=np.float32),
            wav,
        ).astype(np.float32)
        sr = target_sr

    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1)[0]

    id2label  = model.config.id2label
    fake_idx  = _fake_index(id2label)
    fake_prob = float(probs[fake_idx])
    all_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "is_fake":    fake_prob > threshold,
        "confidence": fake_prob,
        "method":     f"HF AutoModelForAudioClassification ({_AUDIO_MODEL_NAME})",
        "details":    (f"fake_class='{id2label[fake_idx]}' (idx {fake_idx}), "
                       f"fake_prob={fake_prob:.4f}, threshold={threshold}, "
                       f"sr={sr}, all_probs={all_probs}"),
    }


def _audio_fallback(wav: np.ndarray, threshold: float, reason: str) -> dict:
    TARGET = 32000
    wav = wav[:TARGET] if len(wav) >= TARGET else np.pad(wav, (0, TARGET - len(wav)))

    frame_len = 512
    n_frames  = len(wav) // frame_len
    frames    = wav[:n_frames * frame_len].reshape(n_frames, frame_len)

    zcr      = np.array([np.mean(np.abs(np.diff(np.sign(f)))) / 2 for f in frames])
    zcr_var  = float(np.var(zcr))

    spec      = np.abs(np.fft.rfft(wav))
    eps       = 1e-10
    flatness  = float(np.exp(np.mean(np.log(spec + eps))) / (np.mean(spec) + eps))

    real_score = min(zcr_var / 0.01, 1.0) * 0.5 + min(flatness / 0.3, 1.0) * 0.5
    fake_prob  = float(max(0.0, min(1.0, 1.0 - real_score)))

    return {
        "is_fake":    fake_prob > threshold,
        "confidence": fake_prob,
        "method":     "statistical heuristic (HF model unavailable)",
        "details":    (f"ZCR_var={zcr_var:.5f}, spectral_flatness={flatness:.4f}. "
                       f"HF load failed: {reason}"),
    }