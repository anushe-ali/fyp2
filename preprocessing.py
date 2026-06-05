"""
preprocessing.py
────────────────
Matches the exact preprocessing from the training notebook:

Audio  (load_wav_eval):
  - soundfile.read → mono → pad/trim to max_len=32000 → FloatTensor [T]

Face   (face_eval_transform):
  - PIL open → Resize(112,112) → ToTensor()  [3, 112, 112] in [0,1]
  (matches the MAVTestDataset eval pipeline, NOT the BGR training loader)
"""

import numpy as np
import torch
import cv2
import soundfile as sf
from PIL import Image
import torchvision.transforms as transforms
import tempfile
import os


# ── Constants (match notebook) ────────────────────────────────────────────
AUDIO_MAX_LEN = 32000          # samples  (~2 s at 16 kHz)
FACE_SIZE     = 112

# Eval face transform (mirrors face_eval_transform in notebook Cell 36)
_face_transform = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),          # → [3,112,112] in [0,1]
])
# ─────────────────────────────────────────────────────────────────────────


def preprocess_face_from_bytes(img_bytes: bytes, face_size: int = FACE_SIZE) -> torch.Tensor:
    """
    Parameters
    ----------
    img_bytes : raw bytes from an uploaded image file
    face_size : target face size in pixels (H = W = face_size)

    Returns
    -------
    torch.Tensor  [3, face_size, face_size]  float32  in [0, 1]
    """
    # Decode via OpenCV (handles JPG, PNG, etc.)
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode face image.")

    # BGR → RGB then PIL (to use torchvision transform)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    transform = transforms.Compose([
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
    ])
    return transform(pil)


def preprocess_audio_from_bytes(audio_bytes: bytes, suffix: str = ".wav") -> torch.Tensor:
    """
    Parameters
    ----------
    audio_bytes : raw bytes from an uploaded audio file
    suffix      : file extension hint (e.g. '.wav', '.flac', '.mp3')

    Returns
    -------
    torch.Tensor  [T]  float32   T = AUDIO_MAX_LEN = 32000
    """
    # Write to a temp file so soundfile can read it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, _ = sf.read(tmp_path, dtype="float32")
    finally:
        os.unlink(tmp_path)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Trim or pad — exactly matches load_wav_eval in notebook
    if len(audio) < AUDIO_MAX_LEN:
        audio = np.pad(audio, (0, AUDIO_MAX_LEN - len(audio)))
    else:
        audio = audio[:AUDIO_MAX_LEN]

    return torch.FloatTensor(audio)      # [32000]
