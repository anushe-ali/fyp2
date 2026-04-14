import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os

from model_loader import load_models
from preprocessing import preprocess_face_from_bytes, preprocess_audio_from_bytes

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AV Identity Verifier",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0a0f; color: #e8e8f0; }
.main, .stApp { background: #0a0a0f; }
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
    letter-spacing: -2px; line-height: 1.1;
    background: linear-gradient(135deg, #c8f5ff 0%, #7eb8f7 50%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #5a5a7a;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2rem;
}
.card {
    background: #12121f; border: 1px solid #1e1e35; border-radius: 16px;
    padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 3px;
    text-transform: uppercase; color: #5a5a7a; margin-bottom: 0.8rem;
}
.result-match {
    background: linear-gradient(135deg, #0d2e1f, #0a1a14);
    border: 1px solid #1a5c3a; border-radius: 16px; padding: 2rem; text-align: center;
}
.result-nomatch {
    background: linear-gradient(135deg, #2e0d0d, #1a0a0a);
    border: 1px solid #5c1a1a; border-radius: 16px; padding: 2rem; text-align: center;
}
.result-label-match  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#4ade80; letter-spacing:-1px; }
.result-label-nomatch{ font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#f87171; letter-spacing:-1px; }
.score-display       { font-family:'JetBrains Mono',monospace; font-size:1rem; color:#8888aa; margin-top:0.5rem; }
.confidence-bar-bg   { background:#1e1e35; border-radius:8px; height:8px; margin-top:1rem; overflow:hidden; }
.info-row            { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#5a5a7a; margin-top:0.4rem; }

.stSelectbox label, .stFileUploader label {
    font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important;
    letter-spacing:2px !important; text-transform:uppercase !important; color:#5a5a7a !important;
}
.stButton > button {
    font-family:'Syne',sans-serif !important; font-weight:600 !important; font-size:1rem !important;
    background:linear-gradient(135deg,#3b5bdb,#7048e8) !important; color:white !important;
    border:none !important; border-radius:12px !important; padding:0.75rem 2rem !important;
    width:100% !important;
}
.stButton > button:hover { opacity:0.88 !important; }
div[data-testid="stFileUploadDropzone"] {
    background:#0e0e1c !important; border:1px dashed #2a2a50 !important; border-radius:12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">AV Identity<br>Verifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ECAPA-TDNN · IResNet18 · Audio–Visual Fusion</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — model paths
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Paths")
    english_model_path = st.text_input("English model", value="model_english")
    urdu_model_path    = st.text_input("Urdu model",    value="model_urdu")    
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("""
**Audio**: ECAPA-TDNN  
• C=1024, emb=512  
• Input: raw waveform [T=32000]

**Visual**: IResNet-18  
• num_features=512  
• Input: RGB [3×112×112]

**Scoring**: MSE distance  
in normalised embedding space  
(lower = more similar)
    """)

# ─────────────────────────────────────────────
# Language selector
# ─────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">01 — Select Language</div>', unsafe_allow_html=True)
language = st.radio("Language", options=["English", "Urdu"], horizontal=True, label_visibility="collapsed")
chosen_model_path = english_model_path if language == "English" else urdu_model_path
st.markdown(f'<div class="info-row">Model → {chosen_model_path}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Upload inputs
# ─────────────────────────────────────────────
from PIL import Image

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">02 — Face Image</div>', unsafe_allow_html=True)
    face_file = st.file_uploader("Upload face", type=["jpg","jpeg","png"],
                                  label_visibility="collapsed", key="face")
    if face_file:
        st.image(Image.open(face_file).convert("RGB"), use_container_width=True)
    st.markdown('<div class="info-row">JPG / PNG · resized to 112×112 internally</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">03 — Voice Clip</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload audio", type=["wav","flac","mp3"],
                                   label_visibility="collapsed", key="audio")
    if audio_file:
        st.audio(audio_file)
    st.markdown('<div class="info-row">WAV / FLAC / MP3 · first 2 s used (32 000 samples)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Threshold
# ─────────────────────────────────────────────
with st.expander("⚙️ Advanced: Decision Threshold"):
    threshold = st.slider(
        "MSE Distance Threshold",
        min_value=0.05, max_value=5.0, value=0.5, step=0.05,
        help="Pairs with MSE distance BELOW this value → MATCH. "
             "Tune based on your model's EER operating point."
    )
    st.markdown(
        '<div class="info-row">Embeddings are L2-normalised, so MSE ∈ [0, 4]. '
        'EER threshold is typically 0.3–0.8 for well-trained models.</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("🔍  Verify Identity")

if run_btn:
    if not face_file or not audio_file:
        st.warning("Please upload both a face image and a voice clip before verifying.")
    elif not os.path.exists(chosen_model_path):
        st.error(
            f"Model file **{chosen_model_path}** not found in the app directory. "
            "Place it alongside app.py, or update the path in the sidebar."
        )
    else:
        with st.spinner("Loading model & extracting embeddings…"):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # ── Load models ───────────────────────────────────────────
                model_a, model_v = load_models(chosen_model_path, device)

                # ── Preprocess ────────────────────────────────────────────
                face_file.seek(0)
                audio_file.seek(0)
                face_bytes  = face_file.read()
                audio_bytes = audio_file.read()
                audio_suffix = "." + audio_file.name.rsplit(".", 1)[-1].lower()

                face_tensor   = preprocess_face_from_bytes(face_bytes).unsqueeze(0).to(device)
                speech_tensor = preprocess_audio_from_bytes(audio_bytes, suffix=audio_suffix).unsqueeze(0).to(device)

                # ── Inference ─────────────────────────────────────────────
                with torch.no_grad():
                    a_emb = F.normalize(model_a(speech_tensor, aug=False), dim=1)
                    v_emb = F.normalize(model_v(face_tensor), dim=1)

                mse_score = F.mse_loss(a_emb, v_emb, reduction="sum").item()
                is_match  = mse_score < threshold

                # ── Confidence (logistic around threshold) ────────────────
                confidence = float(1 / (1 + np.exp(5 * (mse_score - threshold)))) * 100

                # ── Result display ────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                if is_match:
                    st.markdown(f"""
                    <div class="result-match">
                        <div class="result-label-match">✓ &nbsp;IDENTITY VERIFIED</div>
                        <div class="score-display">
                            MSE: {mse_score:.4f} &nbsp;·&nbsp; Threshold: {threshold:.2f}<br>
                            Confidence: {confidence:.1f}%
                        </div>
                        <div class="confidence-bar-bg">
                            <div style="width:{min(confidence,100):.1f}%;height:8px;
                                background:linear-gradient(90deg,#4ade80,#22d3ee);border-radius:8px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-nomatch">
                        <div class="result-label-nomatch">✗ &nbsp;IDENTITY MISMATCH</div>
                        <div class="score-display">
                            MSE: {mse_score:.4f} &nbsp;·&nbsp; Threshold: {threshold:.2f}<br>
                            Mismatch confidence: {100 - confidence:.1f}%
                        </div>
                        <div class="confidence-bar-bg">
                            <div style="width:{min(100-confidence,100):.1f}%;height:8px;
                                background:linear-gradient(90deg,#f87171,#fb923c);border-radius:8px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # ── Debug ─────────────────────────────────────────────────
                with st.expander("🔬 Embedding Debug Info"):
                    st.code(
                        f"Language      : {language}\n"
                        f"Model file    : {chosen_model_path}\n"
                        f"Device        : {device}\n"
                        f"Audio tensor  : {tuple(speech_tensor.shape)}  (should be [1, 32000])\n"
                        f"Face tensor   : {tuple(face_tensor.shape)}   (should be [1, 3, 112, 112])\n"
                        f"Audio emb dim : {tuple(a_emb.shape)}  (should be [1, 512])\n"
                        f"Face  emb dim : {tuple(v_emb.shape)}  (should be [1, 512])\n"
                        f"MSE distance  : {mse_score:.6f}\n"
                        f"Threshold     : {threshold}\n"
                        f"Decision      : {'MATCH' if is_match else 'NO MATCH'}"
                    )

            except KeyError as e:
                st.error(f"Checkpoint key error: {e}")
            except RuntimeError as e:
                st.error(
                    f"Model load error: {e}\n\n"
                    "This usually means the architecture in architectures.py doesn't "
                    "exactly match what was trained. Check C= and embedding_size= in ECAPA_TDNN."
                )
            except Exception as e:
                st.error(f"Unexpected error: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────
# Checkpoint Inspector (bottom of page)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Checkpoint Inspector")
st.markdown(
    '<div class="info-row">Not sure what keys your .pt folder contains? '
    'Run this to find out.</div>', unsafe_allow_html=True
)

inspect_path = st.text_input(
    "Path to inspect (file or folder)",
    value="epoch_final35",
    key="inspect_path"
)

if st.button("Inspect checkpoint keys"):
    if not os.path.exists(inspect_path):
        st.error(f"Path not found: {inspect_path}")
    else:
        try:
            from model_loader import _load_from_folder
            if os.path.isdir(inspect_path):
                ckpt = _load_from_folder(inspect_path, torch.device("cpu"))
            else:
                ckpt = torch.load(inspect_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                st.success("Checkpoint is a **dict**. Top-level keys:")
                for k, v in ckpt.items():
                    vtype = type(v).__name__
                    extra = ""
                    if hasattr(v, "keys"):
                        extra = f"  ({len(v)} sub-keys)"
                    st.code(f"  '{k}': {vtype}{extra}")
                st.info(
                    "👉 Copy the audio key into `_AUDIO_KEYS` and visual key "
                    "into `_VISUAL_KEYS` in model_loader.py if they're not already there."
                )
            elif isinstance(ckpt, (list, tuple)):
                st.success(f"Checkpoint is a **{type(ckpt).__name__}** of length {len(ckpt)}.")
                for i, item in enumerate(ckpt):
                    st.code(f"  [{i}]: {type(item).__name__}")
            elif hasattr(ckpt, "__class__"):
                st.success(f"Checkpoint is a **{type(ckpt).__name__}** (nn.Module or similar).")
                if hasattr(ckpt, "__dict__"):
                    attrs = [k for k in vars(ckpt) if not k.startswith("_")]
                    st.code("Attributes: " + ", ".join(attrs[:20]))
            else:
                st.write(type(ckpt))
        except Exception as e:
            st.error(f"Error loading checkpoint: {e}")