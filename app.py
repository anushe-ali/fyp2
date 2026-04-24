import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
from model_loader import load_models
from preprocessing import preprocess_face_from_bytes, preprocess_audio_from_bytes
from deepfake_detector import detect_face_deepfake, detect_audio_deepfake

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

/* Deepfake result cards */
.result-deepfake {
    background: linear-gradient(135deg, #2e1a00, #1a0f00);
    border: 1px solid #a05000; border-radius: 16px; padding: 2rem; text-align: center;
    margin-bottom: 1rem;
}
.result-deepfake-label {
    font-family:'Syne',sans-serif; font-size:2rem; font-weight:800;
    color:#fb923c; letter-spacing:-1px;
}
.result-clean {
    background: linear-gradient(135deg, #0d1f2e, #0a1520);
    border: 1px solid #1a4a7a; border-radius: 16px; padding: 1rem 1.5rem;
    margin-bottom: 1rem; display:flex; align-items:center; gap:0.8rem;
}
.result-clean-label {
    font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#7eb8f7;
    letter-spacing:1px; text-transform:uppercase;
}

/* Identity result cards */
.result-match {
    background: linear-gradient(135deg, #0d2e1f, #0a1a14);
    border: 1px solid #1a5c3a; border-radius: 16px; padding: 2rem; text-align: center;
}
.result-nomatch {
    background: linear-gradient(135deg, #2e0d0d, #1a0a0a);
    border: 1px solid #5c1a1a; border-radius: 16px; padding: 2rem; text-align: center;
}
.result-label-match   { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#4ade80; letter-spacing:-1px; }
.result-label-nomatch { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#f87171; letter-spacing:-1px; }

.score-display { font-family:'JetBrains Mono',monospace; font-size:1rem; color:#8888aa; margin-top:0.5rem; }
.confidence-bar-bg { background:#1e1e35; border-radius:8px; height:8px; margin-top:1rem; overflow:hidden; }
.info-row { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#5a5a7a; margin-top:0.4rem; }

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
.pipeline-step {
    font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#3a3a5a;
    letter-spacing:1.5px; text-transform:uppercase; display:inline-block;
    background:#0e0e1c; border:1px solid #1a1a30; border-radius:6px;
    padding:0.2rem 0.6rem; margin-right:0.3rem;
}
.pipeline-arrow { color:#2a2a4a; margin-right:0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">AV Identity<br>Verifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ECAPA-TDNN · IResNet18 · Deepfake Guard · Audio–Visual Fusion</div>',
            unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:2rem;">
  <span class="pipeline-step">① Upload</span>
  <span class="pipeline-arrow">→</span>
  <span class="pipeline-step">② Deepfake Check</span>
  <span class="pipeline-arrow">→</span>
  <span class="pipeline-step">③ Identity Verify</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Paths")
    english_model_path = st.text_input("English model", value="model_english")
    urdu_model_path    = st.text_input("Urdu model",    value="model_urdu")
    st.markdown("---")
    st.markdown("### 🛡️ Deepfake Detection")
    face_df_threshold  = st.slider("Face fake threshold",  0.1, 0.9, 0.5, 0.05)
    audio_df_threshold = st.slider("Audio fake threshold", 0.1, 0.9, 0.5, 0.05)
    deepfake_mode = st.radio(
        "Deepfake check mode",
        ["Enabled (block fakes)", "Warn only (don't block)", "Disabled"],
        index=0,
    )

# ─────────────────────────────────────────────
# Language selector
# ─────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">01 — Select Language</div>', unsafe_allow_html=True)
language = st.radio("Language", options=["English", "Urdu"], horizontal=True,
                    label_visibility="collapsed")
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
        face_file.seek(0)
        st.image(Image.open(face_file).convert("RGB"), use_container_width=True)
        st.markdown('<div class="info-row">JPG / PNG · resized to 112×112 internally</div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">03 — Voice Clip</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload audio", type=["wav","flac","mp3"],
                                   label_visibility="collapsed", key="audio")
    if audio_file:
        st.audio(audio_file)
        st.markdown('<div class="info-row">WAV / FLAC / MP3 · first 2 s used (32 000 samples)</div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Threshold
# ─────────────────────────────────────────────
with st.expander("⚙️ Advanced: Identity Decision Threshold"):
    threshold = st.slider(
        "MSE Distance Threshold",
        min_value=0.05, max_value=5.0, value=0.5, step=0.05,
    )
    st.markdown(
        '<div class="info-row">Embeddings are L2-normalised, so MSE ∈ [0, 4]. '
        'EER threshold is typically 0.3–0.8 for well-trained models.</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# Verify button
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("🔍 Verify Identity")

if run_btn:
    if not face_file or not audio_file:
        st.warning("Please upload both a face image and a voice clip before verifying.")
    elif not os.path.exists(chosen_model_path):
        st.error(f"Model folder **{chosen_model_path}** not found.")
    else:
        # ── Read bytes once ───────────────────────────────────────────
        face_file.seek(0);  face_bytes  = face_file.read()
        audio_file.seek(0); audio_bytes = audio_file.read()
        audio_suffix = "." + audio_file.name.rsplit(".", 1)[-1].lower()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ════════════════════════════════════════════════════════════
        # STEP 1 – DEEPFAKE DETECTION
        # ════════════════════════════════════════════════════════════
        deepfake_blocked = False

        if deepfake_mode != "Disabled":
            with st.spinner("🛡️ Running deepfake checks…"):
                face_df_result  = detect_face_deepfake(
                    face_bytes, device=device, threshold=face_df_threshold)
                audio_df_result = detect_audio_deepfake(
                    audio_bytes, suffix=audio_suffix,
                    device=device, threshold=audio_df_threshold)

            st.markdown("#### 🛡️ Deepfake Analysis")
            df_col1, df_col2 = st.columns(2)

            with df_col1:
                if face_df_result["is_fake"]:
                    st.markdown(f"""
                    <div class="result-deepfake">
                        <div class="result-deepfake-label">⚠ FACE DEEPFAKE</div>
                        <div class="score-display">
                            Confidence: {face_df_result['confidence']*100:.1f}%<br>
                            Method: {face_df_result['method']}
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-clean">
                        <span style="font-size:1.4rem;">✅</span>
                        <div>
                            <div class="result-clean-label">Face: Authentic</div>
                            <div class="info-row">
                                Fake-prob: {face_df_result['confidence']*100:.1f}% |
                                {face_df_result['method']}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            with df_col2:
                if audio_df_result["is_fake"]:
                    st.markdown(f"""
                    <div class="result-deepfake">
                        <div class="result-deepfake-label">⚠ AUDIO DEEPFAKE</div>
                        <div class="score-display">
                            Confidence: {audio_df_result['confidence']*100:.1f}%<br>
                            Method: {audio_df_result['method']}
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-clean">
                        <span style="font-size:1.4rem;">✅</span>
                        <div>
                            <div class="result-clean-label">Audio: Authentic</div>
                            <div class="info-row">
                                Fake-prob: {audio_df_result['confidence']*100:.1f}% |
                                {audio_df_result['method']}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            with st.expander("🔬 Deepfake Detection Details"):
                st.code(
                    f"Face  → is_fake={face_df_result['is_fake']}  "
                    f"conf={face_df_result['confidence']:.4f}  "
                    f"method={face_df_result['method']}\n"
                    f"       {face_df_result['details']}\n\n"
                    f"Audio → is_fake={audio_df_result['is_fake']}  "
                    f"conf={audio_df_result['confidence']:.4f}  "
                    f"method={audio_df_result['method']}\n"
                    f"       {audio_df_result['details']}"
                )

            any_fake = face_df_result["is_fake"] or audio_df_result["is_fake"]

            if any_fake and deepfake_mode == "Enabled (block fakes)":
                deepfake_blocked = True
                fake_inputs = []
                if face_df_result["is_fake"]:
                    fake_inputs.append(f"**Face** ({face_df_result['confidence']*100:.1f}%)")
                if audio_df_result["is_fake"]:
                    fake_inputs.append(f"**Audio** ({audio_df_result['confidence']*100:.1f}%)")
                st.error(
                    f"🚫 **Deepfake Detected — Verification Blocked**\n\n"
                    f"Synthetic input(s) detected: {', '.join(fake_inputs)}.\n\n"
                    f"Please upload genuine media and try again."
                )

            elif any_fake and deepfake_mode == "Warn only (don't block)":
                st.warning("⚠️ One or more inputs appear synthetic. Proceeding anyway.")

        # ════════════════════════════════════════════════════════════
        # STEP 2 – IDENTITY VERIFICATION
        # ════════════════════════════════════════════════════════════
        if not deepfake_blocked:
            st.markdown("#### 🔍 Identity Verification")

            with st.spinner("Loading model & extracting embeddings…"):
                try:
                    # FIX 1: load_models returns 3 values (model_a, model_v, model_type)
                    model_a, model_v, _ = load_models(chosen_model_path, device)

                    face_tensor   = preprocess_face_from_bytes(face_bytes).unsqueeze(0).to(device)
                    speech_tensor = preprocess_audio_from_bytes(
                        audio_bytes, suffix=audio_suffix).unsqueeze(0).to(device)

                    with torch.no_grad():
                        a_emb = F.normalize(model_a(speech_tensor, aug=False), dim=1)
                        v_emb = F.normalize(model_v(face_tensor), dim=1)

                    mse_score  = F.mse_loss(a_emb, v_emb, reduction="sum").item()
                    is_match   = mse_score < threshold
                    confidence = float(1 / (1 + np.exp(5 * (mse_score - threshold)))) * 100

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
                                background:linear-gradient(90deg,#4ade80,#22d3ee);
                                border-radius:8px;"></div>
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
                                background:linear-gradient(90deg,#f87171,#fb923c);
                                border-radius:8px;"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    with st.expander("🔬 Embedding Debug Info"):
                        st.code(
                            f"Language    : {language}\n"
                            f"Model       : {chosen_model_path}\n"
                            f"Device      : {device}\n"
                            f"Audio tensor: {tuple(speech_tensor.shape)}\n"
                            f"Face tensor : {tuple(face_tensor.shape)}\n"
                            f"Audio emb   : {tuple(a_emb.shape)}\n"
                            f"Face emb    : {tuple(v_emb.shape)}\n"
                            f"MSE distance: {mse_score:.6f}\n"
                            f"Threshold   : {threshold}\n"
                            f"Decision    : {'MATCH' if is_match else 'NO MATCH'}"
                        )

                except KeyError as e:
                    st.error(f"Checkpoint key error: {e}")
                except RuntimeError as e:
                    st.error(f"Model load error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {type(e).__name__}: {e}")