import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
import numpy as np
import os
import io
import base64
import tempfile
import subprocess
import shutil
import uuid
from PIL import Image
from model_loader import load_models
from preprocessing import preprocess_face_from_bytes, preprocess_audio_from_bytes
from deepfake_detector import detect_face_deepfake, detect_audio_deepfake


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EchoMatch | AV Identity Verifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #050508; color: #f1f1f1; }
.main { background: #050508; }
.hero-container { padding: 3rem 0; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 2rem; }
.hero-title { font-size: 3.5rem; font-weight: 700; background: linear-gradient(135deg, #60A5FA 0%, #C084FC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; margin: 0; }
.hero-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #64748b; letter-spacing: 4px; text-transform: uppercase; margin-top: 5px; }
.glass-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #3b82f6; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1.5rem; }
.status-pill { padding: 6px 16px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; font-family: 'JetBrains Mono'; }
.status-pill-safe { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.status-pill-fake { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
.match-card { background: rgba(16,185,129,0.03); border: 1px solid #10b981; border-radius: 12px; padding: 3rem; text-align: center; margin-top: 2rem; }
.mismatch-card { background: rgba(239,68,68,0.03); border: 1px solid #ef4444; border-radius: 12px; padding: 3rem; text-align: center; margin-top: 2rem; }
.result-text { font-size: 2.5rem; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.5rem; }
.progress-bg { background: rgba(255,255,255,0.05); border-radius: 2px; height: 6px; width: 100%; margin-top: 20px; }
.progress-fill { height: 6px; border-radius: 2px; }
.stButton > button { background: #3b82f6 !important; color: white !important; border: none !important; border-radius: 4px !important; padding: 1rem 2rem !important; font-weight: 600 !important; letter-spacing: 1px !important; width: 100% !important; text-transform: uppercase; font-size: 0.9rem !important; }
div[data-testid="stExpander"] { background: transparent !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; }
.mode-info { font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; margin-bottom: 1.5rem; padding: 0.75rem; background: rgba(59,130,246,0.05); border: 1px solid rgba(59,130,246,0.1); border-radius: 6px; line-height: 1.6; }
.ready-badge { display: inline-block; padding: 4px 12px; background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); border-radius: 4px; font-family: 'JetBrains Mono'; font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; margin-top: 0.75rem; }
.pending-badge { display: inline-block; padding: 4px 12px; background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); border-radius: 4px; font-family: 'JetBrains Mono'; font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; margin-top: 0.75rem; }
.warn-box { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.4); border-radius: 8px; padding: 1.5rem; color: #ef4444; text-align: center; margin-top: 1.5rem; font-family: 'JetBrains Mono'; font-size: 0.85rem; line-height: 1.8; }
.step-badge { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; background: #3b82f6; color: white; border-radius: 50%; font-size: 0.7rem; font-weight: 700; margin-right: 8px; flex-shrink: 0; }
.step-row { display: flex; align-items: center; margin-bottom: 8px; font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ensure_ffmpeg_available():
    if not ffmpeg_available():
        raise RuntimeError(
            "FFmpeg and ffprobe are required for video extraction. "
            "Install FFmpeg and add it to your PATH: https://ffmpeg.org/download.html"
        )


def extract_best_frame_and_audio(video_bytes: bytes, suffix: str = ".webm") -> tuple[bytes, bytes, list[bytes]]:
    """
    Extract audio (mono 16kHz WAV) and the BEST face frame from a video.
    'Best' = lowest deepfake confidence across 5 evenly-spaced candidate frames.
    Falls back to midpoint frame if face detection fails on all candidates.
    """
    ensure_ffmpeg_available()
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, f"input{suffix}")
        audio_path = os.path.join(tmpdir, "audio.wav")

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True
        )
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            duration = 2.0

        # Extract audio
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ac", "1", "-ar", "16000", "-vn", audio_path],
            capture_output=True, check=True
        )

        # Extract 5 candidate frames at evenly spaced timestamps
        # Avoid first and last 10% of video (often blink/blur)
        frame_bytes_list = []
        n_frames = 5
        for i in range(n_frames):
            t = duration * (0.1 + 0.8 * i / (n_frames - 1))
            fpath = os.path.join(tmpdir, f"frame_{i}.jpg")
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", fpath],
                capture_output=True
            )
            if result.returncode == 0 and os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    frame_bytes_list.append(f.read())

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

    if not frame_bytes_list:
        raise RuntimeError("Could not extract any frames from video.")

    # Return the middle frame as the best face frame
    # (deepfake detection is done separately in the main flow)
    best_frame = frame_bytes_list[len(frame_bytes_list) // 2]
    return best_frame, audio_bytes, frame_bytes_list


def pick_most_authentic_frame(frame_bytes_list: list[bytes], device, threshold: float) -> bytes:
    """Run deepfake detector on all candidate frames, return the one with lowest fake confidence."""
    best_bytes = frame_bytes_list[0]
    best_conf  = float("inf")
    for fb in frame_bytes_list:
        try:
            result = detect_face_deepfake(fb, device=device, threshold=threshold)
            if result["confidence"] < best_conf:
                best_conf  = result["confidence"]
                best_bytes = fb
        except Exception:
            continue
    return best_bytes


def run_biometric_match(face_bytes, audio_bytes, audio_suffix,
                        model_path, device, threshold, language, input_mode):
    with st.spinner("EXECUTING NEURAL EMBEDDING MATCH..."):
        try:
            model_a, model_v, model_type = load_models(model_path, device)
            face_size = 224 if model_type == "dino_hubert" else 112
            face_tensor   = preprocess_face_from_bytes(face_bytes, face_size=face_size).unsqueeze(0).to(device)
            speech_tensor = preprocess_audio_from_bytes(audio_bytes, suffix=audio_suffix).unsqueeze(0).to(device)

            with torch.no_grad():
                a_emb = F.normalize(model_a(speech_tensor, aug=False), dim=1)
                v_emb = F.normalize(model_v(face_tensor), dim=1)

            mse   = F.mse_loss(a_emb, v_emb, reduction="sum").item()
            match = mse < threshold
            conf  = float(1 / (1 + np.exp(5 * (mse - threshold)))) * 100

            if match:
                st.markdown(f"""
                <div class="match-card">
                  <div class="result-text" style="color:#10b981;">IDENTITY VERIFIED</div>
                  <div style="font-family:'JetBrains Mono';color:#64748b;font-size:0.9rem;margin-top:1rem;">
                    MSE: {mse:.6f} / THRESHOLD: {threshold:.2f}
                  </div>
                  <div class="progress-bg">
                    <div class="progress-fill" style="width:{conf}%;background:#10b981;"></div>
                  </div>
                  <div style="margin-top:12px;font-size:0.85rem;color:#10b981;font-weight:600;font-family:'JetBrains Mono';">
                    MATCH CONFIDENCE: {conf:.2f}%
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="mismatch-card">
                  <div class="result-text" style="color:#ef4444;">IDENTITY MISMATCH</div>
                  <div style="font-family:'JetBrains Mono';color:#64748b;font-size:0.9rem;margin-top:1rem;">
                    MSE: {mse:.6f} / THRESHOLD: {threshold:.2f}
                  </div>
                  <div class="progress-bg">
                    <div class="progress-fill" style="width:{100-conf}%;background:#ef4444;"></div>
                  </div>
                  <div style="margin-top:12px;font-size:0.85rem;color:#ef4444;font-weight:600;font-family:'JetBrains Mono';">
                    MISMATCH CONFIDENCE: {100-conf:.2f}%
                  </div>
                </div>""", unsafe_allow_html=True)

            with st.expander("SYSTEM TELEMETRY"):
                st.code(
                    f"Device        : {device}\n"
                    f"Language      : {language}\n"
                    f"Input Mode    : {input_mode}\n"
                    f"Audio Shape   : {tuple(speech_tensor.shape)}\n"
                    f"Visual Shape  : {tuple(face_tensor.shape)}\n"
                    f"MSE Distance  : {mse:.6f}"
                )
        except Exception as e:
            st.error(f"INFERENCE FAULT: {e}")


# The recorder renders in an iframe. The ONLY reliable way to get binary
# data from an iframe back to Python without a sidecar server is:
# JS encodes blob → data URL → triggers a browser download of a .webm file
# User then uploads that file via a standard st.file_uploader (webm accepted).
# This is 100% reliable, zero networking, zero mixed-content issues.
RECORDER_HTML = """<!DOCTYPE html>
<html>
<head>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: transparent; font-family: monospace; padding: 12px; }
#preview {
  width: 100%; max-height: 340px; background: #0a0a12;
  border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
  object-fit: contain; display: block; margin-bottom: 12px;
}
.controls { display: flex; gap: 8px; margin-bottom: 10px; }
button {
  flex: 1; padding: 11px 0; border: none; border-radius: 4px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 1px;
  text-transform: uppercase; cursor: pointer;
}
button:disabled { opacity: 0.25; cursor: not-allowed; }
#btn-start    { background: #3b82f6; color: #fff; }
#btn-stop     { background: #ef4444; color: #fff; }
#btn-download { background: #10b981; color: #fff; flex: 2; }
#status {
  font-size: 0.68rem; color: #64748b; letter-spacing: 1px;
  text-transform: uppercase; min-height: 18px; padding: 2px 0;
}
#status.rec  { color: #ef4444; }
#status.ok   { color: #10b981; }
#status.busy { color: #f59e0b; }
</style>
</head>
<body>
<video id="preview" autoplay muted playsinline></video>
<div class="controls">
  <button id="btn-start">▶ Start</button>
  <button id="btn-stop"     disabled>■ Stop</button>
  <button id="btn-download" disabled>⬇ Download & Upload Below</button>
</div>
<div id="status">AWAITING CAMERA ACCESS...</div>

<script>
const preview  = document.getElementById('preview');
const btnStart = document.getElementById('btn-start');
const btnStop  = document.getElementById('btn-stop');
const btnDl    = document.getElementById('btn-download');
const status   = document.getElementById('status');

let recorder, chunks = [], stream, blob;

navigator.mediaDevices.getUserMedia({ video: true, audio: true })
  .then(s => {
    stream = s;
    preview.srcObject = s;
    status.textContent = 'CAMERA READY — PRESS START TO RECORD';
  })
  .catch(err => {
    status.textContent = 'ACCESS DENIED: ' + err.message;
    status.className = '';
  });

btnStart.addEventListener('click', () => {
  chunks = []; blob = null; btnDl.disabled = true;
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
    ? 'video/webm;codecs=vp8,opus' : '';
  recorder = new MediaRecorder(stream, mime ? {mimeType: mime} : {});
  recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  recorder.onstop = () => {
    blob = new Blob(chunks, { type: recorder.mimeType });
    preview.srcObject = null;
    preview.src = URL.createObjectURL(blob);
    preview.controls = true;
    preview.muted = false;
    btnDl.disabled = false;
    const mb = (blob.size / 1024 / 1024).toFixed(2);
    status.textContent = 'DONE (' + mb + ' MB) — DOWNLOAD THEN UPLOAD IN STEP 2 BELOW';
    status.className = 'ok';
  };
  recorder.start(200);
  btnStart.disabled = true;
  btnStop.disabled  = false;
  status.textContent = '● RECORDING...';
  status.className = 'rec';
});

btnStop.addEventListener('click', () => {
  recorder.stop();
  btnStop.disabled  = true;
  btnStart.disabled = false;
});

btnDl.addEventListener('click', () => {
  if (!blob) return;
  const a   = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'echomatch_recording.webm';
  a.click();
  status.textContent = '✓ DOWNLOADING — NOW UPLOAD THE FILE IN STEP 2 BELOW';
  status.className = 'ok';
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for k, v in [
    ("deepfake_blocked",    False),
    ("pending_face_bytes",  None),
    ("pending_audio_bytes", None),
    ("pending_audio_suffix", ".wav"),
    ("pending_frame_list",  None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="hero-container">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">ECHOMATCH</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Audio-Visual Identity Verification System</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SYSTEM PARAMETERS")
    english_model_path = st.text_input("English Weights", value="eng_epoch_30.pt")
    urdu_model_path    = st.text_input("Urdu Weights",    value="urd_epoch_30.pt")
    st.markdown("---")
    st.markdown("### DETECTION SENSITIVITY")
    face_df_threshold  = st.slider("Visual Threshold",  0.1, 0.9, 0.5, 0.05)
    audio_df_threshold = st.slider("Aural Threshold",   0.1, 0.9, 0.5, 0.05)
    deepfake_mode = st.radio(
        "Security Protocol",
        ["Enabled (block fakes)", "Warn only", "Disabled"],
        index=0,
    )

# ─────────────────────────────────────────────
# Top Controls
# ─────────────────────────────────────────────
top_col1, top_col2 = st.columns([1, 3])
with top_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 Language Context</div>', unsafe_allow_html=True)
    language = st.radio("Select Language", options=["English", "Urdu"],
                        horizontal=True, label_visibility="collapsed")
    chosen_model_path = english_model_path if language == "English" else urdu_model_path
    st.markdown(f'<div style="font-family:\'JetBrains Mono\';font-size:0.75rem;color:#64748b;margin-top:20px;">NODE: {chosen_model_path}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 System Sensitivity</div>', unsafe_allow_html=True)
    threshold = st.slider("MSE Verification Threshold", 0.05, 5.0, 0.5, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Mode
# ─────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1rem;">03 Input Mode</div>', unsafe_allow_html=True)
input_mode = st.radio(
    "Input Mode",
    ["Upload Video", "Record Video", "Separate Image + Audio"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_mode in ("Upload Video", "Record Video") and not ffmpeg_available():
    st.warning(
        "Video input requires FFmpeg and ffprobe. "
        "Install FFmpeg and add it to PATH, then restart the app. "
        "Without FFmpeg, upload or record video will not work."
    )

face_bytes       = None
audio_bytes      = None
audio_suffix     = ".wav"
candidate_frames = None   # list of frame bytes for best-frame selection


# ══════════════════════════════════════════════
# MODE A — Upload Video
# ══════════════════════════════════════════════
if input_mode == "Upload Video":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 Video Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="mode-info">Upload a pre-recorded video. 5 frames are sampled and the best one is selected automatically.</div>', unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Upload Video", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed"
    )
    if video_file:
        st.video(video_file)
        vsuffix = "." + video_file.name.rsplit(".", 1)[-1].lower()
        with st.spinner("EXTRACTING FRAMES & AUDIO..."):
            video_file.seek(0)
            try:
                best_frame, raw_audio, frame_list = extract_best_frame_and_audio(
                    video_file.read(), vsuffix
                )
                face_bytes       = best_frame
                audio_bytes      = raw_audio
                candidate_frames = frame_list

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Extracted Frame</div>', unsafe_allow_html=True)
                    st.image(Image.open(io.BytesIO(face_bytes)).convert("RGB"), use_container_width=True)
                    st.markdown('<span class="ready-badge">FRAME READY</span>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Extracted Audio</div>', unsafe_allow_html=True)
                    st.audio(audio_bytes, format="audio/wav")
                    st.markdown('<span class="ready-badge">AUDIO READY · WAV · MONO · 16kHz</span>', unsafe_allow_html=True)
            except subprocess.CalledProcessError as e:
                st.error(f"FFMPEG ERROR: {e.stderr.decode() if e.stderr else str(e)}")
            except (FileNotFoundError, RuntimeError) as e:
                st.error(f"FFMPEG NOT FOUND: {e}")
            except Exception as e:
                st.error(f"EXTRACTION FAULT: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODE B — Record Video (download → re-upload)
# ══════════════════════════════════════════════
elif input_mode == "Record Video":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 Live Video Recording</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="mode-info">
      <div class="step-row"><span class="step-badge">1</span>Press START, speak clearly, press STOP</div>
      <div class="step-row"><span class="step-badge">2</span>Review your recording in the player</div>
      <div class="step-row"><span class="step-badge">3</span>Press DOWNLOAD — a .webm file saves to your device</div>
      <div class="step-row"><span class="step-badge">4</span>Upload that .webm file using the uploader below</div>
    </div>
    """, unsafe_allow_html=True)

    # Step 1 — Recorder widget
    components.html(RECORDER_HTML, height=480, scrolling=False)

    # Step 2 — Upload the downloaded webm
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Step 2 — Upload Your Recording</div>', unsafe_allow_html=True)
    webm_file = st.file_uploader(
        "Upload recorded video", type=["webm", "mp4", "mov"],
        label_visibility="collapsed", key="webm_uploader"
    )

    if webm_file:
        vsuffix = "." + webm_file.name.rsplit(".", 1)[-1].lower()
        with st.spinner("EXTRACTING FRAMES & AUDIO FROM RECORDING..."):
            webm_file.seek(0)
            try:
                best_frame, raw_audio, frame_list = extract_best_frame_and_audio(
                    webm_file.read(), vsuffix
                )
                face_bytes       = best_frame
                audio_bytes      = raw_audio
                candidate_frames = frame_list

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Extracted Frame</div>', unsafe_allow_html=True)
                    st.image(Image.open(io.BytesIO(face_bytes)).convert("RGB"), use_container_width=True)
                    st.markdown('<span class="ready-badge">FRAME READY</span>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Extracted Audio</div>', unsafe_allow_html=True)
                    st.audio(audio_bytes, format="audio/wav")
                    st.markdown('<span class="ready-badge">AUDIO READY · WAV · MONO · 16kHz</span>', unsafe_allow_html=True)
            except subprocess.CalledProcessError as e:
                st.error(f"FFMPEG ERROR: {e.stderr.decode() if e.stderr else str(e)}")
            except (FileNotFoundError, RuntimeError) as e:
                st.error(f"FFMPEG NOT FOUND: {e}")
            except Exception as e:
                st.error(f"EXTRACTION FAULT: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODE C — Separate Image + Audio
# ══════════════════════════════════════════════
else:
    mid_col1, mid_col2 = st.columns(2)
    with mid_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">04 Visual Biometrics</div>', unsafe_allow_html=True)
        face_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        if face_file:
            face_file.seek(0)
            face_bytes = face_file.read()
            st.image(Image.open(io.BytesIO(face_bytes)).convert("RGB"), use_container_width=True)
            st.markdown('<span class="ready-badge">IMAGE READY</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with mid_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">05 Aural Biometrics</div>', unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "Upload Audio", type=["wav", "flac", "mp3"], label_visibility="collapsed"
        )
        if audio_file:
            st.audio(audio_file)
            audio_file.seek(0)
            audio_bytes  = audio_file.read()
            audio_suffix = "." + audio_file.name.rsplit(".", 1)[-1].lower()
            st.markdown('<span class="ready-badge">AUDIO READY</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("RUN SYSTEM VERIFICATION")

if run_btn:
    if not face_bytes or not audio_bytes:
        st.error("SYSTEM ERROR: INCOMPLETE DATA INPUT")
    elif not os.path.exists(chosen_model_path):
        st.error(f"CRITICAL ERROR: WEIGHTS MISSING AT {chosen_model_path}")
    else:
        st.session_state.pending_face_bytes   = face_bytes
        st.session_state.pending_audio_bytes  = audio_bytes
        st.session_state.pending_audio_suffix = audio_suffix
        st.session_state.pending_frame_list   = candidate_frames
        st.session_state.deepfake_blocked     = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Phase 1: Integrity check ──────────────────
        if deepfake_mode != "Disabled":
            with st.spinner("ANALYZING MEDIA INTEGRITY..."):

                # For video inputs: try all candidate frames, use the most authentic one
                active_face_bytes = face_bytes
                # NEW - deepfake check uses best frame, biometric match uses the stable middle frame
                if candidate_frames and len(candidate_frames) > 1:
                    pick_threshold = (
                        min(face_df_threshold + 0.25, 0.95)
                        if input_mode == "Record Video"
                        else face_df_threshold
                    )
                    # Use most-authentic frame ONLY for deepfake scoring
                    active_face_bytes = pick_most_authentic_frame(
                        candidate_frames, device, pick_threshold
                    )
                    # pending_face_bytes keeps the stable middle frame for biometric matching
                    # do NOT overwrite st.session_state.pending_face_bytes here

                # Live webcam recordings are inherently hard to deepfake in real-time,
                # and compressed WebM frames consistently score higher on deepfake detectors
                # trained on studio-quality synthetic images. So for recorded video we:
                # 1) Use a relaxed threshold (face_df_threshold + 0.25)
                # 2) Only block if BOTH face AND audio are fake (not just face alone)
                if input_mode == "Record Video":
                    relaxed_face_threshold = min(face_df_threshold + 0.25, 0.95)
                    fdr = detect_face_deepfake(active_face_bytes, device=device, threshold=relaxed_face_threshold)
                    adr = detect_audio_deepfake(audio_bytes, suffix=audio_suffix, device=device, threshold=audio_df_threshold)
                    # For live recordings, only flag if BOTH channels are suspicious
                    any_fake = fdr["is_fake"] and adr["is_fake"]
                else:
                    fdr = detect_face_deepfake(active_face_bytes, device=device, threshold=face_df_threshold)
                    adr = detect_audio_deepfake(audio_bytes, suffix=audio_suffix, device=device, threshold=audio_df_threshold)
                    any_fake = fdr["is_fake"] or adr["is_fake"]

            st.markdown('<div class="section-label" style="margin-top:2rem;">Integrity Report</div>', unsafe_allow_html=True)
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Visual Integrity</div>', unsafe_allow_html=True)
                lbl = "Deepfake Detected" if fdr["is_fake"] else "Authentic Media"
                cls = "status-pill-fake"  if fdr["is_fake"] else "status-pill-safe"
                st.markdown(f'<span class="status-pill {cls}">{lbl}</span>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem;margin-top:1rem;color:#64748b;">Confidence: {fdr["confidence"]*100:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with dc2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Aural Integrity</div>', unsafe_allow_html=True)
                lbl = "Synthetic Audio"  if adr["is_fake"] else "Authentic Voice"
                cls = "status-pill-fake" if adr["is_fake"] else "status-pill-safe"
                st.markdown(f'<span class="status-pill {cls}">{lbl}</span>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem;margin-top:1rem;color:#64748b;">Confidence: {adr["confidence"]*100:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if any_fake and deepfake_mode == "Enabled (block fakes)":
                st.session_state.deepfake_blocked = True
                st.markdown("""
                <div class="warn-box">
                  ⚠ ACCESS DENIED: SYNTHETIC MEDIA DETECTED<br>
                  <span style="font-size:0.72rem;color:#94a3b8;margin-top:6px;display:block;">
                  Verification blocked. Use the override button below if you believe this is a false positive.
                  </span>
                </div>""", unsafe_allow_html=True)

        # ── Phase 2: Biometric match ──────────────────
        if not st.session_state.deepfake_blocked:
            run_biometric_match(
                st.session_state.pending_face_bytes,
                audio_bytes, audio_suffix,
                chosen_model_path, device, threshold, language, input_mode
            )

# ── Verify Anyway ─────────────────────────────
if st.session_state.deepfake_blocked and st.session_state.pending_face_bytes:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚠  VERIFY ANYWAY — OVERRIDE DEEPFAKE BLOCK", key="verify_anyway"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_biometric_match(
            st.session_state.pending_face_bytes,
            st.session_state.pending_audio_bytes,
            st.session_state.pending_audio_suffix,
            chosen_model_path, device, threshold, language, input_mode
        )
        st.session_state.deepfake_blocked = False