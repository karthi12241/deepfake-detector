"""
=============================================================
  DeepFake Detection System — Forensic Source Attribution
  M.Tech Project: Enhanced Deepfake Detection
  Model: Xception + ViT-B/16 with Gated Feature Fusion
  Run: streamlit run demo_app.py
=============================================================
"""
import os
from pathlib import Path
import zipfile

import gdown

ROOT = Path(__file__).resolve().parent
DRIVE_MODEL_PATH = Path("/content/drive/MyDrive/new_dataset_deepfake/outputs/cross_attention_v1/best.pt")

def download_model(destination=ROOT / "best.pt"):
    """Download the checkpoint from Google Drive and verify its container."""
    destination = Path(destination)
    if destination.exists() and zipfile.is_zipfile(destination):
        return

    if destination.exists():
        destination.unlink()

    print("Downloading model checkpoint...")
    file_id = os.environ.get("MODEL_FILE_ID", "1ssg4HbrIpbxewCu3E7Tp7vTw2CChTD3S")
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    gdown.download(id=file_id, output=str(temporary_path), quiet=False)

    if not temporary_path.exists() or not zipfile.is_zipfile(temporary_path):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Google Drive did not return a valid PyTorch checkpoint. "
            "Check that MODEL_FILE_ID points to the shared best.pt file."
        )

    temporary_path.replace(destination)
    print("Model downloaded ✅")


def get_model_path() -> Path:
    """Use the mounted Drive checkpoint, or download it for local runs."""
    if DRIVE_MODEL_PATH.exists() and zipfile.is_zipfile(DRIVE_MODEL_PATH):
        return DRIVE_MODEL_PATH

    local_model_path = ROOT / "best.pt"
    download_model(local_model_path)
    return local_model_path


import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import datetime
import hashlib
import requests
import numpy as np

ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    MODEL_PATH = get_model_path()
except Exception as error:
    st.error(f"Unable to obtain the model checkpoint: {error}")
    st.stop()

# Custom CSS — professional business styling
st.markdown("""
<style>
    /* Result cards */
    .fake-box {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 100%);
        border-left: 4px solid #e53935;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .fake-box h2 { color: #ef5350; font-size: 1.4rem; margin: 0; letter-spacing: 0.5px; }

    .real-box {
        background: linear-gradient(135deg, #0a1a0e 0%, #0f2d18 100%);
        border-left: 4px solid #2e7d32;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .real-box h2 { color: #43a047; font-size: 1.4rem; margin: 0; letter-spacing: 0.5px; }

    /* Forensic evidence card */
    .forensic-card {
        background: #0e1117;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    /* Badge */
    .badge-logged {
        display: inline-block;
        background: #1a2332;
        border: 1px solid #1e3a5f;
        color: #4fc3f7;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    /* Sidebar cleanup */
    section[data-testid="stSidebar"] { background: #0d1117; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0e1117;
        border: 1px solid #1e2432;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODEL ARCHITECTURE (matches train.ipynb exactly)
# ──────────────────────────────────────────────
class GatedFeatureFusion(nn.Module):
    def __init__(self, cnn_dim: int, vit_dim: int,
                 fusion_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.cnn_proj = nn.Sequential(
            nn.LayerNorm(cnn_dim),
            nn.Linear(cnn_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vit_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )

    def forward(self, cnn_features, vit_features):
        cnn_proj = self.cnn_proj(cnn_features)
        vit_proj = self.vit_proj(vit_features)
        joined   = torch.cat([cnn_proj, vit_proj], dim=1)
        gate     = self.gate(joined)
        fused    = gate * cnn_proj + (1.0 - gate) * vit_proj
        # This order must match the classifier weights in best.pt.
        return torch.cat([cnn_proj, vit_proj, fused], dim=1)  # [B, 1536]


class XceptionViTFusionModel(nn.Module):
    def __init__(self, fusion_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.cnn = timm.create_model(
            'legacy_xception', pretrained=False,
            num_classes=0, global_pool=''
        )
        self.vit = timm.create_model(
            'vit_base_patch16_224', pretrained=False,
            num_classes=0
        )
        self.fusion = GatedFeatureFusion(2048, 768, fusion_dim, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3),
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x):
        cnn_f = self.cnn(x).mean(dim=(2, 3))
        vit_f = self.vit.forward_features(x)[:, 0, :]
        fused = self.fusion(cnn_f, vit_f)
        return self.classifier(fused).squeeze(1)


# ──────────────────────────────────────────────
# LOAD MODEL (cached — runs once at startup)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path: str):
    model = XceptionViTFusionModel()
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        state = (ckpt.get('model_state')
                 or ckpt.get('model_state_dict')
                 or ckpt.get('state_dict')
                 or ckpt)
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def predict(model, image: Image.Image):
    """Returns (label, confidence, tensor) where label is 'FAKE' or 'REAL'."""
    tensor = val_transform(image.convert("RGB")).unsqueeze(0)  # [1,3,224,224]
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    label      = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if label == "FAKE" else 1.0 - prob
    return label, confidence, tensor


# ──────────────────────────────────────────────
# GRAD-CAM EXPLAINABILITY (XAI)
# ──────────────────────────────────────────────
def get_gradcam(model, tensor, pil_image):
    """
    Grad-CAM on Xception's last conv activation (act4).
    Returns (PIL overlay, None) on success, (None, error_str) on failure.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # ── Find act4 (last activation before GAP) in legacy_xception ────────
        # Walk named_modules to find the last activation layer reliably
        target_layer = None
        for name, mod in model.cnn.named_modules():
            if name in ("act4", "bn4", "conv4"):
                target_layer = mod
                break          # act4 is preferred; break on first match

        if target_layer is None:
            # Fallback: last non-container child of cnn
            leaves = [m for m in model.cnn.modules()
                      if len(list(m.children())) == 0]
            target_layer = leaves[-1] if leaves else None

        if target_layer is None:
            return None, "Cannot find target conv layer in Xception backbone."

        # ── Wrapper so GradCAM sees the full pipeline ─────────────────────────
        class _Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                cnn_f = self.m.cnn(x).mean(dim=(2, 3))      # [B, 2048]
                vit_f = self.m.vit.forward_features(x)[:, 0, :]  # [B, 768]
                fused = self.m.fusion(cnn_f, vit_f)
                return self.m.classifier(fused).squeeze(1)   # [B]  ← GradCAM needs 1-D

        wrapper   = _Wrap(model)
        cam       = GradCAM(model=wrapper, target_layers=[target_layer])
        # Grad-CAM passes each sample's scalar binary output to the target.
        targets   = [lambda output: output.squeeze()]
        grayscale = cam(input_tensor=tensor, targets=targets)[0]

        # ── Overlay on resized original ───────────────────────────────────────
        rgb     = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        return Image.fromarray(overlay), None

    except Exception as exc:
        return None, str(exc)


# ──────────────────────────────────────────────
# IP GEOLOCATION
# ──────────────────────────────────────────────
def get_location(ip: str):
    """
    Returns location dict on success, or None if unavailable.
    Uses ipapi.co (HTTPS, free, works on Streamlit Cloud).
    """
    if not ip or ip in ("127.0.0.1", "localhost", "::1"):
        return None   # no real IP — caller will skip location display
    try:
        r = requests.get(
            f"https://ipapi.co/{ip}/json/",
            headers={"User-Agent": "deepfake-detector/1.0"},
            timeout=5,
        )
        data = r.json()
        if data.get("error"):
            return None
        return {
            "city":       data.get("city", ""),
            "regionName": data.get("region", ""),
            "country":    data.get("country_name", ""),
            "isp":        data.get("org", ""),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────
# JSON FILE LOGGING
# ──────────────────────────────────────────────
def get_uploader_ip() -> str:
    """Get the client IP forwarded by a trusted reverse proxy.

    Only use this after deploying behind Nginx and restricting Streamlit's
    port 8501 to localhost. Otherwise forwarded headers can be forged.
    """
    try:
        forwarded = st.context.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return str(st.context.ip or "")
    except Exception:
        return ""


LOG_PATH = ROOT / "forensic_log.json"


def init_db():
    """Create empty JSON log file if it doesn't exist."""
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            json.dump([], f, indent=2)


def load_log() -> list:
    """Load all detections from JSON file."""
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_log(data: list):
    """Save full list back to JSON file."""
    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def log_detection(ip: str, loc: dict, confidence: float,
                  image_hash: str, filename: str):
    """Append a new detection entry to the JSON log."""
    data = load_log()
    entry = {
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip_address": ip,
        "city":       loc.get("city", "Unknown"),
        "region":     loc.get("regionName", "Unknown"),
        "country":    loc.get("country", "Unknown"),
        "isp":        loc.get("isp", "Unknown"),
        "confidence": round(confidence * 100, 2),
        "image_hash": image_hash,
        "filename":   filename,
    }
    data.append(entry)
    save_log(data)


def fetch_log(limit: int = 20) -> list:
    """Return last N detections as list of tuples (for table display)."""
    data = load_log()
    recent = data[-limit:][::-1]  # last N, newest first
    return [
        (
            d["timestamp"], d["ip_address"], d["city"],
            d["country"],   d["isp"],        d["confidence"],
            d["filename"],  d["image_hash"],
        )
        for d in recent
    ]


def count_detections() -> int:
    """Return total number of logged detections."""
    return len(load_log())


# ──────────────────────────────────────────────
# INITIALISE
# ──────────────────────────────────────────────
init_db()

# ──────────────────────────────────────────────
# SIDEBAR — minimal, professional
# ──────────────────────────────────────────────
st.sidebar.markdown("## 🔍 DeepFake Detector")
st.sidebar.caption("Xception + ViT-B/16 · Gated Fusion · FF++")
st.sidebar.divider()

# IP Source — only Demo or Auto (no manual entry)
st.sidebar.markdown("**Uploader Identification**")
ip_mode = st.sidebar.radio(
    "Mode",
    ["Demo Mode", "Auto-detect"],
    label_visibility="collapsed",
    help="Demo Mode uses a simulated IP for presentation."
)

if ip_mode == "Auto-detect":
    demo_ip = get_uploader_ip()   # silent — no display in sidebar
else:
    demo_ip = st.sidebar.selectbox(
        "Simulated Region",
        [
            "103.45.67.89  — Mumbai",
            "49.36.122.5   — Delhi",
            "117.96.0.1    — Bengaluru",
            "182.68.15.10  — Chennai",
        ],
        label_visibility="collapsed",
    ).split()[0]   # extract IP only

st.sidebar.divider()
st.sidebar.caption("M.Tech Project · Enhanced Deepfake Detection Framework")

# ──────────────────────────────────────────────
# LOAD MODEL (silent — no UI clutter)
# ──────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("Model checkpoint not found. Please check deployment configuration.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
st.markdown("## DeepFake Image Detection System")
st.caption("Forensic Source Attribution · Xception + ViT-B/16 · Gated Feature Fusion · FaceForensics++")
st.divider()

# Stats row
total_logged = count_detections()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", "96.0%", "FF++ Test Set")
c2.metric("ROC-AUC", "0.994", "Gated Fusion")
c3.metric("Precision", "98.8%", "Fake Class")
c4.metric("Cases Logged", str(total_logged))

st.divider()

# ── UPLOAD + RESULT ──
label       = None   # initialised here so Grad-CAM check below never crashes
img_tensor  = None
analyse_btn = False
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📤 Upload Image for Analysis")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"📎 {uploaded_file.name}",
                 use_container_width=True)

        analyse_btn = st.button(
            "🔍 Analyse Image", type="primary",
            use_container_width=True
        )
    else:
        st.info("👆 Upload an image to begin analysis")
        analyse_btn = False

with col_right:
    st.subheader("📊 Detection Result")

    if uploaded_file and analyse_btn:
        with st.spinner("🧠 Running model inference..."):
            label, confidence, img_tensor = predict(model, image)
            img_bytes  = uploaded_file.getvalue()
            image_hash = hashlib.sha256(img_bytes).hexdigest()

        # ── RESULT DISPLAY ──
        if label == "FAKE":
            st.markdown("""
            <div class="fake-box">
                <h2>🚨 DEEPFAKE DETECTED</h2>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Confidence Score", f"{confidence:.1%}")
            m2.metric("Threshold", "50.0%")

            st.progress(float(confidence), text=f"Manipulation probability: {confidence:.1%}")

            # Get geolocation silently
            location = get_location(demo_ip)

            # Log to DB
            log_detection(
                ip=demo_ip,
                loc=location or {},
                confidence=confidence,
                image_hash=image_hash,
                filename=uploaded_file.name
            )

            # Forensic Evidence Panel
            st.markdown("---")
            st.markdown(
                '<span class="badge-logged">● Evidence Recorded</span>',
                unsafe_allow_html=True
            )
            st.markdown("**Forensic Source Attribution**")

            # Build location rows only if available
            loc_rows = ""
            if location:
                loc_rows = f"""
| City | {location.get('city', '—')} |
| Region | {location.get('regionName', '—')} |
| Country | {location.get('country', '—')} |
| ISP | {location.get('isp', '—')} |"""

            st.markdown(f"""
| Attribute | Value |
|:----------|:------|
| IP Address | `{demo_ip}` |{loc_rows}
| Timestamp | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} |
| Image Fingerprint | `{image_hash[:32]}...` |
| Filename | `{uploaded_file.name}` |
            """)

        else:
            st.markdown("""
            <div class="real-box">
                <h2>AUTHENTIC — No Manipulation Detected</h2>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Authenticity Score", f"{confidence:.1%}")
            m2.metric("Forensic Record", "Not Created")

            st.progress(float(confidence), text=f"Authenticity confidence: {confidence:.1%}")

            st.caption("Image passed forensic verification. No record created.")

    elif not uploaded_file:
        st.markdown("""
        *Upload an image to begin analysis.*

        **How it works:**
        1. Upload any face image
        2. Dual-branch model analyses spatial & semantic features
        3. Manipulated images trigger forensic source attribution
        4. All evidence is timestamped and hash-verified
        """)

# ── GRAD-CAM — full width below both columns, FAKE detections only ──────────
if analyse_btn and label == "FAKE" and img_tensor is not None:
    st.divider()
    st.subheader("🔬 Explainable AI — Where Was Manipulation Detected?")
    st.caption(
        "Grad-CAM highlights regions that most influenced the FAKE decision. "
        "🔴 Red = high attention  ·  🔵 Blue = ignored region."
    )
    with st.spinner("Generating Grad-CAM heatmap..."):
        heatmap, cam_error = get_gradcam(model, img_tensor, image)

    if heatmap is not None:
        g1, g2 = st.columns(2)
        with g1:
            st.image(image.resize((224, 224)),
                     caption="📷 Original Image",
                     use_container_width=True)
        with g2:
            st.image(heatmap,
                     caption="🔴 Grad-CAM Heatmap — Manipulation Region",
                     use_container_width=True)
        st.info(
            "💡 Red/warm areas = where the model detected manipulation. "
            "Typical patterns: mouth area (Face2Face) · jaw boundary (FaceSwap) · "
            "skin texture (NeuralTextures)."
        )
    else:
        st.warning(f"⚠️ Grad-CAM could not be generated. Reason: `{cam_error}`")

# ──────────────────────────────────────────────
# FORENSIC LOG TABLE
# ──────────────────────────────────────────────
st.divider()
st.subheader("📋 Forensic Detection Log")
st.caption(
    "All deepfake detections are automatically logged here. "
    "This log can be exported as evidence for investigation."
)

log_rows = fetch_log(limit=25)

if log_rows:
    import pandas as pd
    df = pd.DataFrame(log_rows, columns=[
        "Timestamp", "IP Address", "City", "Country",
        "ISP", "Confidence (%)", "Filename", "Image Hash"
    ])
    df["Image Hash"] = df["Image Hash"].str[:16] + "..."

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence (%)": st.column_config.ProgressColumn(
                "Confidence (%)", min_value=50, max_value=100
            )
        }
    )

    # Export button
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Export Log as CSV",
        data=csv,
        file_name=f"forensic_log_{datetime.date.today()}.csv",
        mime="text/csv"
    )
else:
    st.caption("No detections recorded yet. Upload an image to begin.")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Enhanced Deepfake Detection Framework · M.Tech Project · "
    "Xception + ViT-B/16 · Gated Feature Fusion · FaceForensics++ · "
    "Accuracy 96.0% · AUC 0.994"
)
