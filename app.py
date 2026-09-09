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
DRIVE_MODEL_PATH = Path("/content/drive/MyDrive/new_dataset_deepfake/best.pt")

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
import os
from pathlib import Path


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

# Custom CSS for better UI
st.markdown("""
<style>
    .fake-box {
        background-color: #ff4b4b22;
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .real-box {
        background-color: #00c85322;
        border: 2px solid #00c853;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .forensic-box {
        background-color: #1e1e2e;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 15px;
        font-family: monospace;
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
    """Returns (label, confidence) where label is 'FAKE' or 'REAL'."""
    tensor = val_transform(image.convert("RGB")).unsqueeze(0)  # [1,3,224,224]
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    label      = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if label == "FAKE" else 1.0 - prob
    return label, confidence


# ──────────────────────────────────────────────
# IP GEOLOCATION (free, no API key needed)
# ──────────────────────────────────────────────
def get_location(ip: str) -> dict:
    """Returns city, country, ISP for a given IP address."""
    if ip in ("127.0.0.1", "localhost", "::1", ""):
        return {
            "status":  "demo",
            "country": "Demo Mode",
            "regionName": "Local",
            "city":    "Localhost",
            "isp":     "Local Network",
            "query":   ip,
        }
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=4)
        return r.json()
    except Exception:
        return {
            "status":  "fail",
            "country": "Unknown",
            "regionName": "Unknown",
            "city":    "Unknown",
            "isp":     "Unknown",
            "query":   ip,
        }


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
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/search.png", width=60
)
st.sidebar.title("⚙️ Configuration")

checkpoint_path = st.sidebar.text_input(
    "Model Checkpoint (.pt)",
    value=str(MODEL_PATH),
    help="Path to your trained Gated Fusion model checkpoint"
)

st.sidebar.divider()
st.sidebar.subheader("🌐 IP Configuration")
ip_mode = st.sidebar.radio(
    "IP Source",
    ["Detect Uploader IP", "Simulate IP (Demo)", "Enter IP Manually"],
    help="Automatic mode requires a trusted Nginx/reverse-proxy deployment."
)

if ip_mode == "Detect Uploader IP":
    demo_ip = get_uploader_ip()
    if demo_ip:
        st.sidebar.success(f"Uploader network IP: {demo_ip}")
    else:
        st.sidebar.warning("Uploader IP is unavailable. Deploy behind a configured reverse proxy.")
elif ip_mode == "Simulate IP (Demo)":
    demo_ip = st.sidebar.selectbox(
        "Simulated Uploader IP",
        [
            "103.45.67.89   (Mumbai, Jio)",
            "49.36.122.5    (Delhi, Airtel)",
            "117.96.0.1     (Bengaluru, BSNL)",
            "182.68.15.10   (Chennai, Airtel)",
            "Custom...",
        ]
    )
    if demo_ip == "Custom...":
        demo_ip = st.sidebar.text_input("Enter IP:", "8.8.8.8")
    else:
        demo_ip = demo_ip.split()[0]  # extract IP part
else:
    demo_ip = st.sidebar.text_input("IP Address:", "")

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
if not os.path.exists(checkpoint_path):
    st.sidebar.error(f"❌ Not found: `{checkpoint_path}`")
    st.error(f"""
    ### Model checkpoint not found!
    Please set the correct path to your `best.pt` file in the sidebar.
    
    Your checkpoint should be at the path printed at the end of training.
    """)
    st.stop()

with st.sidebar:
    try:
        with st.spinner("Loading model..."):
            model = load_model(checkpoint_path)
    except Exception as error:
        st.error(f"Could not load the model checkpoint: {error}")
        st.stop()
    st.success("✅ Model loaded (110M params)")
    st.caption("Xception + ViT-B/16 | Gated Fusion")

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
st.title("🔍 DeepFake Image Detection System")
st.markdown(
    "**M.Tech Project** — Enhanced Deepfake Detection with "
    "Forensic Source Attribution | Xception + ViT-B/16 + Gated Fusion"
)
st.divider()

# Stats row
total_logged = count_detections()
c1, c2, c3 = st.columns(3)
c1.metric("Model Accuracy", "96.0%", "On FF++ Test Set")
c2.metric("ROC-AUC", "0.994", "Gated Fusion")
c3.metric("Total Deepfakes Logged", str(total_logged), "In forensic DB")

st.divider()

# ── UPLOAD + RESULT ──
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
            label, confidence = predict(model, image)
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
            m1.metric("Fake Confidence", f"{confidence:.1%}")
            m2.metric("Decision Threshold", "50.0%")

            # Get geolocation
            with st.spinner("📡 Looking up IP location..."):
                location = get_location(demo_ip)

            # Log to DB
            log_detection(
                ip=demo_ip,
                loc=location,
                confidence=confidence,
                image_hash=image_hash,
                filename=uploaded_file.name
            )

            # Forensic panel
            st.markdown("---")
            st.markdown("### 🕵️ Forensic Evidence Captured")

            st.markdown(f"""
| Field | Captured Value |
|:------|:--------------|
| 🌐 **IP Address** | `{demo_ip}` |
| 🏙️ **City** | {location.get('city','Unknown')} |
| 🗺️ **Region** | {location.get('regionName','Unknown')} |
| 🌍 **Country** | {location.get('country','Unknown')} |
| 📡 **ISP** | {location.get('isp','Unknown')} |
| ⏰ **Timestamp** | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| 🔑 **Image Hash** | `{image_hash[:24]}...` |
| 📁 **Filename** | `{uploaded_file.name}` |
            """)

            st.warning(
                "⚠️ This detection has been saved to the forensic database. "
                "Evidence is available for cybercrime investigation."
            )

        else:
            st.markdown("""
            <div class="real-box">
                <h2>✅ AUTHENTIC IMAGE</h2>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Real Confidence", f"{confidence:.1%}")
            m2.metric("Forensic Log", "Not Created")

            st.info(
                "ℹ️ Image appears genuine. "
                "No forensic record created."
            )

    elif not uploaded_file:
        st.markdown("""
        *Results will appear here after you upload and analyse an image.*
        
        **How it works:**
        1. Upload any image
        2. Model analyses for manipulation artifacts
        3. If FAKE → forensic log automatically created
        4. Evidence stored with IP, location, timestamp
        """)

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
    st.info(
        "📭 No detections logged yet. "
        "Upload a deepfake image to create the first entry."
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
col_f1, col_f2 = st.columns(2)
col_f1.caption(
    "**M.Tech Project** — Enhanced Deepfake Image Detection Framework"
)
col_f2.caption(
    "Xception + ViT-B/16 | Gated Feature Fusion | "
    "FaceForensics++ | Accuracy: 96.0% | AUC: 0.994"
)
