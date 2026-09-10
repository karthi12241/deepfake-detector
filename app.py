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
    file_id = os.environ.get("MODEL_FILE_ID", "1q56SwOAoPCYlhskiMZ-HkKwu3yjpI16m")
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
from PIL import Image
import json
import datetime
import hashlib
import requests
import numpy as np
import cv2

# Keep app inference identical to the evaluation notebook.  These helpers
# inspect the checkpoint to select the correct fusion architecture and use the
# checkpoint's recorded image size.
from test import build_model, make_transform, unpack_checkpoint
from ip_utils import get_uploader_ip as resolve_uploader_ip, get_ip_diagnostics

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
    """Load exactly the model variant and input settings saved in the checkpoint.

    The evaluation notebook calls ``unpack_checkpoint`` and ``build_model``;
    using the same path here is essential because a Gated and a Bidirectional
    checkpoint have different forward passes despite sharing the backbones.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # Support older PyTorch releases too.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict, model_config = unpack_checkpoint(checkpoint)
    model = build_model(model_config, state_dict).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    if hasattr(model, "cnn") and hasattr(model.cnn, "act4") and hasattr(model.cnn.act4, "inplace"):
        model.cnn.act4.inplace = False
    if hasattr(model, "cnn") and hasattr(model.cnn, "act3") and hasattr(model.cnn.act3, "inplace"):
        model.cnn.act3.inplace = False

    saved_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved_config, dict):
        saved_config = {}
    image_size = int(saved_config.get("training", {}).get("image_size", 224))
    model_kind = "Gated" if "fusion.gate.0.weight" in state_dict else "Bidirectional"
    return model, make_transform(image_size), device, image_size, model_kind


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def predict(model, transform, device, image: Image.Image, threshold: float = 0.5):
    """Match ``Combined_evaluation(2).ipynb`` single-image inference exactly.

    The notebook evaluates the full uploaded image (no automatic face crop),
    applies ``make_transform(image_size)``, and marks probabilities >= 0.5 as
    FAKE.  Return the fake probability as well so the UI does not hide it.
    """
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    label      = "FAKE" if prob >= threshold else "REAL"
    confidence = prob if label == "FAKE" else 1.0 - prob
    return label, confidence, prob, tensor


# ──────────────────────────────────────────────
# GRAD-CAM EXPLAINABILITY (XAI)
# ──────────────────────────────────────────────
class SingleLogitTarget:
    """Explicit Grad-CAM target for binary single-logit classification.

    This model produces a single scalar logit representing evidence for the FAKE class:
    logit -> sigmoid(logit) -> FAKE if probability >= 0.5.
    This target returns the model's scalar logit directly to compute d(logit)/d(conv4),
    without assuming two classes or indexing class 1.
    """
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 0:
            return model_output
        return model_output.reshape(-1)[0]


def find_last_conv_layer(cnn_backbone: nn.Module) -> tuple[nn.Module | None, str]:
    """Identify the last meaningful convolutional layer in Xception (conv4).

    In timm's legacy_xception, the exit-flow layers before global average pooling are:
    - conv4: SeparableConv2d (1024 -> 2048 channels), the final conv layer.
    - act4: ReLU activation immediately following conv4/bn4.
    - block12: Final residual block preceding conv3/conv4.

    Returns (layer_module, layer_name).
    """
    if hasattr(cnn_backbone, "conv4"):
        return cnn_backbone.conv4, "model.cnn.conv4 (SeparableConv2d)"

    named_mods = dict(cnn_backbone.named_modules())
    for target_name in ("conv4", "act4", "bn4", "block12"):
        if target_name in named_mods:
            mod = named_mods[target_name]
            return mod, f"model.cnn.{target_name} ({mod.__class__.__name__})"

    # Fallback: scan for any convolutional module in reverse order
    conv_layers = [
        (name, mod) for name, mod in cnn_backbone.named_modules()
        if "conv" in mod.__class__.__name__.lower() and not isinstance(mod, nn.Sequential)
    ]
    if conv_layers:
        name, mod = conv_layers[-1]
        return mod, f"model.cnn.{name} ({mod.__class__.__name__})"

    leaves = [(name, m) for name, m in cnn_backbone.named_modules() if len(list(m.children())) == 0]
    if leaves:
        name, mod = leaves[-1]
        return mod, f"model.cnn.{name} ({mod.__class__.__name__})"
    return None, "Unknown"


def inspect_layer_attribution(model, layer, tensor):
    """Compute and extract complete Grad-CAM diagnostics for a candidate layer.

    Evaluates:
    - captured activations A
    - captured gradients G
    - channel weights alpha_k = GAP(G)
    - pre-ReLU weighted sum S = sum(alpha_k * A_k)
    - post-ReLU CAM = max(0, S)
    """
    from pytorch_grad_cam import GradCAM

    cam = GradCAM(model=model, target_layers=[layer])
    # Execute Grad-CAM on model scalar logit
    _ = cam(input_tensor=tensor, targets=[SingleLogitTarget()])[0]

    # Captured activations [1, C, H, W]
    act = cam.activations_and_grads.activations[0].detach().cpu().numpy()
    # Captured gradients [1, C, H, W]
    grad = cam.activations_and_grads.gradients[0].detach().cpu().numpy()

    # Clean non-finite if any
    act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)





    # Channel weights: global average pooling over spatial dimensions
    alpha = np.mean(grad, axis=(2, 3), keepdims=True)  # [1, C, 1, 1]

    # Pre-ReLU spatial weighted combination: S = sum_k (alpha_k * A_k)
    weighted_sum = np.sum(alpha * act, axis=1)[0]  # [H, W]

    # Post-ReLU spatial combination: max(0, S)
    post_relu = np.maximum(weighted_sum, 0.0)

    grad_max_abs = max(abs(float(np.min(grad))), abs(float(np.max(grad))))
    grad_std = float(np.std(grad))

    # Categorize status
    if grad_max_abs <= 1e-7 or grad_std <= 1e-7:
        status = "ZERO_GRADIENT"
    elif float(np.max(post_relu)) <= 1e-6 or float(np.std(post_relu)) <= 1e-6:
        status = "ZERO_CAM"
    else:
        status = "VALID_CAM"

    diag = {
        "status": status,
        "act_min": float(np.min(act)),
        "act_max": float(np.max(act)),
        "act_mean": float(np.mean(act)),
        "act_std": float(np.std(act)),
        "grad_min": float(np.min(grad)),
        "grad_max": float(np.max(grad)),
        "grad_mean": float(np.mean(grad)),
        "grad_std": grad_std,
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha)),
        "pre_relu_min": float(np.min(weighted_sum)),
        "pre_relu_max": float(np.max(weighted_sum)),
        "pre_relu_mean": float(np.mean(weighted_sum)),
        "pre_relu_std": float(np.std(weighted_sum)),
        "post_relu_min": float(np.min(post_relu)),
        "post_relu_max": float(np.max(post_relu)),
        "post_relu_mean": float(np.mean(post_relu)),
        "post_relu_std": float(np.std(post_relu)),
        "cam_min": float(np.min(post_relu)),
        "cam_max": float(np.max(post_relu)),
        "cam_mean": float(np.mean(post_relu)),
        "cam_std": float(np.std(post_relu)),
    }
    return status, diag, post_relu


def get_gradcam(model, tensor, pil_image):
    """
    Compute Grad-CAM attribution on Xception's final spatial feature representation.

    Deterministic Primary Target: model.cnn.act4 (ReLU)
    Academic & Research Justification:
    1. In timm's legacy_xception, act4 is the final rectified spatial feature map (2048 x 7 x 7)
       following conv4 (SeparableConv2d) and bn4 (BatchNorm2d).
    2. act4 is the exact representation that directly feeds Global Average Pooling:
       cnn_f = model.cnn(x).mean(dim=(2, 3))
       which passes into the GatedFeatureFusion module.
    3. Its activations are non-negative (A >= 0), ensuring positive channel weights (alpha_k > 0)
       unambiguously represent positive contributions toward the FAKE logit. In contrast, raw conv4
       contains negative pre-activations that cause mathematical suppression and all-zero CAMs.
    4. Deterministic selection ensures reproducible, scientifically sound evaluations rather than
       heuristically switching layers to force a visual artifact.

    Optional Diagnostic Comparison:
    - Also inspects conv4 as an auxiliary baseline for research diagnostics without altering the primary CAM.

    Returns:
    - (PIL.Image overlay, diagnostics_dict, None) on success.
    - (None, diagnostics_dict, "ZERO_CAM") if gradients exist but post-ReLU CAM has no positive attribution.
    - (None, diagnostics_dict, "ZERO_GRADIENT") if gradients are negligible.
    - (None, diagnostics_dict, error_str) on execution error.
    """
    diagnostics = {}
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # Deterministic primary target: act4
        primary_target_mod = getattr(model.cnn, "act4", None)
        primary_target_name = "model.cnn.act4 (ReLU)"

        if primary_target_mod is None:
            # Fallback only if act4 is absent in unexpected architecture
            primary_target_mod, primary_target_name = find_last_conv_layer(model.cnn)

        if primary_target_mod is None:
            return None, {"status": "ERROR"}, "Cannot locate candidate convolutional layers in Xception backbone."

        # Compute Grad-CAM deterministically on the primary target
        status, diag, post_relu = inspect_layer_attribution(model, primary_target_mod, tensor)
        diag["target_layer_name"] = primary_target_name
        diagnostics.update(diag)

        # Auxiliary research comparison: also inspect conv4 diagnostics if available
        if hasattr(model.cnn, "conv4") and model.cnn.conv4 is not primary_target_mod:
            try:
                c4_status, c4_diag, _ = inspect_layer_attribution(model, model.cnn.conv4, tensor)
                c4_diag["target_layer_name"] = "model.cnn.conv4 (SeparableConv2d)"
                diagnostics["conv4_comparison"] = c4_diag
            except Exception:
                pass

        # If the deterministic primary target produces ZERO_CAM or ZERO_GRADIENT,
        # report honestly without fabricating a map or silently switching layers.
        if status != "VALID_CAM":
            return None, diagnostics, status

        # Robust min-max normalization
        cam_min = diag["post_relu_min"]
        cam_max = diag["post_relu_max"]
        norm_cam = (post_relu - cam_min) / (cam_max - cam_min)
        norm_cam = np.clip(norm_cam, 0.0, 1.0)

        # Resize CAM to original image dimensions
        orig_w, orig_h = pil_image.size
        cam_resized = cv2.resize(norm_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        # Overlay onto original RGB image
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb, cam_resized, use_rgb=True, image_weight=0.5)
        return Image.fromarray(overlay), diagnostics, None

    except Exception as exc:
        diagnostics["status"] = "ERROR"
        return None, diagnostics, str(exc)


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# IP GEOLOCATION
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_location(ip: str):
    """
    Returns location dict on success, or fallback info.
    Uses multi-provider fallback (ipapi.co -> freeipapi.com -> ip-api.com).
    Cached for 1 hour to optimize performance.
    """
    if not ip or ip.startswith("Unavailable") or ip in ("127.0.0.1", "localhost", "::1", "Unknown"):
        if not ip or ip.startswith("Unavailable"):
            return {
                "city": "Unavailable",
                "regionName": "Not exposed by hosting platform",
                "country": "Unknown",
                "isp": "N/A (Streamlit Cloud proxy limitation)",
            }
        return {
            "city": "Localhost / Internal",
            "regionName": "Local Network",
            "country": "Local Machine",
            "isp": "Loopback",
        }

    # Provider 1: ipapi.co (HTTPS)
    try:
        r = requests.get(
            f"https://ipapi.co/{ip}/json/",
            headers={"User-Agent": "deepfake-detector/1.0"},
            timeout=3,
        )
        if r.status_code == 200:
            data = r.json()
            if not data.get("error"):
                return {
                    "city":       data.get("city") or "Unknown",
                    "regionName": data.get("region") or "Unknown",
                    "country":    data.get("country_name") or "Unknown",
                    "isp":        data.get("org") or "Unknown",
                }
    except Exception:
        pass

    # Provider 2: freeipapi.com (HTTPS)
    try:
        r = requests.get(
            f"https://freeipapi.com/api/json/{ip}",
            headers={"User-Agent": "deepfake-detector/1.0"},
            timeout=3,
        )
        if r.status_code == 200:
            data = r.json()
            return {
                "city":       data.get("cityName") or "Unknown",
                "regionName": data.get("regionName") or "Unknown",
                "country":    data.get("countryName") or "Unknown",
                "isp":        data.get("asnOrganization") or "Unknown",
            }
    except Exception:
        pass

    # Provider 3: ip-api.com (HTTP)
    try:
        r = requests.get(
            f"http://ip-api.com/json/{ip}",
            headers={"User-Agent": "deepfake-detector/1.0"},
            timeout=3,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return {
                    "city":       data.get("city") or "Unknown",
                    "regionName": data.get("regionName") or "Unknown",
                    "country":    data.get("country") or "Unknown",
                    "isp":        data.get("isp") or "Unknown",
                }
    except Exception:
        pass

    return {
        "city":       "Unknown",
        "regionName": "Unknown",
        "country":    "Unknown",
        "isp":        "Unknown",
    }


# ──────────────────────────────────────────────
# JSON FILE LOGGING
# ──────────────────────────────────────────────
def get_uploader_ip() -> str:
    """Resolve the uploader's real public IP automatically."""
    try:
        headers = getattr(st.context, "headers", {}) or {}
        peer_ip = getattr(st.context, "ip", "") or ""
        return resolve_uploader_ip(headers, peer_ip, trust_forwarded_headers=True, allow_local_dev_fallback=False)
    except Exception:
        return "Unavailable (Not exposed by hosting platform)"


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
                  image_hash: str, filename: str, verdict: str = "FAKE"):
    """Append a new detection entry to the JSON log."""
    data = load_log()
    entry = {
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdict":    verdict,
        "ip_address": ip or "Unknown",
        "city":       loc.get("city") or "Unknown",
        "region":     loc.get("regionName") or loc.get("region") or "Unknown",
        "country":    loc.get("country") or "Unknown",
        "isp":        loc.get("isp") or "Unknown",
        "confidence": round(confidence * 100, 2),
        "image_hash": image_hash,
        "filename":   filename,
    }
    data.append(entry)
    save_log(data)


def fetch_log(limit: int = 25) -> list:
    """Return last N detections as list of tuples (for table display)."""
    data = load_log()
    recent = data[-limit:][::-1]  # last N, newest first
    return [
        (
            d.get("timestamp", ""),
            d.get("verdict", "FAKE"),
            d.get("ip_address") or "Unknown",
            d.get("city", "Unknown"),
            d.get("country", "Unknown"),
            d.get("isp", "Unknown"),
            d.get("confidence", 0.0),
            d.get("filename", "Unknown"),
            d.get("image_hash", ""),
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
st.sidebar.caption("Xception + ViT-B/16 · checkpoint-matched fusion · FF++")
st.sidebar.divider()

# IP Source — Auto-detect as default, Demo Mode as optional
st.sidebar.markdown("**Uploader Identification**")
ip_mode = st.sidebar.radio(
    "Mode",
    ["Auto-detect (Real Public IP)", "Demo Mode (Simulated IP)"],
    index=0,
    label_visibility="collapsed",
    help="Auto-detect captures the uploader's real public IP address automatically. Demo mode allows selecting simulated IPs."
)

if ip_mode == "Auto-detect (Real Public IP)":
    uploader_ip = get_uploader_ip()
    if uploader_ip.startswith("Unavailable"):
        st.sidebar.markdown("⚠️ **IP:** `Not exposed by platform`")
        st.sidebar.caption("Streamlit Cloud ingress proxy did not forward client IP.")
    else:
        st.sidebar.markdown(f"🟢 **Live IP:** `{uploader_ip}`")
        loc_sidebar = get_location(uploader_ip)
        if loc_sidebar and loc_sidebar.get("city") and loc_sidebar.get("city") not in ("Unknown", "Unavailable"):
            st.sidebar.caption(f"📍 {loc_sidebar.get('city')}, {loc_sidebar.get('country')}")

    # Temporary testing diagnostics for deployment verification
    with st.sidebar.expander("🔍 Network IP Diagnostics (Testing)", expanded=False):
        ip_diag = get_ip_diagnostics(getattr(st.context, "headers", {}) or {}, getattr(st.context, "ip", "") or "")
        st.write(f"**st.context.ip:** `{ip_diag['peer_ip_raw']}`")
        st.write(f"**Environment:** `{ip_diag['environment']}`")
        st.write(f"**Resolution Source:** `{ip_diag['resolution_source']}`")
        st.write(f"**Resolved IP:** `{ip_diag['final_resolved_ip']}`")
        st.caption("Forwarded Headers (Sanitized):")
        if ip_diag["relevant_headers"]:
            for k, v in ip_diag["relevant_headers"].items():
                st.text(f"{k}: {v}")
        else:
            st.caption("None present in request context")
else:
    sim_selection = st.sidebar.selectbox(
        "Simulated Region",
        [
            "103.45.67.89  — Mumbai",
            "49.36.122.5   — Delhi",
            "117.96.0.1    — Bengaluru",
            "182.68.15.10  — Chennai",
            "Custom IP...",
        ],
        label_visibility="collapsed",
    )
    if sim_selection == "Custom IP...":
        uploader_ip = st.sidebar.text_input("Enter IP:", "8.8.8.8").strip()
    else:
        uploader_ip = sim_selection.split()[0]
    st.sidebar.caption(f"🎭 Using simulated IP: `{uploader_ip}`")

st.sidebar.divider()
st.sidebar.caption("M.Tech Project · Enhanced Deepfake Detection Framework")

# ──────────────────────────────────────────────
# LOAD MODEL (silent — no UI clutter)
# ──────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("Model checkpoint not found. Please check deployment configuration.")
    st.stop()

try:
    model, inference_transform, device, image_size, model_kind = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
st.markdown("## DeepFake Image Detection System")
st.caption(
    f"Forensic Source Attribution · Xception + ViT-B/16 · {model_kind} Fusion · "
    f"FaceForensics++ · Evaluation input: {image_size}px"
)
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
            label, confidence, fake_probability, img_tensor = predict(
                model, inference_transform, device, image
            )
            img_bytes  = uploaded_file.getvalue()
            image_hash = hashlib.sha256(img_bytes).hexdigest()

        # Geolocation lookup for uploader IP
        location = get_location(uploader_ip)

        # Log detection to database (both REAL and FAKE submissions)
        log_detection(
            ip=uploader_ip,
            loc=location or {},
            confidence=confidence,
            image_hash=image_hash,
            filename=uploaded_file.name,
            verdict=label
        )

        # Build location rows if available
        loc_rows = ""
        if location and location.get("city") not in ("Unknown", "Unavailable"):
            loc_rows = f"""
| City | {location.get('city', '—')} |
| Region | {location.get('regionName', '—')} |
| Country | {location.get('country', '—')} |
| ISP | {location.get('isp', '—')} |"""
        elif location and location.get("city") == "Unavailable":
            loc_rows = """
| Geolocation | `Not Available (Proxy / Cloud Limitation)` |
| Note | `Streamlit Cloud proxy did not forward client IP` |"""

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

            st.caption(f"Fake probability: {fake_probability:.4%} · Evaluation mode: full image (no face crop)")

            st.progress(float(confidence), text=f"Manipulation probability: {confidence:.1%}")

            # Forensic Evidence Panel
            st.markdown("---")
            st.markdown(
                '<span class="badge-logged">● Evidence Recorded</span>',
                unsafe_allow_html=True
            )
            st.markdown("**Forensic Source Attribution**")

            st.markdown(f"""
| Attribute | Value |
|:----------|:------|
| Verdict | `🚨 DEEPFAKE (Manipulated)` |
| IP Address | `{uploader_ip}` |{loc_rows}
| Timestamp | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} |
| Image Fingerprint | `{image_hash[:32]}...` |
| Filename | `{uploaded_file.name}` |
            """)

        else:
            st.markdown("""
            <div class="real-box">
                <h2>✅ AUTHENTIC — No Manipulation Detected</h2>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Authenticity Score", f"{confidence:.1%}")
            m2.metric("Forensic Record", "Recorded ✅")

            st.caption(f"Fake probability: {fake_probability:.4%} · Evaluation mode: full image (no face crop)")

            st.progress(float(confidence), text=f"Authenticity confidence: {confidence:.1%}")

            # Forensic Verification Panel
            st.markdown("---")
            st.markdown(
                '<span class="badge-logged" style="background:#0e2a18; border-color:#2e7d32; color:#81c784;">● Verification Recorded</span>',
                unsafe_allow_html=True
            )
            st.markdown("**Forensic Source Attribution & Verification**")

            st.markdown(f"""
| Attribute | Value |
|:----------|:------|
| Verdict | `✅ AUTHENTIC (Genuine)` |
| IP Address | `{uploader_ip}` |{loc_rows}
| Timestamp | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} |
| Image Fingerprint | `{image_hash[:32]}...` |
| Filename | `{uploaded_file.name}` |
            """)

    elif not uploaded_file:
        st.markdown("""
        *Upload an image to begin analysis.*

        **How it works:**
        1. Upload any face image
        2. Dual-branch model analyses spatial & semantic features
        3. Real public IP & geolocation automatically identified for all submissions
        4. Complete forensic audit trail recorded with cryptographic hash
        """)

# ── GRAD-CAM — full width below both columns, FAKE detections only ──────────
if analyse_btn and label == "FAKE" and img_tensor is not None:
    st.divider()
    st.subheader("🔬 Explainable AI — Xception Grad-CAM")
    st.caption("🔴 Red/warm regions indicate areas with stronger contribution to the FAKE prediction through the Xception branch.")

    with st.spinner("Computing Xception Grad-CAM attribution..."):
        heatmap, diagnostics, cam_error = get_gradcam(model, img_tensor, image)

    if heatmap is not None:
        g1, g2 = st.columns(2)
        with g1:
            st.image(image,
                     caption=f"📷 Uploaded Image ({image.size[0]}×{image.size[1]})",
                     use_container_width=True)
        with g2:
            st.image(heatmap,
                     caption="🔴 Xception Grad-CAM Heatmap (act4)",
                     use_container_width=True)

        st.markdown("""
        **Color Mapping Guide:**  
        🔵 **Blue:** Low contribution toward FAKE prediction  
        🟡 **Yellow / Green:** Medium contribution  
        🔴 **Red / Warm:** High contribution toward FAKE prediction
        """)

        st.info(
            "💡 **Attribution Note:** Warm regions indicate Xception feature regions contributing positively "
            "toward the model's FAKE logit. This is an attribution map, not proof of pixel-level manipulation.\n\n"
            "**Architecture Context:** In this dual-branch architecture (`Xception + ViT-B/16 with Feature Fusion`), "
            "Xception Grad-CAM targets the final spatial representation (`act4`) directly preceding global pooling and fusion. "
            "Global semantic cues and long-range patch dependencies are processed in parallel by the Vision Transformer branch."
        )
    elif cam_error == "ZERO_CAM":
        target_name = diagnostics.get("target_layer_name", "model.cnn.act4")
        st.info(
            f"ℹ️ **No Positive Spatial Attribution in Xception (`ZERO_CAM`)**\n\n"
            f"Xception `{target_name}` produced no positive localized attribution for this prediction under standard Grad-CAM. "
            f"The model may rely more on the ViT/fusion pathway or on features not localized at this layer.\n\n"
            f"**Technical Context:** Active gradients were captured at `{target_name}`, confirming backward gradient flow "
            f"from the FAKE logit through the classifier and gated fusion into the Xception branch. However, after channel-weighting "
            f"($\\alpha_k$), the spatial feature combination before ReLU is $\\le 0$ across all locations, resulting in an all-zero post-ReLU CAM.\n\n"
            f"- In Grad-CAM, negative pre-ReLU values represent features that vote *against* the FAKE class (evidence for authenticity). "
            f"Displaying negative features as manipulated regions would be scientifically invalid.\n"
            f"- To preserve forensic and scientific integrity, the system does not fabricate, invert, or artificially color this heatmap."
        )
    elif cam_error == "ZERO_GRADIENT":
        st.info(
            "ℹ️ **No Localized Xception Attribution Detected (Gradients ≈ 0)**\n\n"
            "The Xception branch gradients are essentially zero for this image, indicating insufficient localized "
            "spatial attribution in the convolutional filters. The model may rely more on the ViT/fusion pathway "
            "or on features not localized at this layer."
        )
    else:
        st.warning(f"⚠️ Grad-CAM could not be computed. Reason: `{cam_error}`")



# ──────────────────────────────────────────────
# FORENSIC LOG TABLE
# ──────────────────────────────────────────────
st.divider()
st.subheader("📋 Forensic Detection Log")
st.caption(
    "All image submissions (Deepfake and Authentic) are automatically logged with uploader network attribution. "
    "This log can be exported as forensic evidence for verification and cybercrime investigation."
)

log_rows = fetch_log(limit=25)

if log_rows:
    import pandas as pd
    df = pd.DataFrame(log_rows, columns=[
        "Timestamp", "Verdict", "IP Address", "City", "Country",
        "ISP", "Confidence (%)", "Filename", "Image Hash"
    ])
    df["Image Hash"] = df["Image Hash"].str[:16] + "..."

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Verdict": st.column_config.TextColumn(
                "Verdict",
                help="Prediction verdict: FAKE (Manipulated) or REAL (Authentic)"
            ),
            "Confidence (%)": st.column_config.ProgressColumn(
                "Confidence (%)", min_value=0, max_value=100
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
