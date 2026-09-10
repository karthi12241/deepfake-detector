# Comprehensive System Guide: Deepfake Detection, Cryptographic Verification & Explainable AI (Grad-CAM)

This document is a complete, beginner-friendly, and technically rigorous guide to understand how every part of this deepfake detection and digital forensics system works. It is structured so that you can study and explain any component during a project review, exam, or thesis defense.

---

## 1. System at a Glance (The Big Picture)

When a user uploads a face image to the Streamlit web application (`app.py`), the system executes a 3-step pipeline:

```
[1. User Uploads Image]
          │
          ├──► [Step 2: Cryptographic Verification & Audit Logging]
          │      • SHA-256 cryptographic image fingerprinting.
          │      • Tamper-evident hash generation from raw image bytes.
          │      • Appended to forensic_log.json audit trail.
          │
          ├──► [Step 3: Deepfake Neural Network Inference]
          │      • Preprocessed to 224×224 RGB.
          │      • Evaluated by Dual-Branch Model (Xception + ViT-B/16).
          │      • Gated Feature Fusion merges convolutional & transformer cues.
          │      • Outputs single scalar logit -> sigmoid -> probability.
          │      • Decision threshold at 0.5: FAKE vs REAL.
          │
          └──► [Step 4: Explainable AI (Grad-CAM)]
                 • Only computed for FAKE predictions (to explain manipulation cues).
                 • Targets Xception's final spatial representation (model.cnn.act4).
                 • Computes backward gradients from the scalar FAKE logit.
                 • Generates heatmaps (Warm = positive contribution to FAKE).
                 • Robust min-max normalization with transparent zero-CAM messaging.
```

---

## 2. Deep Learning Model Architecture

### 2.1 Why a Dual-Branch Architecture?
Deepfakes exhibit two distinct types of artifacts:
1. **Low-level spatial/blending artifacts:** Blurring, color discrepancies, boundary edges, and texture smoothing.
2. **High-level semantic artifacts:** Inconsistent facial symmetry, unnatural eye gaze, distorted ear lobes, and lighting inconsistencies.

To capture both simultaneously, the architecture combines:
- **CNN Branch (Legacy Xception):** Outstanding at detecting local high-frequency textures and blending borders across receptive fields.
- **Transformer Branch (ViT-B/16):** Divides the image into $16 \times 16$ non-overlapping patches and computes global self-attention across the whole face, excelling at long-range structural dependencies.

### 2.2 Gated Feature Fusion (GFF)
Instead of simply concatenating the features, the model uses an adaptive gating mechanism:
1. `cnn_features` ($2048$-dim) are projected to $512$-dim via LayerNorm + Linear + GELU + Dropout.
2. `vit_features` ($768$-dim) are projected to $512$-dim via LayerNorm + Linear + GELU + Dropout.
3. A gating network computes an attention weight vector $G \in (0, 1)^{512}$ via Sigmoid:
   $$\text{Gate} = \sigma(W_g [\text{proj}_{\text{cnn}} \,\|\, \text{proj}_{\text{vit}}] + b_g)$$
4. The gated combination is:
   $$\text{Fused} = G \odot \text{proj}_{\text{cnn}} + (1 - G) \odot \text{proj}_{\text{vit}}$$
5. The classifier receives all three: $[\text{proj}_{\text{cnn}} \,\|\, \text{proj}_{\text{vit}} \,\|\, \text{Fused}] \in \mathbb{R}^{1536}$.

### 2.3 Single Scalar Logit & Classification
- Unlike standard models that output 2 numbers `[logit_real, logit_fake]`, this model outputs **one single scalar number** ($y \in \mathbb{R}$):
   $$\text{Probability} = \sigma(y) = \frac{1}{1 + e^{-y}}$$
- If $\text{Probability} \ge 0.5$, verdict is **FAKE**.
- If $\text{Probability} < 0.5$, verdict is **REAL**.
- **Why this matters for Grad-CAM:** There is no "class index 1". The single output scalar $y$ directly represents evidence for the FAKE class.

---

## 3. Cryptographic Image Verification & Audit Trail (`forensic_log.json`)

### 3.1 Tamper-Evident SHA-256 Fingerprinting
In digital forensics and legal evidentiary chains, proving that an analyzed image has not been altered or substituted is paramount.
- Every uploaded image has its exact raw binary byte sequence digested using the cryptographic SHA-256 algorithm:
  $$\text{Fingerprint} = \text{SHA256}(\text{ImageBytes})$$
- Because cryptographic hash functions possess the *avalanche effect*, modifying even a single pixel or byte flips approximately 50% of the output bits, producing a completely different hash string.
- This creates an immutable digital fingerprint identifying the exact submitted artifact.

### 3.2 Audit Log Structure
Every submission (both FAKE and REAL) is appended to `forensic_log.json`:
- **Timestamp:** Exact date and time of analysis (`YYYY-MM-DD HH:MM:SS`).
- **Verdict:** Model classification (`FAKE` or `REAL`).
- **Confidence:** Percentage confidence score (e.g. `97.5%`).
- **Image Hash:** Full 64-character SHA-256 hexadecimal digest.
- **Filename:** Original uploaded file name.

### 3.3 CSV Export for Reporting
The application displays the most recent 25 detection records in an interactive table and provides an instant one-click CSV export button (`⬇️ Export Log as CSV`) for forensic reporting and documentation.

---

## 4. Explainable AI (Grad-CAM) Deep Dive

### 4.1 What is Grad-CAM in Simple Words?
Grad-CAM (Gradient-weighted Class Activation Mapping) asks the neural network:
> *"Which specific parts of this face image made you think it was a deepfake?"*

It looks at the final convolutional feature maps before pooling and checks their mathematical gradients with respect to the model's final FAKE decision.

---

### 4.2 The 5 Mathematical Steps of Grad-CAM

Let $A^k$ be the $k$-th feature map of the target layer (dimensions $H \times W$), and $y$ be the model's scalar FAKE logit:

1. **Step 1: Captured Activations ($A^k_{i,j}$)**
   - The forward pass values at spatial position $(i, j)$ in feature map $k$.

2. **Step 2: Captured Gradients ($G^k_{i,j}$)**
   - The backward pass derivatives flowing from the FAKE logit:
     $$G^k_{i,j} = \frac{\partial y}{\partial A^k_{i,j}}$$

3. **Step 3: Channel Weights ($\alpha_k$)**
   - Global average pooling of gradients over all spatial coordinates ($Z = H \times W$):
     $$\alpha_k = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y}{\partial A^k_{i,j}}$$
   - $\alpha_k > 0$: Feature map $k$ votes **for** the image being FAKE.
   - $\alpha_k < 0$: Feature map $k$ votes **against** the image being FAKE (evidence for authenticity).

4. **Step 4: Pre-ReLU Spatial Weighted Sum ($S_{i,j}$)**
   - Linear combination of all feature maps weighted by their importance:
     $$S_{i,j} = \sum_{k=1}^{C} \alpha_k A^k_{i,j}$$

5. **Step 5: Post-ReLU CAM ($L_{i,j}$)**
   - Rectification using ReLU:
     $$L_{i,j} = \max(0, S_{i,j})$$
   - Any spatial location where the combined evidence is non-positive is clipped to zero.

---

### 4.3 Why Target `act4` Instead of `conv4`?

In `legacy_xception`, the exit flow consists of:
$$\mathbf{conv4} \text{ (SeparableConv2d)} \longrightarrow \mathbf{bn4} \text{ (BatchNorm2d)} \longrightarrow \mathbf{act4} \text{ (ReLU)} \longrightarrow \text{GlobalAvgPool} \longrightarrow \text{Gated Fusion}$$

#### The Root Cause of the "All-Zero CAM at `conv4`":
1. `conv4` is an **unrectified** convolution. Its raw values $A^k_{i,j}$ are frequently negative (mean $\approx -0.31$ to $-0.48$).
2. When multiplied by channel weights $\alpha_k$, positive gradients combined with negative activations produce negative values.
3. The spatial sum $S_{i,j} = \sum \alpha_k A^k_{i,j}$ frequently ended up $\le 0$ across the entire $7 \times 7$ grid.
4. When $\text{ReLU}$ was applied, $\max(0, S_{i,j}) = 0$ everywhere, resulting in an all-zero map ($\min=0, \max=0, \text{std}=0$).

#### Why `act4` is the Deterministic Primary Target:
- **True Final Representation:** `act4` is the exact output tensor that enters global average pooling to create the CNN feature vector fed to the gated fusion.
- **Non-Negative Activations:** Because `act4` is post-ReLU ($A \ge 0$), channels with positive importance ($\alpha_k > 0$) produce strictly positive spatial contributions ($\alpha_k A^k \ge 0$).
- **High Spatial Contrast:** On test benchmarks (e.g. Tom Hanks face), `act4` produces over $8\times$ higher peak attribution contrast ($2.31 \times 10^{-3}$) than `conv4` ($0.28 \times 10^{-3}$).
- **Scientific Reproducibility:** Targeting `act4` deterministically avoids heuristic "layer shopping" simply to find a colorful image.

---

### 4.4 Why Removing ReLU or Inverting Signs is Invalid
If $S_{i,j} < 0$, that region contains features that **reduced** the FAKE score (i.e. made the model think the image looks real/authentic).
- If you removed ReLU or inverted negative numbers to positive, you would highlight authentic regions and falsely present them to a judge or user as "manipulated regions".
- Preserving the ReLU ensures forensic honesty: only features that genuinely increased the fake logit are highlighted.

---

### 4.5 The Three Attribution Statuses

| Status | Condition | Meaning | UI Behavior |
| :--- | :--- | :--- | :--- |
| **`VALID_CAM`** | Non-zero gradients & positive post-ReLU spatial variance. | Meaningful localized spatial features in Xception contributed to the FAKE verdict. | Displays color overlay (Red/Warm = strong contribution; Blue = low). |
| **`ZERO_CAM`** | Active gradients exist ($|G_k| > 10^{-7}$), but pre-ReLU sum $\le 0$ everywhere. | Active gradient flow exists, but the Xception branch contains no positive spatial features for the FAKE class. | Transparently explains that the model may rely more on the ViT branch or non-localized cues. Does **not** fabricate a heatmap. |
| **`ZERO_GRADIENT`** | Gradients are negligible ($|G_k| \le 10^{-7}$). | Gradients from the classifier did not flow back to this branch. | Explains that Xception convolutional filters had no gradient attribution. |

---

## 5. How to Run, Test, and Verify the System

### 5.1 Syntax and Compilation Verification
To verify that all Python modules compile cleanly without syntax errors:
```bash
python3 -m py_compile app.py test.py
```
*(Exit code 0 confirms clean compilation).*

### 5.2 Running the Grad-CAM Diagnostic Verification Script
To test all candidate images (Strongly FAKE, Borderline, and Authentic) and verify numerical ranges for both `act4` and `conv4`:
```bash
/Users/sharonkv/projects/deepfake1/.venv/bin/python -c "
import numpy as np, torch
from PIL import Image
from test import build_model, make_transform, unpack_checkpoint
import app

ckpt = torch.load('best.pt', map_location='cpu', weights_only=False)
state_dict, cfg = unpack_checkpoint(ckpt)
model = build_model(cfg, state_dict)
model.load_state_dict(state_dict)
model.eval()

transform = make_transform(224)
img = Image.open('WhatsApp Image 2026-09-10 at 18.57.38.jpeg').convert('RGB')
t = transform(img).unsqueeze(0)

overlay, diag, err = app.get_gradcam(model, t, img)
print('Target Layer:', diag['target_layer_name'])
print('Status:', diag['status'])
print('CAM Max:', diag['post_relu_max'], 'Std:', diag['post_relu_std'])
"
```

### 5.3 Launching the Application
To run the Streamlit forensic application:
```bash
/Users/sharonkv/projects/deepfake1/.venv/bin/streamlit run app.py
```

---

## 6. Academic & Viva Quick Reference (Q&A Cheat Sheet)

**Q1: What is the purpose of Gated Feature Fusion (GFF)?**  
*Answer:* Rather than assuming CNN and ViT features are equally reliable for every face, GFF dynamically assigns attention weights ($0$ to $1$) to each feature dimension based on the input. If texture artifacts are prominent, the gate prioritizes Xception. If spatial proportions are distorted, it prioritizes ViT.

**Q2: Why does the model use a single output logit instead of two classes?**  
*Answer:* In binary classification, $P(\text{FAKE}) = \sigma(\text{logit})$ and $P(\text{REAL}) = 1 - P(\text{FAKE})$. Having 1 scalar logit is mathematically minimal, prevents redundant parameters, and provides an unambiguous directional scalar for computing Grad-CAM derivatives $\frac{\partial y}{\partial A}$.

**Q3: Why was the Grad-CAM output zero for some images on `conv4`?**  
*Answer:* `conv4` is an unrectified convolutional layer with negative mean activations. When multiplied by channel weights, the spatial linear sum was non-positive across all locations, causing ReLU to clip everything to zero. Moving to `act4` (the final post-ReLU spatial feature map feeding global pooling) resolves this because activations are strictly non-negative.

**Q4: Does a red region on Grad-CAM prove that those specific pixels were swapped?**  
*Answer:* No. Grad-CAM is a model attribution method that visualizes which spatial feature regions positively influenced the model's decision. It is not pixel-level ground-truth segmentation. The UI explicitly includes this disclaimer.

**Q5: How is evidentiary integrity ensured without IP tracking?**  
*Answer:* Evidentiary integrity is guaranteed cryptographically using SHA-256 digital fingerprinting. Every uploaded image's raw byte payload is hashed to produce an immutable 64-character fingerprint logged alongside detection verdicts, confidence percentages, and timestamps in `forensic_log.json`. This provides tamper-evident proof of exactly which image was evaluated.
