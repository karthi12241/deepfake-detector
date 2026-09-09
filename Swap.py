from pathlib import Path

import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "inswapper_128.onnx"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing face-swap model: {MODEL_PATH}. "
        "Place inswapper_128.onnx in the project folder."
    )

providers = ["CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=-1, det_size=(640, 640))
swapper = get_model(str(MODEL_PATH), providers=providers)

# Paths
src_path = r"C:\Users\HP\Downloads\woo\real_test.jpg"
target_path = r"C:\Users\HP\Downloads\woo\real.jpeg"

# Check files exist (avoids silent failure)
if not Path(src_path).exists():
    print("Source image not found!")
    raise SystemExit(1)

if not Path(target_path).exists():
    print("Target image not found!")
    raise SystemExit(1)

# Do face swap
print("Swapping faces...")
source_image = cv2.imread(src_path)
target_image = cv2.imread(target_path)
source_faces = app.get(source_image)
target_faces = app.get(target_image)

if not source_faces:
    raise RuntimeError(f"No face detected in source image: {src_path}")
if not target_faces:
    raise RuntimeError(f"No face detected in target image: {target_path}")

result = swapper.get(target_image, target_faces[0], source_faces[0], paste_back=True)

# Save output
output_path = r"C:\Users\HP\Downloads\woo\Gated\output.jpg"
cv2.imwrite(output_path, result)

print(f"✅ Done! Saved at: {output_path}")