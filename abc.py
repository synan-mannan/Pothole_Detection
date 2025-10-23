from flask import Flask, request, jsonify
import cv2, numpy as np, onnxruntime as ort, os
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Create uploads folder if not exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ONNX model once
MODEL_PATH = "best.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def sigmoid(x):
    return 1 / (1 + np.exp(x))

# ---- Utility: Preprocess image ----
def preprocess_image(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# ---- Endpoint: Upload + Detect ----
@app.route("/upload", methods=["POST"])
def detect_potholes():
    try:
        # Save uploaded video
        file = request.files["video"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"📁 Saved uploaded video to: {file_path}")

        # Read video (use first frame for demo)
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return jsonify({"error": "Could not read video frame"}), 400

        # Preprocess and run inference
        img = preprocess_image(frame)
        outputs = session.run(None, {input_name: img})[0]

        # Handle YOLO output: (1, N, 85)
        if outputs.ndim == 3:
            outputs = outputs[0]

        # Extract objectness and class scores
        objectness_raw = outputs[:, 4]
        class_scores_raw = outputs[:, 5:]

        objectness = sigmoid(objectness_raw)
        class_conf = sigmoid(class_scores_raw)
        final_conf = objectness * np.max(class_conf, axis=1)
        max_conf = float(np.max(final_conf))

        print(f"🔍 Max confidence detected: {max_conf:.4f}")

        # Decide threshold
        threshold = 0.5
        if max_conf > threshold:
            message = "⚠️ Potholes detected"
        else:
            message = "✅ No potholes detected"

        return jsonify({
            "message": message,
            "max_confidence": round(max_conf, 4)
        })

    except Exception as e:
        print("❌ Error during detection:", str(e))
        return jsonify({"error": str(e)}), 500

# ---- Run App ----
if __name__ == "__main__":
    app.run(debug=True)
