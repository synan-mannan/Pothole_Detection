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
    return 1 / (1 + np.exp(-x))

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

        # Open video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        max_conf = 0.0
        frame_index = -1
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Process every nth frame (optional)
            if frame_count % 5 != 0:
                frame_count += 1
                continue

            img = preprocess_image(frame)
            outputs = session.run(None, {input_name: img})[0]

            # Transpose if needed
            if outputs.shape[0] < outputs.shape[1]:
                outputs = outputs.T

            boxes = outputs[:, :4]
            objectness_raw = outputs[:, 4]
            class_scores_raw = outputs[:, 5:]

            # Apply sigmoid
            objectness = sigmoid(objectness_raw)
            class_conf = sigmoid(class_scores_raw)

            # For single-class, just take first column
            if class_conf.shape[1] == 1:
                final_conf = objectness * class_conf[:, 0]
            else:
                final_conf = objectness * np.max(class_conf, axis=1)

            frame_max_conf = float(np.max(final_conf))

            if frame_max_conf > max_conf:
                max_conf = frame_max_conf
                frame_index = frame_count

            frame_count += 1

        cap.release()
        print(max_conf)
        # Decide threshold
        threshold = 0.6  # lower threshold for demo
        if max_conf > threshold:
            message = f"⚠️ Potholes detected (frame {frame_index})"
        else:
            message = "✅ No potholes detected"

        return jsonify({
            "message": message,
            "max_confidence": round(max_conf, 4),
            "frame_index": frame_index
        })

    except Exception as e:
        print("❌ Error during detection:", str(e))
        return jsonify({"error": str(e)}), 500
# ---- Run App ----
if __name__ == "__main__":
    app.run(debug=True)
