# TODO: Make Realtime Pothole Detection for Mobile

- [x] Update index.html: Add onnxruntime-web CDN script
- [x] Update index.js: Load ONNX model asynchronously, process video stream frames continuously (every 100ms), run inference, check confidence > 0.3, trigger flash and vibration alert if detected
- [x] Update index.css: Enhance flash styles for better visibility on mobile
- [x] Test on mobile browser: Ensure camera access, model loading, detection, and alerts work (Instructions: Open index.html in mobile browser, grant camera permissions, start detection. Backend no longer needed for realtime.)
