window.addEventListener("submit", (e) => e.preventDefault());
window.addEventListener("beforeunload", (e) => e.preventDefault());

const video = document.getElementById('preview');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const downloadLink = document.getElementById('downloadLink');

let mediaRecorder;
let recordedChunks = [];

// Step 1: Ask permission & show live video
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;

    // Step 2: Setup MediaRecorder
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    };

    mediaRecorder.onstop = uploadVideo; // when recording stops → upload

  } catch (err) {
    console.error('Camera access denied:', err);
  }
}
async function uploadVideo() {
  const blob = new Blob(recordedChunks, { type: "video/mp4" });
  const formData = new FormData();
  formData.append("video", blob, "recording.mp4");

  try {
    console.log("⏫ Uploading video...");

    const response = await fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });

    console.log("📩 Raw response:", response);

    // Read response safely
    const text = await response.text();
    console.log("📦 Raw response text:", text);

    // Try to parse JSON
    let result;
    try {
      result = JSON.parse(text);
      console.log("✅ Parsed result:", result);
    } catch {
      console.warn("⚠️ Response is not JSON:", text);
    }

  } catch (err) {
    console.error("❌ Upload failed:", err);
  }
}
// Step 3: Start recording
startBtn.onclick = (e) => {
  e.preventDefault();
  recordedChunks = [];
  mediaRecorder.start();
  startBtn.disabled = true;
  stopBtn.disabled = false;
};

// Step 4: Stop recording
stopBtn.onclick = (e) => {
  e.preventDefault();
  mediaRecorder.stop();
  startBtn.disabled = false;
  stopBtn.disabled = true;
};

// Initialize camera on load
  initCamera();

