window.addEventListener("submit", (e) => e.preventDefault());
window.addEventListener("beforeunload", (e) => e.preventDefault());

const video = document.getElementById('preview');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const messageBox = document.querySelector('.message');
const flash = document.querySelector(".flash");

let session;
let isDetecting = false;
let detectionInterval;

// Load ONNX model
async function loadModel() {
  try {
    session = await ort.InferenceSession.create('model.onnx');
    console.log('Model loaded successfully');
  } catch (err) {
    console.error('Failed to load model:', err);
  }
}

// Preprocess image for model
function preprocessImage(frame) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 640;
  canvas.height = 640;
  ctx.drawImage(frame, 0, 0, 640, 640);
  const imageData = ctx.getImageData(0, 0, 640, 640);
  const data = imageData.data;
  const input = new Float32Array(640 * 640 * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    input[j] = data[i] / 255.0;     // R
    input[j + 1] = data[i + 1] / 255.0; // G
    input[j + 2] = data[i + 2] / 255.0; // B
  }
  // Transpose to CHW
  const chw = new Float32Array(3 * 640 * 640);
  for (let c = 0; c < 3; c++) {
    for (let h = 0; h < 640; h++) {
      for (let w = 0; w < 640; w++) {
        chw[c * 640 * 640 + h * 640 + w] = input[h * 640 * 3 + w * 3 + c];
      }
    }
  }
  return new ort.Tensor('float32', chw, [1, 3, 640, 640]);
}

// Sigmoid function
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Detect potholes in frame
async function detectPotholes(frame) {
  if (!session) return 0;
  const input = preprocessImage(frame);
  const outputs = await session.run({ [session.inputNames[0]]: input });
  const outputTensor = outputs[session.outputNames[0]];
  const dims = outputTensor.dims;
  const data = outputTensor.data;

  let numBoxes, stride;
  if (dims.length === 3) {
    numBoxes = dims[1];
    stride = dims[2];
  } else if (dims.length === 2) {
    numBoxes = dims[0];
    stride = dims[1];
  } else {
    console.error('Unexpected output dims:', dims);
    return 0;
  }

  let maxConf = 0;
  for (let i = 0; i < numBoxes; i++) {
    const offset = i * stride;
    const conf = data[offset + 4]; // Use the 5th element as confidence (assuming it's already the final confidence)
    if (conf > maxConf) maxConf = conf;
  }
  return maxConf;
}

// Step 1: Ask permission & show live video
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false }); // Use back camera for mobile
    video.srcObject = stream;
    await loadModel();
  } catch (err) {
    console.error('Camera access denied:', err);
  }
}

// Start detection
function startDetection() {
  if (isDetecting) return;
  isDetecting = true;
  messageBox.innerHTML = 'Detecting potholes...';
  detectionInterval = setInterval(async () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    let conf = await detectPotholes(canvas);
    conf = conf/100
    conf.toFixed(2)
    // console.log(conf)
    if (conf > 0.72) { 
      console.log(conf)
      // Adjusted threshold
      messageBox.innerHTML = `⚠️ Pothole detected! Confidence: ${(conf * 100).toFixed(2)}%`;
      flashScreen();
      vibrate();
    } else {
      messageBox.innerHTML = 'No potholes detected.';
    }
  }, 200); // Process every 100ms
}

// Stop detection
function stopDetection() {
  if (!isDetecting) return;
  isDetecting = false;
  clearInterval(detectionInterval);
  messageBox.innerHTML = 'Detection stopped.';
}

// Step 3: Start detection
startBtn.onclick = (e) => {
  e.preventDefault();
  startDetection();
  startBtn.disabled = true;
  stopBtn.disabled = false;
};

// Step 4: Stop detection
stopBtn.onclick = (e) => {
  e.preventDefault();
  stopDetection();
  startBtn.disabled = false;
  stopBtn.disabled = true;
};

function flashScreen() {
  flash.style.display = 'flex';
  setTimeout(() => {
    flash.style.display = 'none';
  }, 1000);
}

function vibrate() {
  if ('vibrate' in navigator) {
    navigator.vibrate(500); // Vibrate for 500ms
  }
}

// Initialize camera on load
initCamera();

