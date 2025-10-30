// Multi-Object Detection using COCO-SSD (TensorFlow.js)
// Create a container to hold video and canvas together
const videoContainer = document.createElement('div');
videoContainer.style.position = 'relative';
videoContainer.style.width = '640px';
videoContainer.style.height = '480px';
videoContainer.style.margin = '40px auto';
document.body.insertBefore(videoContainer, document.body.firstChild);

const video = document.createElement('video');
video.autoplay = true;
video.width = 640;
video.height = 480;
video.style.display = 'block';
videoContainer.appendChild(video);

// Create canvas for drawing bounding boxes
const canvas = document.createElement('canvas');
canvas.width = 640;
canvas.height = 480;
canvas.style.position = 'absolute';
canvas.style.top = '0';
canvas.style.left = '0';
canvas.style.pointerEvents = 'none';
videoContainer.appendChild(canvas);
const ctx = canvas.getContext('2d');

// Create camera switch button
const cameraSwitchBtn = document.createElement('button');
cameraSwitchBtn.textContent = '🔄 Switch Camera';
cameraSwitchBtn.style.display = 'none';
cameraSwitchBtn.style.margin = '10px auto';
cameraSwitchBtn.style.padding = '10px 20px';
cameraSwitchBtn.style.fontSize = '14px';
cameraSwitchBtn.style.cursor = 'pointer';
cameraSwitchBtn.style.borderRadius = '5px';
cameraSwitchBtn.style.background = '#2563eb';
cameraSwitchBtn.style.color = 'white';
cameraSwitchBtn.style.border = 'none';
cameraSwitchBtn.style.transition = 'background 0.3s ease';
cameraSwitchBtn.onmouseover = () => cameraSwitchBtn.style.background = '#1d4ed8';
cameraSwitchBtn.onmouseout = () => cameraSwitchBtn.style.background = '#2563eb';
document.body.insertBefore(cameraSwitchBtn, videoContainer.nextSibling);

// Create a container for results
const resultDiv = document.createElement('div');
resultDiv.id = 'results';
resultDiv.style.maxWidth = '640px';
resultDiv.style.margin = '20px auto';
resultDiv.style.fontFamily = 'monospace';
resultDiv.style.background = 'var(--card-bg)';
resultDiv.style.color = 'var(--text-color)';
resultDiv.style.padding = '16px';
resultDiv.style.borderRadius = '8px';
resultDiv.style.boxShadow = '0 2px 8px var(--card-shadow)';
resultDiv.style.transition = 'background-color 0.3s ease, color 0.3s ease';
document.body.insertBefore(resultDiv, cameraSwitchBtn.nextSibling);

let cocoSsdModel = null;
let currentStream = null;
let availableCameras = [];
let currentCameraIndex = 0;

async function loadModel() {
    try {
        resultDiv.innerHTML = '<p>🔄 Loading COCO-SSD object detection model...</p><p style="font-size: 12px; color: #666;">This may take a few seconds...</p>';
        cocoSsdModel = await cocoSsd.load();
        console.log('COCO-SSD model loaded successfully');
        resultDiv.innerHTML = '<p>✅ Model loaded! Starting webcam...</p>';
    } catch (error) {
        console.error('Failed to load model:', error);
        resultDiv.innerHTML = '<p style="color: red;">❌ Error loading model: ' + error.message + '</p>';
    }
}

async function detectObjects() {
    if (!cocoSsdModel) return;

    try {
        const predictions = await cocoSsdModel.detect(video);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const objectCounts = {};

        predictions.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;
            const label = prediction.class;
            const score = (prediction.score * 100).toFixed(1);
            objectCounts[label] = (objectCounts[label] || 0) + 1;

            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
            ctx.fillStyle = '#00FF00';
            const labelText = `${label} ${score}%`;
            const textWidth = ctx.measureText(labelText).width;
            ctx.fillRect(x, y - 25, textWidth + 10, 25);
            ctx.fillStyle = '#000000';
            ctx.font = '16px Arial';
            ctx.fillText(labelText, x + 5, y - 7);
        });

        let summaryHTML = '<h3 style="margin-top: 0;">🎯 Multi-Object Detection</h3>';

        if (Object.keys(objectCounts).length > 0) {
            summaryHTML += '<div style="background: white; padding: 12px; border-radius: 4px; margin: 10px 0;">';
            summaryHTML += '<strong>Detected Objects:</strong><br>';
            for (const [className, count] of Object.entries(objectCounts)) {
                summaryHTML += `<span style="display: inline-block; margin: 5px 10px 5px 0; padding: 5px 10px; background: #4CAF50; color: white; border-radius: 3px;">${className}: ${count}</span>`;
            }
            summaryHTML += '</div>';
            summaryHTML += `<p style="font-size: 12px; color: #666;">Total objects detected: ${predictions.length}</p>`;
        } else {
            summaryHTML += '<p>No objects detected in frame</p>';
        }

        summaryHTML += '<p style="font-size: 11px; color: #999;">✓ Running entirely in your browser - no data sent to server!</p>';
        resultDiv.innerHTML = summaryHTML;
    } catch (error) {
        console.error('Detection error:', error);
    }

    requestAnimationFrame(detectObjects);
}

async function getCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableCameras = devices.filter(device => device.kind === 'videoinput');
        console.log(`Found ${availableCameras.length} camera(s)`);

        // Show switch button only if multiple cameras available
        if (availableCameras.length > 1) {
            cameraSwitchBtn.style.display = 'block';
        }
    } catch (error) {
        console.error('Error enumerating cameras:', error);
    }
}

async function switchCamera() {
    // Stop current stream
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    // Switch to next camera
    currentCameraIndex = (currentCameraIndex + 1) % availableCameras.length;
    const deviceId = availableCameras[currentCameraIndex].deviceId;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: deviceId },
                width: 640,
                height: 480
            }
        });

        currentStream = stream;
        video.srcObject = stream;

        const cameraName = availableCameras[currentCameraIndex].label || `Camera ${currentCameraIndex + 1}`;
        console.log(`Switched to: ${cameraName}`);

    } catch (error) {
        console.error('Error switching camera:', error);
        resultDiv.innerHTML = '<p style="color: red;">❌ Failed to switch camera. Please try again.</p>';
    }
}

// Camera switch button event
cameraSwitchBtn.onclick = switchCamera;

async function startCamera() {
    try {
        await loadModel();
        await getCameras();

        resultDiv.innerHTML = '<p>📹 Requesting camera access...</p>';

        const constraints = availableCameras.length > 0 && availableCameras[0].deviceId
            ? { video: { deviceId: { exact: availableCameras[0].deviceId }, width: 640, height: 480 } }
            : { video: { width: 640, height: 480 } };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        currentStream = stream;
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();
            const cameraName = availableCameras[0]?.label || 'Default Camera';
            resultDiv.innerHTML = `<p>✓ Camera connected: ${cameraName}</p><p>Starting detection...</p>`;
            detectObjects();
        };
    } catch (err) {
        console.error('Camera error:', err);
        let errorMessage = '<h3 style="color: #f44336; margin-top: 0;">⚠️ Camera Access Denied or Not Available</h3>';
        errorMessage += '<p><strong>How to fix:</strong></p>';
        errorMessage += '<ol style="text-align: left; font-size: 13px;">';
        errorMessage += '<li>Click the camera icon 📷 in your browser address bar</li>';
        errorMessage += '<li>Select "Allow" for camera access</li>';
        errorMessage += '<li>Refresh the page</li>';
        errorMessage += '</ol>';
        resultDiv.innerHTML = errorMessage;
    }
}

// Start the application
startCamera();
