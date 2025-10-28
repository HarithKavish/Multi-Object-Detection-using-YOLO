// Multi-Object Detection using YOLO API and TensorFlow.js CNN
const video = document.createElement('video');
video.autoplay = true;
video.width = 640;
video.height = 480;
video.style.display = 'block';
video.style.margin = '40px auto';
document.body.insertBefore(video, document.body.firstChild);

// Create a container for results
const resultDiv = document.createElement('div');
resultDiv.id = 'api-results';
resultDiv.style.maxWidth = '640px';
resultDiv.style.margin = '20px auto';
resultDiv.style.fontFamily = 'monospace';
resultDiv.style.background = '#f8f8f8';
resultDiv.style.padding = '16px';
resultDiv.style.borderRadius = '8px';
resultDiv.style.boxShadow = '0 2px 8px #0001';
document.body.insertBefore(resultDiv, video.nextSibling);

// CIFAR-10 class names
const cifar10Classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'];

let cnnModel = null;

async function loadCNNModel() {
    try {
        // Load the TensorFlow.js model
        cnnModel = await tf.loadLayersModel('https://harithkavish-multi-object-detection-using-yolo.hf.space/tfjs_model/model.json');
        console.log('CNN model loaded successfully');
    } catch (error) {
        console.error('Failed to load CNN model:', error);
    }
}

async function classifyWithCNN(canvas) {
    if (!cnnModel) return null;

    try {
        // Resize to 32x32 for CIFAR-10 CNN
        const tensor = tf.browser.fromPixels(canvas)
            .resizeBilinear([32, 32])
            .toFloat()
            .div(255.0)
            .expandDims(0);

        const prediction = await cnnModel.predict(tensor);
        const probabilities = await prediction.data();
        const predictedClass = probabilities.indexOf(Math.max(...probabilities));
        const confidence = probabilities[predictedClass];

        tensor.dispose();
        prediction.dispose();

        return {
            class: cifar10Classes[predictedClass],
            confidence: confidence
        };
    } catch (error) {
        console.error('CNN classification error:', error);
        return null;
    }
}

async function startCamera() {
    try {
        await loadCNNModel();
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        processFrames();
    } catch (err) {
        alert('Camera access denied or not available.');
    }
}

async function processFrames() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.width;
    canvas.height = video.height;

    while (true) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Run CNN classification in browser
        const cnnResult = await classifyWithCNN(canvas);

        // Send frame to YOLO API for object detection
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));

        try {
            const response = await fetch('https://harithkavish-multi-object-detection-using-yolo.hf.space/api/detect-object', {
                method: 'POST',
                body: JSON.stringify({
                    data: [await blobToBase64(blob)]
                }),
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const data = await response.json();
                const yoloResult = data.data && data.data[0] ? data.data[0] : data;
                displayResults(yoloResult, cnnResult);
            } else {
                resultDiv.textContent = 'API error: ' + response.status;
            }
        } catch (e) {
            resultDiv.textContent = 'Network error: ' + e.message;
        }

        await new Promise(r => setTimeout(r, 200)); // Process every 200ms
    }
}

function displayResults(yoloData, cnnData) {
    let html = '';

    // Display CNN classification
    if (cnnData) {
        html += `<div style="margin-bottom: 16px;">
                    <strong>CNN Classification:</strong><br>
                    ${cnnData.class} (${(cnnData.confidence * 100).toFixed(1)}% confidence)
                 </div>`;
    }

    // Display YOLO object counts
    html += '<div><strong>YOLO Detections:</strong>';
    if (yoloData.object_counts && typeof yoloData.object_counts === 'object') {
        html += '<ul style="margin:8px 0 0 20px">';
        for (const [label, count] of Object.entries(yoloData.object_counts)) {
            html += `<li>${label}: ${count}</li>`;
        }
        html += '</ul>';
    } else {
        html += '<p style="margin:8px 0">No objects detected</p>';
    }
    html += '</div>';

    resultDiv.innerHTML = html;
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

window.onload = startCamera;
