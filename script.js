// Browser-based CNN classification using TensorFlow.js (No API!)
const video = document.createElement('video');
video.autoplay = true;
video.width = 640;
video.height = 480;
video.style.display = 'block';
video.style.margin = '40px auto';
document.body.insertBefore(video, document.body.firstChild);

// Create a container for results
const resultDiv = document.createElement('div');
resultDiv.id = 'results';
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
        resultDiv.textContent = 'Loading CNN model...';
        // Load the TensorFlow.js model from GitHub Pages
        cnnModel = await tf.loadLayersModel('https://harithkavish.github.io/Multi-Object-Detection-using-YOLO/tfjs_model/model.json');
        console.log('CNN model loaded successfully');
        resultDiv.textContent = 'CNN model loaded! Starting webcam...';
    } catch (error) {
        console.error('Failed to load CNN model:', error);
        resultDiv.textContent = 'Error loading model: ' + error.message;
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
        resultDiv.textContent = 'Camera access denied or not available: ' + err.message;
        console.error('Camera error:', err);
    }
}

async function processFrames() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.width;
    canvas.height = video.height;

    while (true) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Run CNN classification in browser (100% client-side, no API!)
        const cnnResult = await classifyWithCNN(canvas);
        displayResults(cnnResult);

        await new Promise(r => setTimeout(r, 500)); // Process every 500ms
    }
}

function displayResults(cnnData) {
    let html = '<h3 style="margin-top:0">Browser-based CNN Classification</h3>';

    // Display CNN classification
    if (cnnData) {
        html += `<div style="font-size: 18px; margin: 16px 0;">
                    <strong>Detected:</strong> ${cnnData.class}<br>
                    <strong>Confidence:</strong> ${(cnnData.confidence * 100).toFixed(1)}%
                 </div>`;

        // Add progress bar
        html += `<div style="background: #ddd; border-radius: 4px; overflow: hidden;">
                    <div style="background: #4CAF50; height: 20px; width: ${(cnnData.confidence * 100).toFixed(1)}%; transition: width 0.3s;"></div>
                 </div>`;
    } else {
        html += '<p>Waiting for classification...</p>';
    }

    html += '<p style="margin-top: 16px; font-size: 12px; color: #666;">✓ Running entirely in your browser - no data sent to server!</p>';

    resultDiv.innerHTML = html;
}

window.onload = startCamera;
