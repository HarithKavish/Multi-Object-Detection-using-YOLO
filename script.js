// Blank JS file for custom scripts
// Access the camera, show preview, send frames to API, and display API response
const video = document.createElement('video');
video.autoplay = true;
video.width = 640;
video.height = 480;
video.style.display = 'block';
video.style.margin = '40px auto';
document.body.insertBefore(video, document.body.firstChild);

// Create a container for API results
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

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        sendFrames();
    } catch (err) {
        alert('Camera access denied or not available.');
    }
}

async function sendFrames() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.width;
    canvas.height = video.height;
    while (true) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));

        // Use Gradio API format
        const formData = new FormData();
        formData.append('data', blob);

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
                displayApiResult(data.data && data.data[0] ? data.data[0] : data);
            } else {
                resultDiv.textContent = 'API error: ' + response.status;
            }
        } catch (e) {
            resultDiv.textContent = 'Network error: ' + e.message;
        }
        await new Promise(r => setTimeout(r, 100)); // Send every 100ms
    }
}

function displayApiResult(data) {
    // Display only object counts, without heading
    let html = '';
    if (data.object_counts && typeof data.object_counts === 'object') {
        html += '<ul style="margin:0 0 0 20px">';
        for (const [label, count] of Object.entries(data.object_counts)) {
            html += `<li>${label}: ${count}</li>`;
        }
        html += '</ul>';
    } else {
        html += '-';
    }
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
