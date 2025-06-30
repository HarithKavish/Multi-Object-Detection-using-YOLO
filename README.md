# YOLO Backend API for Multi-Object Detection (Hugging Face Spaces)

This FastAPI backend provides multi-object detection and image classification using YOLOv8 and a custom CNN. Designed for deployment on Hugging Face Spaces.

## Endpoints

### POST `/detect-object`
- **Description:** Accepts an image file and returns detected objects, their counts, and CNN class prediction.
- **Request:**
  - `file`: Image file (form-data)
- **Response:**
  ```json
  {
    "cnn_class": 3,
    "detected_objects": ["person", "cell phone", ...],
    "object_counts": {"person": 1, "cell phone": 2}
  }
  ```

### GET `/health`
- **Description:** Health check endpoint. Returns `{ "status": "ok" }` if the service is running.

## Deployment (Hugging Face Spaces)
1. Ensure `app.py`, `requirements.txt`, `space.yaml`, `image_classifier.keras`, and `yolov8n-seg.pt` are in the repo.
2. Create a new Space on https://huggingface.co/spaces and select FastAPI as the SDK.
3. Upload or connect your repo. The service will be available at your Hugging Face Space URL.

## Example Request (Python)
```python
import requests
files = {'file': open('your_image.jpg', 'rb')}
r = requests.post('https://your-space-url.hf.space/detect-object', files=files)
print(r.json())
```

---

**Note:** If your model files are large, check Hugging Face’s storage limits (5GB for free tier).
