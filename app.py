from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2 as cv
from tensorflow import keras
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# Allow CORS for specific origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://harithkavish.github.io/",
        "https://harithkavish.github.io/Multi-Object-Detection-using-YOLO"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
cnn_model = keras.models.load_model('image_classifier.keras')
yolo_model = YOLO('yolov8n-seg.pt')

# Helper functions
def resize_for_cnn(frame, target_size=(32, 32)):
    return cv.resize(frame, target_size)

def preprocess_for_yolo(frame):
    return cv.resize(frame, (620, 620))

@app.post("/detect-object")
async def detect_object(file: UploadFile = File(...)):
    # Read image from frontend
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    frame = np.array(image)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # CNN prediction
    preprocessed_frame_cnn = resize_for_cnn(frame)
    prediction = cnn_model.predict(np.expand_dims(preprocessed_frame_cnn, axis=0))
    predicted_class = int(np.argmax(prediction))

    # YOLO detection
    preprocessed_frame_yolo = preprocess_for_yolo(frame)
    results = yolo_model(preprocessed_frame_yolo)
    detected_objects = []
    object_counts = {}
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.model.names[cls] if hasattr(yolo_model.model, 'names') else str(cls)
            detected_objects.append(label)
            object_counts[label] = object_counts.get(label, 0) + 1

    return JSONResponse({
        "cnn_class": predicted_class,
        "detected_objects": detected_objects,
        "object_counts": object_counts
    })

@app.get("/health")
def health_check():
    return {"status": "ok"}
