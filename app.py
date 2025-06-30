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
try:
    cnn_model = keras.models.load_model('image_classifier.keras')
except Exception as e:
    from tensorflow import keras as keras_build
    from tensorflow.keras import layers
    # Define a simple CNN model structure matching the original
    def build_cnn_model():
        model = keras_build.Sequential([
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu', name='dense_layer_2'),
            layers.Dense(10, activation='softmax', name='dense_1')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    cnn_model = build_cnn_model()
    print('Warning: image_classifier.keras not found or incompatible. Created a new model instead.')

# Patch PyTorch safe globals for YOLOv8 segmentation model loading (PyTorch 2.6+)
try:
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import SegmentationModel
    import torch.nn.modules.container
    from ultralytics.nn.modules import Conv
    add_safe_globals([SegmentationModel, torch.nn.modules.container.Sequential, Conv])
except Exception as e:
    print('Warning: Could not patch torch safe globals for YOLOv8:', e)

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
