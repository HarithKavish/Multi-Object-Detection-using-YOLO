import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import cv2 as cv
import numpy as np
from tensorflow import keras
from keras import datasets, models, layers
from tqdm import tqdm

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def resize_for_cnn(frame, target_size=(32, 32)):
    return cv.resize(frame, target_size)

def start():
    try:
        training_count = int(input("\nEnter the number of times you want to train the model: "))
    except (ValueError, EOFError):
        print("Using default training count of 5 epochs for automated environment.")
        training_count = 5

    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    training_images, testing_images = training_images / 255.0, testing_images / 255.0

    input_shape = (32, 32, 3)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu', name='dense_layer_2'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("______________________________________________________________________________")
    print(f"Training model for {training_count} epochs...")
    model.fit(training_images, training_labels, epochs=training_count, validation_data=(testing_images, testing_labels), verbose=1)
    print("______________________________________________________________________________")
    
    # Save in both formats
    model.save('image_classifier.keras')
    model.save('image_classifier.h5')  # For TensorFlow.js converter
    print("Model trained and saved as image_classifier.keras and image_classifier.h5")

def main():
    start()

if __name__ == "__main__":
    main()


# Contributors:
# 1) Harith Kavish S
# 2) Sharwan Krishnan P
# 3) Sanjay R