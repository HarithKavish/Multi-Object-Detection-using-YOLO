import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import cv2 as cv
import numpy as np
from tensorflow import keras
from keras import datasets, models, layers
from keras.callbacks import Callback
from keras.layers import Input
from ultralytics import YOLO  # Assuming you have a YOLO.py file that contains a YOLO class
from tqdm import tqdm

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TqdmProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        self.total_batches = self.params['epochs'] * self.params['steps']
        self.tqdm_bar = tqdm(total=self.total_batches, desc="Training Progress")

    def on_batch_end(self, batch, logs=None):
        self.tqdm_bar.update(1)

    def on_train_end(self, logs=None):
        self.tqdm_bar.close()

def resize_for_cnn(frame, target_size=(32, 32)):
    return cv.resize(frame, target_size)

def preprocess_for_yolo(frame):
    return cv.resize(frame, (620, 620))  # Assuming YOLOv8 input size is 640x640

def start():
    try:
        training_count = int(input("\nEnter the number of times you want to train the model: "))
    except ValueError:
        print("Please enter a valid integer for the training count.")
        return

    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    training_images, testing_images = training_images / 255.0, testing_images / 255.0

    input_shape = (32, 32, 3)
    input_layer = Input(shape=input_shape)

    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu')(pool2)

    flatten = layers.Flatten()(conv3)
    dense1 = layers.Dense(64, activation='relu')(flatten)
    dense2 = layers.Dense(128, activation='relu', name='dense_layer_2')(dense1)
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("______________________________________________________________________________")
    progress_callback = TqdmProgressCallback()
    model.fit(training_images, training_labels, epochs=training_count, validation_data=(testing_images, testing_labels), verbose=0, callbacks=[progress_callback])
    print("______________________________________________________________________________")
    
    model.save('image_classifier.keras')

    yolov8_model = YOLO('yolov8n-seg.pt')

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        preprocessed_frame_yolo = preprocess_for_yolo(frame)
        preprocessed_frame_cnn = resize_for_cnn(frame)

        prediction = model.predict(np.expand_dims(preprocessed_frame_cnn, axis=0))
        predicted_class = np.argmax(prediction)

        detected_objects = yolov8_model(preprocessed_frame_yolo)
        
        print("______________________________________________________________________________")

        cv.imshow('Frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    

def end():
    print("\nThe program has ended.")

def main():
    while True:
        command = input("Enter 'start' to start the program or 'end' to end the program: ").lower()
        if command == 'start':
            start()
            break  # Exit the loop after starting the program
        elif command == 'end':
            end()
            break  # Exit the loop after ending the program
        else:
            print("\nInvalid command. Please enter either 'start' or 'end'.")
            print("______________________________________________________________________________")

if __name__ == "__main__":
    print("______________________________________________________________________________")
    main()
