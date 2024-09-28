import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from keras.models import load_model

# Load the trained model
model = load_model('hand_gesture_recognition_model.h5')

# Define categories
CATEGORIES = ["01_palm", '02_l', '03_fist', '04_fist_moved', '05_thumb',
              '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    img = img / 255.0
    img = img.reshape(-1, 50, 50, 1)  # Reshape for the model
    return img

# Function to predict the gesture
def predict_gesture():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = preprocess_image(file_path)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)
            gesture = CATEGORIES[predicted_class[0]]
            messagebox.showinfo("Prediction", f"Predicted Gesture: {gesture[3:]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Setting up the GUI
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("300x200")

# Add a button to upload and predict
upload_btn = tk.Button(root, text="Upload Image", command=predict_gesture)
upload_btn.pack(pady=20)

# Start the GUI
root.mainloop()
