import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('traffic_signal_model.keras')

# Define the function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = np.array(image) / 255.0
    return image

# Define the function to predict using the model
def predict(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    color_pred, gender_pred = model.predict(image)
    color_label = np.argmax(color_pred, axis=1)[0]
    gender_label = np.argmax(gender_pred, axis=1)[0]
    colors = ['Blue', 'Red', 'Other']
    genders = ['Male', 'Female']
    return colors[color_label], genders[gender_label]

# Define the function to upload and predict the image
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    try:
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        color, gender = predict(file_path)
        result_label.config(text=f'Car Color: {color}, Gender: {gender}')
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main application window
root = tk.Tk()
root.title("Traffic Signal Prediction")
root.geometry("500x600")

# Add widgets to the window
upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Start the application
root.mainloop()
