import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImagePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection")

        # Load the pre-trained model
        self.model = load_model('BrainTumor10EpochsCategorical.h5')

        # Create GUI components
        self.label = tk.Label(root, text="Select an Image:")
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.browse_button = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)

        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

        # Set window size
        self.root.geometry("600x500")


    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_image(file_path)


    def display_image(self, file_path):
        image = cv2.imread(file_path)
        image = cv2.resize(image, (64, 64))  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.image_label.config(image=image)
        self.image_label.image = image

        self.file_path = file_path


    def predict_image(self):
        if hasattr(self, 'file_path'):
            image = cv2.imread(self.file_path)
            image = cv2.resize(image, (64, 64))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            input_img = np.expand_dims(image, axis=0)

            predict_classes = self.model.predict(input_img)
            result = np.argmax(predict_classes, axis=1)
            if result[0] == 0:
                prediction = "No Tumor"
            else:
                prediction = "Tumor Detected"

            self.result_label.config(text=f"Prediction: {prediction}")
        else:
            self.result_label.config(text="Please select an image first.")


    def reset(self):
        self.image_label.config(image="")
        self.result_label.config(text="")
        if hasattr(self, 'file_path'):
            del self.file_path

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictorApp(root)
    root.mainloop()
