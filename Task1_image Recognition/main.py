import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

# Load Haar Cascade XML file for stop sign detection
stop_data = cv2.CascadeClassifier("stop_data.xml")

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect stop signs
        found = stop_data.detectMultiScale(img_gray, minSize=(20, 20))
        if len(found) != 0:
            for (x, y, width, height) in found:
                cv2.rectangle(img_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)

        # Display the image with detected stop signs
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.show()

# Create a simple Tkinter GUI
root = tk.Tk()
root.title("Stop Sign Detector")

# Add a button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

root.mainloop()
