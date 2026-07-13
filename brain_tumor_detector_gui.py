import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('brain_tumor_detection.h5')

# Define the CNN model
def predict_tumor():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Make a prediction on the selected image
    img = image.load_img(file_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    # Display the prediction result
    if classes[0] > 0.5:
        result_text = "The image has a tumor."
    else:
        result_text = "The image does not have a tumor."

    # Display the selected image in the main window
    img = Image.open(file_path)
    img = img.resize((512, 512), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    img_label.configure(image=img)
    img_label.image = img

    result_label.configure(text=result_text)

    # Show the "Submit another image" button
    submit_button.pack(side=tk.BOTTOM, pady=20)

# Create the GUI window
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("800x600")

# Add a label to display the selected image and the prediction result
img_label = tk.Label(root)
img_label.pack(pady=20)

result_label = tk.Label(root, font=("Arial Bold", 20))
result_label.pack()

# Add a button to open the file dialog and run the prediction function
button = tk.Button(root, text="Select an image", command=predict_tumor, font=("Arial Bold", 20))
button.pack(pady=50)

# Add a button to submit another image
submit_button = tk.Button(root, text="Submit another image", command=predict_tumor, font=("Arial Bold", 20))

root.mainloop()