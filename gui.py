from tkinter import *
import numpy as np
from PIL import ImageGrab
import tensorflow as tf

# Load the trained TensorFlow model
model = tf.keras.models.load_model('mnist_digit_model.h5')

# Initialize the main window
window = Tk()
window.title("Handwritten Digit Recognition")

# Variables
lastx, lasty = None, None
l1 = None

# Function to make predictions

def predict_digit():
    global l1

    # Capture the canvas content
    x = window.winfo_rootx() + cv.winfo_x()
    y = window.winfo_rooty() + cv.winfo_y()
    x1 = x + cv.winfo_width()
    y1 = y + cv.winfo_height()

    # Grab the canvas as an image and preprocess it
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
    img_array = img_array / 255.0  # Normalize pixel values

    # Make a prediction
    pred = model.predict(img_array)
    digit = np.argmax(pred)

    # Display the result
    if l1:
        l1.destroy()
    l1 = Label(window, text=f"Digit = {digit}", font=('Algerian', 20))
    l1.place(x=230, y=420)

# Function to clear the canvas
def clear_canvas():
    global l1
    cv.delete("all")
    if l1:
        l1.destroy()

# Function to activate drawing on the canvas
def activate_drawing(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y
    cv.bind('<B1-Motion>', draw_lines)

# Function to draw lines on the canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

# GUI Components
Label(window, text="Handwritten Digit Recognition", font=('Algerian', 25), fg="blue").place(x=35, y=10)

Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_canvas).place(x=120, y=370)
Button(window, text="2. Predict", font=('Algerian', 15), bg="white", fg="red", command=predict_digit).place(x=320, y=370)

# Canvas for drawing
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)
cv.bind('<Button-1>', activate_drawing)

# Window settings
window.geometry("600x500")
window.mainloop()
