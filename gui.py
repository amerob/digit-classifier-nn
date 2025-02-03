import tkinter as tk
from tkinter import messagebox, Menu
from tkinter import ttk
from ttkthemes import ThemedTk
import numpy as np
import cv2
from PIL import Image, ImageDraw
from keras.models import Sequential
from keras.layers import Dense, Flatten

class RoundedButton(tk.Canvas):
    def __init__(self, master, text, color, command=None, width=120, height=40):
        super().__init__(master, width=width, height=height, 
                        highlightthickness=0, bg='#2C3E50')
        self.command = command
        self.color = color
        self.id = None
        self.create_rounded_rect(0, 0, width, height, radius=20, fill=color)
        self.text = self.create_text(width/2, height/2, text=text, 
                                   fill='white', font=('Arial', 12, 'bold'))
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                x1+radius, y1,
                x2-radius, y1,
                x2-radius, y1,
                x2, y1,
                x2, y1+radius,
                x2, y1+radius,
                x2, y2-radius,
                x2, y2-radius,
                x2, y2,
                x2-radius, y2,
                x2-radius, y2,
                x1+radius, y2,
                x1+radius, y2,
                x1, y2,
                x1, y2-radius,
                x1, y2-radius,
                x1, y1+radius,
                x1, y1+radius,
                x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)

    def on_enter(self, event):
        self.itemconfig(self.text, fill='#ECF0F1')
        self.config(cursor="hand2")

    def on_leave(self, event):
        self.itemconfig(self.text, fill='white')

    def on_click(self, event):
        if self.command:
            self.command()

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognition by Amer O.")
        self.master.geometry("650x750")
        self.master.configure(bg='#2C3E50')
        self.model = self.load_model()
        self.init_drawing_vars()
        self.create_menu()
        self.create_canvas()
        self.create_controls()
        self.create_status_bar()

    def load_model(self):
        try:
            model = Sequential()
            model.add(Flatten(input_shape=(28, 28)))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.load_weights('FFNN-MNIST.h5')
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.master.destroy()

    def init_drawing_vars(self):
        self.img = Image.new('RGB', (500, 500), (255, 255, 255))
        self.img_draw = ImageDraw.Draw(self.img)
        self.count = 0

    def create_menu(self):
        menu_bar = Menu(self.master)
        file_menu = Menu(menu_bar, tearoff=0, bg='#34495E', fg='white')
        file_menu.add_command(label="Save", command=self.save)
        file_menu.add_command(label="Clear", command=self.clear)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        help_menu = Menu(menu_bar, tearoff=0, bg='#34495E', fg='white')
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.master.config(menu=menu_bar)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.master, width=500, height=500, 
                              bg='#FFFFFF', bd=5, relief='ridge',
                              highlightbackground='#2C3E50')
        self.canvas.grid(row=0, column=0, columnspan=4, pady=20, padx=20)
        self.canvas.bind('<B1-Motion>', self.draw)

    def create_controls(self):
        btn_frame = ttk.Frame(self.master)
        btn_frame.grid(row=1, column=0, columnspan=4, pady=10)
        self.btn_save = RoundedButton(btn_frame, "Save", "#27AE60", 
                                    command=self.save, width=120, height=40)
        self.btn_save.grid(row=0, column=0, padx=10)
        self.btn_predict = RoundedButton(btn_frame, "Predict", "#2980B9", 
                                       command=self.predict, width=120, height=40)
        self.btn_predict.grid(row=0, column=1, padx=10)
        self.btn_clear = RoundedButton(btn_frame, "Clear", "#F39C12", 
                                     command=self.clear, width=120, height=40)
        self.btn_clear.grid(row=0, column=2, padx=10)
        self.btn_exit = RoundedButton(btn_frame, "Exit", "#E74C3C", 
                                    command=self.master.quit, width=120, height=40)
        self.btn_exit.grid(row=0, column=3, padx=10)
        self.lbl_status = ttk.Label(self.master, text='PREDICTED DIGIT: NONE', 
                                  font=('Arial', 16, 'bold'), foreground='#ECF0F1',
                                  background='#2C3E50')
        self.lbl_status.grid(row=2, column=0, columnspan=4, pady=10)

    def create_status_bar(self):
        self.status_bar = ttk.Label(self.master, text="Ready", 
                                  relief='sunken', anchor='w',
                                  font=('Arial', 10),
                                  foreground='#ECF0F1',
                                  background='#34495E')
        self.status_bar.grid(row=3, column=0, columnspan=4, 
                           sticky='we', padx=10, pady=5)

    def draw(self, event):
        x, y = event.x, event.y
        x1, y1 = x - 15, y - 15
        x2, y2 = x + 15, y + 15
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.img_draw.ellipse((x1, y1, x2, y2), fill='black')

    def save(self):
        img_array = np.array(self.img)
        img_array = cv2.resize(img_array, (28, 28))
        cv2.imwrite(f'{self.count}.jpg', img_array)
        self.count += 1
        self.status_bar.config(text=f"Image saved as {self.count - 1}.jpg")
        messagebox.showinfo("Save", f"Image saved as {self.count - 1}.jpg")

    def clear(self):
        self.canvas.delete('all')
        self.img = Image.new('RGB', (500, 500), (255, 255, 255))
        self.img_draw = ImageDraw.Draw(self.img)
        self.lbl_status.config(text='PREDICTED DIGIT: NONE')
        self.status_bar.config(text="Canvas cleared")

    def predict(self):
        img_array = np.array(self.img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (28, 28))
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)
        result = self.model.predict(img_array)
        label = np.argmax(result, axis=1)[0]
        self.lbl_status.config(text=f'PREDICTED DIGIT: {label}')
        self.status_bar.config(text=f"Prediction complete: {label}")

    def show_about(self):
        about_text = ("Handwritten Digit Recognition System\n\n"
                     "Version 2.0\n"
                     "Developed by Amer O.\n"
                     "Using Keras Neural Network Model")
        messagebox.showinfo("About", about_text)

if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    app = DigitRecognizerApp(root)
    root.mainloop()
