 Handwritten Digit Classification with Neural Networks

Welcome to the **Handwritten Digit Classification** project! This project demonstrates how to classify handwritten digits using a deep learning model and features an intuitive **GUI** for real-time drawing and classification.

---

## Features

- **Deep Learning**: A neural network trained on the **MNIST dataset** for digit recognition.
- **Interactive GUI**: Draw digits on a canvas and get instant predictions.
- **Customizable**: Experiment with different model architectures and hyperparameters to improve results.

---

## Prerequisites

Ensure the following dependencies are installed:

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `tkinter` (for the GUI)

Install them using:

```bash
pip install tensorflow keras numpy matplotlib
```

**Note**: `tkinter` is typically included with Python, but verify its availability based on your system.

---

## Project Structure

- `model_training.ipynb` - Code for training and evaluating the neural network.
- `gui.py` - Script for the interactive GUI.

---

## Getting Started

### 1. **Train the Model**

Use `model_training.ipynb` to:

- Load and preprocess data.
- Define the model architecture.
- Train the model on the MNIST dataset.
- Save the trained model for later use.

### 2. **Run the GUI**

After training the model, launch the GUI with:

```bash
python gui.py
```

---

## Using the GUI

1. **Draw**: Use the canvas to draw a digit (0-9).
2. **Classify**: Click "Classify" to see the model's prediction.
3. **Clear**: Reset the canvas to try again.

---

## Model Performance

The default neural network achieves high accuracy on the MNIST dataset. Experiment with the architecture and hyperparameters to further enhance performance.

---

## Future Enhancements

- **Model Optimization**: Improve accuracy with advanced architectures.
- **Enhanced GUI**: Add features like eraser tools and advanced visualization of predictions.

---

## 🤝 Contributing

Contributions are welcome! Fork the repository to add features, fix bugs, or enhance functionality.
