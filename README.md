# Hand Gesture Recognition

This project implements a hand gesture recognition system using convolutional neural networks (CNN) and TensorFlow. The model is designed to classify hand gestures based on images, providing a real-time prediction capability via a graphical user interface (GUI).
## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
## Features

- **Model Training**: A CNN model trained on hand gesture images.
- **Real-Time Prediction**: A GUI that allows users to interact with the model for gesture recognition.
- **Model Saving and Loading**: The trained model can be saved to disk and loaded for predictions.

## Dataset

The model is trained using the **Leap Gesture Recognition Dataset**. This dataset consists of various hand gestures captured under different conditions. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data). Ensure to structure the dataset appropriately in your project directory.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Keras
- Tkinter
- Numpy
- Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Hand-Gesture-Recognition
   ```
2. Install required libraries:
   ```bash
   pip install tensorflow opencv-python
   ```
### Usage

1. **Train the model**:
   Run `train.py` to train the model using your dataset. The model will be saved to disk for later use.
   ```bash
   python train.py
   ```
2. **Run the GUI for predictions**:
   Launch the GUI application:
   ```bash
   python gui.py
   ```
3. **Use the GUI Application**:
   Upload an image of a hand gesture and receive a prediction from the model.



