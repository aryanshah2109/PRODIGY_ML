# Hand Gesture Identification using CNN

This project implements a Convolutional Neural Network (CNN) to classify static hand gestures from the **LeapGestRecog** dataset.

---

## Dataset

- Source: [Kaggle - LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- Contains grayscale images of 10 different hand gestures:
  - **Palm**
  - **L**
  - **Fist**
  - **Fist Moved**
  - **Thumb**
  - **Index**
  - **OK**
  - **Palm Moved**
  - **C**
  - **Down**

Images are organized in subfolders with gesture labels.

---

## Project Structure

```
HandGesturesIdentification.py      # Main model training and testing script
/kaggle/input/leapgestrecog/       # Dataset directory (auto-loaded on Kaggle)
```

---

## Features

- Image loading and preprocessing (grayscale, resizing to 64x64)
- CNN with 2 Convolutional layers and Dense layers
- Train/Test split with performance evaluation
- Training and validation accuracy/loss visualization
- Predict individual hand gestures with readable labels

---

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

---

## Model Architecture

- 2 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer
- Dense layer with 128 units
- Dropout layer (0.1)
- Output layer with softmax activation for 10 classes

---

## Usage

Run the training and evaluation:

```python
python HandGesturesIdentification.py
```

Test prediction example:

```python
output = predict_input(X_test[3500])
predicted = labels_names[output]
actual = labels_names[y_test[3500]]
print(f'Predicted: {predicted.upper()}, Actual: {actual.upper()}')
```

---

## Results

- Achieved high accuracy on both training and validation sets
- Visual plots provided for monitoring performance

---

## Future Improvements

- Integrate real-time video prediction using OpenCV
- Improve robustness with hand detection (e.g., MediaPipe)
- Experiment with deeper CNN architectures for enhanced accuracy

