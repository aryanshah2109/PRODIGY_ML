# Cat vs Dog Classifier using SVM | PRODIGY\_ML\_03

This project implements a Cat vs Dog Image Classifier using Support Vector Machine (SVM) with raw pixel features. It is part of the PRODIGY\_ML internship tasks focused on traditional machine learning for image classification.

## Project Structure

```
PRODIGY_ML_03/
├── image_classifier_svm.py    # Main Python script
├── kaggle.json                # Kaggle API credentials
├── microsoft-catsvsdogs-dataset/  # Dataset folder
└── README.md                  # Documentation
```

## Dataset

We use the [Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) from Kaggle.

**Download Steps:**

1. Place your `kaggle.json` API key in:

```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```

2. Run in terminal:

```bash
kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset
unzip microsoft-catsvsdogs-dataset.zip -d .
```

## Requirements

```bash
pip install numpy opencv-python scikit-learn matplotlib kaggle
```

## How to Run

1. Download and extract dataset.
2. Run:

```bash
python image_classifier_svm.py
```

The script:

- Loads grayscale images (64x64)
- Flattens them into feature vectors
- Scales the data
- Trains an SVM with GridSearch
- Prints best parameters and accuracy

## Sample Output

```
Best Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy: 58.20%
```

*Note:* Accuracy is limited with raw pixel features. CNN feature extraction is recommended for better results.

## Future Improvements

- Extract CNN features + classify with SVM
- Use feature engineering (e.g., HOG)
- Visualize predictions
- Compare with deep learning models

## Author

Aryan Shah\
[GitHub Profile](https://github.com/aryanshah2109)

