PRODIGY_ML_02 - Customer Segmentation
Overview
This project implements customer segmentation using the K-means clustering algorithm. It analyzes the 'Mall_Customers.csv' dataset to identify customer groups based on spending habits and demographic data.
Requirements

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle

Installation

Clone the repository or download the files.
Navigate to the project directory: cd PRODIGY_ML_02.
Install dependencies: pip install -r requirements.txt (create a requirements.txt file with the listed packages if needed).
Ensure the 'Mall_Customers.csv' file is placed in the 'data' folder.

Usage

Run the script: python prodigy_ml_02 (2).py.
The script will:
Load and preprocess the data.
Perform K-means clustering with 8 clusters.
Generate plots for WCSS and Silhouette Scores.
Save the trained model to the 'models' folder.



Output

Plots showing WCSS vs K values and Silhouette Scores.
A scatter plot of customer clusters.
Saved model file: 'MallCustomers.pkl' in the 'models' folder.

Notes

The 'data' folder must contain 'Mall_Customers.csv'.
The 'models' folder will be created automatically to store the model file.

