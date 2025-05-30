# Task4
Logistic Regression Binary Classifier
This project demonstrates how to build a binary classifier using logistic regression on a breast cancer dataset. It walks through all key steps of a standard machine learning workflow using Scikit-learn, Pandas, and Matplotlib.

📁 Dataset
The dataset is the Breast Cancer Wisconsin (Diagnostic) dataset.

The target variable diagnosis is binary:
M (Malignant) is encoded as 1
B (Benign) is encoded as 0
Irrelevant columns like id and Unnamed: 32 are removed.

🧪 Workflow Summary

Data Preprocessing
Categorical target encoded using LabelEncoder.
Features are standardized using StandardScaler.
Model Training
Dataset is split into train and test sets (80/20).
A logistic regression model is trained on the standardized training data.
Evaluation Metrics
Confusion matrix
Precision, recall, F1-score (via classification report)
ROC-AUC score
ROC curve visualization
Threshold Tuning
Performance evaluated at thresholds 0.3, 0.5, and 0.7 to show trade-offs between precision and recall.
Sigmoid Function Visualization
A plot of the sigmoid function shows how logistic regression converts linear outputs to probability scores between 0 and 1.

📈 Output
The model achieves high accuracy and ROC-AUC, showing good predictive power.
Visualization of ROC curve and sigmoid function help interpret model decisions and probability outputs.

