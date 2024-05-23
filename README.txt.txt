Titanic - Machine Learning from Disaster

This repository contains a complete solution for the Kaggle competition "Titanic: Machine Learning from Disaster" using the XGBoost algorithm. The solution includes data preprocessing, feature engineering, hyperparameter tuning, model training, evaluation, and submission preparation.

Key Steps in the Process:

Data Preprocessing:
Handling missing values for categorical and numerical features.
Feature engineering including family size, name length, and number of parts in the name.
One-hot encoding for categorical variables.

Model Training:
Splitting the data into training and validation sets.
Hyperparameter tuning using GridSearchCV.
Training the XGBoost classifier with the best hyperparameters.

Model Evaluation:
Evaluating the model using accuracy score and classification report.
Generating a confusion matrix and visualizing feature importance.
Prediction and Submission:

Making predictions on the test dataset.
Preparing and saving the submission file for Kaggle.

Files in the Repository:
train.csv: Training dataset.
test.csv: Test dataset.
titanic_notebook.ipynb: Jupyter notebook containing the full solution.

How to Use:
Clone the repository.
Ensure all dependencies are installed (numpy, pandas, xgboost, scikit-learn, matplotlib).
Adjust the train and test file paths.
Run the Jupyter notebook to reproduce the solution and generate predictions for the Titanic dataset.
