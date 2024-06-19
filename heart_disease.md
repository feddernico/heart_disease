---
jupyter:
  jupytext:
    cell_metadata_filter: title,-all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
---

### How to Create a Predictive Model for Heart Disease Prediction

**Introduction**

Predictive modeling for heart disease can significantly aid in early detection and treatment. This guide aims to provide a step-by-step approach to creating a predictive model, covering data exploration, feature engineering, and machine learning model building. It is tailored for junior data scientists looking to enhance their skills.

### 1. Introduction

Heart disease is one of the leading causes of death worldwide. By leveraging machine learning techniques, we can create predictive models to identify individuals at risk and facilitate early intervention. This guide walks you through the entire process of building a predictive model using a dataset of patient information and risk factors for heart disease.

### 2. Loading and Understanding the Data

The first step in any data science project is to load the dataset and understand its structure. Here, we use a dataset containing various attributes related to heart disease risk factors.

<!-- #region code -->
import pandas as pd

# Load the dataset
file_path = '/mnt/data/heart_disease_dataset.csv'
heart_disease_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
heart_disease_data.head()
<!-- #endregion -->

### 3. Exploratory Data Analysis (EDA)

EDA is crucial to understand the data and uncover patterns that can inform model building.

#### Data Cleaning
Before diving into analysis, it's crucial to clean the data. This involves handling missing values, correcting data types, and removing duplicates.

<!-- #region code -->
# Check for missing values
heart_disease_data.isnull().sum()

# Drop duplicates
heart_disease_data = heart_disease_data.drop_duplicates()
<!-- #endregion -->

#### Statistical Summary
Generate a statistical summary to understand the distribution and central tendencies of the data.

<!-- #region code -->
# Statistical summary
heart_disease_data.describe()
<!-- #endregion -->

#### Data Visualization
Visualize the data to uncover patterns and relationships between variables.

<!-- #region code -->
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for age
plt.figure(figsize=(10, 6))
sns.histplot(heart_disease_data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = heart_disease_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
<!-- #endregion -->

### 4. Data Preprocessing

#### Handling Missing Values
Identify and handle missing values appropriately. Techniques include imputation, deletion, or using algorithms that support missing values.

<!-- #region code -->
# Impute missing values if necessary
heart_disease_data.fillna(heart_disease_data.mean(), inplace=True)
<!-- #endregion -->

#### Encoding Categorical Variables
Convert categorical variables into numerical values.

<!-- #region code -->
# One-hot encoding for categorical variables
heart_disease_data = pd.get_dummies(heart_disease_data, drop_first=True)
<!-- #endregion -->

#### Feature Scaling
Normalize or standardize the data to ensure all features contribute equally to the model.

<!-- #region code -->
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
features = heart_disease_data.drop('Heart Disease', axis=1)
scaled_features = scaler.fit_transform(features)
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
scaled_data['Heart Disease'] = heart_disease_data['Heart Disease']
<!-- #endregion -->

### 5. Building the Predictive Model

#### Splitting the Data
Divide the dataset into training and testing sets to evaluate model performance.

<!-- #region code -->
from sklearn.model_selection import train_test_split

# Split the data
X = scaled_data.drop('Heart Disease', axis=1)
y = scaled_data['Heart Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
<!-- #endregion -->

#### Model Selection
Choose appropriate machine learning algorithms.

<!-- #region code -->
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Initialize models
log_reg = LogisticRegression()
rf_clf = RandomForestClassifier()
svm_clf = SVC()
<!-- #endregion -->

#### Model Training
Train the selected model(s) on the training dataset.

<!-- #region code -->
# Train Logistic Regression model
log_reg.fit(X_train, y_train)

# Train Random Forest model
rf_clf.fit(X_train, y_train)

# Train SVM model
svm_clf.fit(X_train, y_train)
<!-- #endregion -->

### 6. Evaluating the Model

#### Performance Metrics
Evaluate the model using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC curve.

<!-- #region code -->
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf_clf = rf_clf.predict(X_test)
y_pred_svm_clf = svm_clf.predict(X_test)

# Evaluation
def evaluate_model(y_test, y_pred):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred)}')

print("Logistic Regression Performance:")
evaluate_model(y_test, y_pred_log_reg)

print("Random Forest Performance:")
evaluate_model(y_test, y_pred_rf_clf)

print("SVM Performance:")
evaluate_model(y_test, y_pred_svm_clf)
<!-- #endregion -->

#### Model Validation
Use techniques like cross-validation to validate the model and prevent overfitting.

<!-- #region code -->
from sklearn.model_selection import cross_val_score

# Cross-validation for Logistic Regression
cv_scores = cross_val_score(log_reg, X, y, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')
<!-- #endregion -->

### 7. Conclusion

Creating a predictive model for heart disease involves several steps, from data cleaning and preprocessing to model selection and evaluation. By following this guide, junior data scientists can build a robust model to aid in the early detection of heart disease.

---

This guide provides a comprehensive overview of building a heart disease predictive model. For further improvement, consider exploring feature engineering techniques and experimenting with different algorithms and hyperparameters.
