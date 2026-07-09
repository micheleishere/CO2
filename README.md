# Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-FFCC00?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-2E8B57?style=for-the-badge)

A machine learning project for predicting the presence of heart disease from clinical patient data. The analysis compares interpretable baseline modelling with ensemble-based approaches and includes model evaluation, feature importance, SHAP explainability, and fairness-oriented discussion.

## Overview

Heart disease remains one of the most important public health challenges worldwide. This project investigates whether standard clinical and diagnostic variables can support early risk identification using supervised machine learning.

The workflow is implemented in a Jupyter notebook and follows a complete data science pipeline:

- exploratory data analysis and feature documentation
- target encoding and train-validation-test splitting
- baseline classification with Logistic Regression
- ensemble modelling with Random Forest
- gradient boosting comparison with CatBoost, XGBoost, GradientBoosting, and AdaBoost
- hyperparameter tuning and model comparison
- ROC-AUC, confusion matrix, accuracy, precision, recall, and F1-score evaluation
- interpretability analysis using feature importance and SHAP values

## Key Results

| Model | Final Test Accuracy | ROC-AUC | Notes |
|---|---:|---:|---|
| Logistic Regression | 80.49% | 0.90 | Strong interpretable baseline with good discrimination |
| Random Forest | 87.80% | 0.91 | Best overall generalization on the independent test set |
| CatBoost | 80.49% | 0.90 | Competitive gradient boosting model with SHAP-based interpretation |

Random Forest achieved the strongest final test performance, while Logistic Regression remained valuable as an interpretable benchmark. Across the interpretability analyses, diagnostic variables such as thallium test results, number of vessels observed under fluoroscopy, chest pain type, and maximum heart rate appeared as important predictors.

> This project is for educational and analytical purposes only. It is not a clinical decision system and should not be used for medical diagnosis.

## Dataset

The project uses a heart disease dataset based on the well-known UCI Heart Disease data, obtained here through Kaggle under the title *Predicting Heart Disease Using Clinical Variables*.

The included dataset contains:

- 270 patient records
- 13 clinical and diagnostic predictors
- 1 binary target variable: `Heart Disease`
- target classes: `Presence` and `Absence`

### Features

| Feature | Description |
|---|---|
| `Age` | Patient age in years |
| `Sex` | Biological sex encoded as a categorical variable |
| `Chest pain type` | Type of chest pain experienced |
| `BP` | Resting blood pressure |
| `Cholesterol` | Serum cholesterol level |
| `FBS over 120` | Fasting blood sugar above 120 mg/dl |
| `EKG results` | Resting electrocardiographic results |
| `Max HR` | Maximum heart rate achieved |
| `Exercise angina` | Exercise-induced angina |
| `ST depression` | ST depression induced by exercise |
| `Slope of ST` | Slope of the peak exercise ST segment |
| `Number of vessels fluro` | Number of major vessels observed by fluoroscopy |
| `Thallium` | Thallium stress test result |
| `Heart Disease` | Target label: presence or absence of heart disease |

## Repository Structure

```text
.
|-- Project_CO2_main.ipynb          # Main analysis notebook
|-- Heart_Disease_Prediction.csv    # Dataset used for modelling
|-- environment.yml                 # Conda environment specification
|-- literatur_heartdisease/         # Supporting literature
|-- catboost_info/                  # CatBoost training artifacts
`-- README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd Heart-Disease-Prediction
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate co2_project
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Open `Project_CO2_main.ipynb` and run the notebook from top to bottom.

## Technical Stack

- Python 3.11
- pandas and NumPy for data handling
- matplotlib and seaborn for visualization
- scikit-learn for preprocessing, model training, tuning, and evaluation
- XGBoost and CatBoost for gradient boosting experiments
- SHAP for model explainability
- Jupyter Notebook for interactive analysis

## Methodology

### Exploratory Data Analysis

The notebook begins with a structured exploration of the dataset, including data types, missing values, duplicate checks, class distribution, categorical feature documentation, and visual inspection of clinical variables.

### Preprocessing

The target variable `Heart Disease` is label-encoded for binary classification. The data is split into training, validation, and test sets to support model selection and final unbiased evaluation.

### Modelling

Three main model families are developed and compared:

- Logistic Regression as a transparent baseline
- Random Forest as a robust bagging ensemble
- CatBoost as the selected gradient boosting approach after comparing boosting variants

Hyperparameter tuning is performed for Random Forest and CatBoost, with attention to validation performance and overfitting behavior.

### Explainability and Fairness

The project investigates feature importance and SHAP values to understand model behavior. The discussion also considers the influence of sensitive or clinically contextual variables such as sex and age, highlighting the importance of careful interpretation in medical machine learning.

## Main Takeaways

- Ensemble methods performed better than the linear baseline on final test accuracy.
- Random Forest delivered the strongest generalization performance in this project.
- ROC-AUC values around 0.90 indicate strong discrimination across all final models.
- False negatives remain especially important in a clinical context because missed heart disease cases may delay diagnosis.
- Explainability is essential for medical ML workflows, especially when model decisions are influenced by demographic and diagnostic variables.

## Authors

Ka Men Ho, Luana Aido da Silva, and Michele Pfister

## Acknowledgements

This project was developed as part of a data challenge project on heart disease prediction. The analysis builds on the UCI Heart Disease dataset and supporting scientific literature on machine learning for cardiovascular risk prediction.
