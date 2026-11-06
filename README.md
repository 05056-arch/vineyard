# Red Wine Quality Classification Project

## a. Problem Statement
The goal of this project is to build an accurate predictive model that classifies red wine quality as either 'Good' (Quality score ≥ 6) or 'Not Good' (Quality score < 6) based on chemical properties. Wine certification and quality control are traditionally subjective; automated machine learning provides objective quality indexing for vineyards.

## b. Dataset Description
We use the official **UCI Red Wine Quality dataset**, consisting of 1,599 instances of red portuguese 'Vinho Verde' wine.
- **Features (Input):** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.
- **Target (Output):** Binary outcome derived from the original `quality` rating score (0 to 10).

## c. GitHub Repository Link
*GitHub Repository Link: https://github.com/05056-arch/vineyard.git

## d. Models Used:
We trained and validated five distinct classification paradigms on standard preprocessed metrics:

### Comparison Table with the evaluation metrics:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 74.3% | 0.812 | 76.0% | 75.4% | 75.7% | 0.483 |
| **Decision Tree** | 73.1% | 0.730 | 74.9% | 74.9% | 74.9% | 0.461 |
| **KNN** | 71.2% | 0.768 | 73.1% | 73.7% | 73.4% | 0.423 |
| **Naive Bayes** | 71.2% | 0.796 | 76.5% | 66.7% | 71.3% | 0.424 |
| **Support Vector Machine** | **75.3%** | **0.825** | **77.2%** | **76.6%** | **76.9%** | **0.503** |

### Observations on the performance of each model:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provides a strong, balanced baseline with excellent linear decision boundaries. |
| **Decision Tree** | Quick to train, but shows slight over-fitting tendencies compared to ensemble/linear counterparts. |
| **KNN** | Highly sensitive to scaling and neighborhood definitions; shows moderate overall descriptive power. |
| **Naive Bayes** | Assumes strong feature independence; has slightly lower recall but maintains high precision values. |
| **Support Vector Machine** | The best-performing classifier, achieving the highest Accuracy (75.3%), AUC (0.83), and F1 Score (76.9%). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** wins across all core classification metrics. |
