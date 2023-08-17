# Heart Attack Prediction

This project uses machine learning techniques to predict the likelihood of a heart attack based on various features. The code is implemented in Python and utilizes libraries like pandas, numpy, seaborn, matplotlib, scikit-learn, and keras.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Prediction](#model-training-and-prediction)
- [Conclusion](#conclusion)

## Introduction

The `Heart_Attack_prediction.ipynb` notebook aims to predict the occurrence of heart attacks using a variety of machine learning algorithms. The dataset includes information about various health indicators and risk factors that could contribute to heart disease.

## Setup

To run the code in the notebook, you'll need:

- Python (3.6+ recommended)
- Required libraries (pandas, numpy, seaborn, matplotlib, scikit-learn, keras)

You can install the required libraries using the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn keras tensorflow
```

## Exploratory Data Analysis (EDA)
The notebook begins with exploratory data analysis to gain insights into the dataset. Various visualizations are used to understand the distribution of variables and their potential impact on heart disease.

- Count plot of gender distribution
- Pie chart of heart failure distribution
- Histograms of numerical variables
- Diagnostic plots for various features
- Cross-tabulation and bar plot of heart disease frequency for different age groups
- Count plot of heart disease distribution based on gender

## Data Preprocessing

The data preprocessing steps include:

- Handling outliers using the IQR method
- Imputing missing values using median
- Label encoding categorical variables
- Scaling numerical variables using StandardScaler
- Splitting the dataset into training and test sets

## Model Training and Prediction
The notebook employs different machine learning algorithms for heart attack prediction:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Support Vector Machine (SVM)

The accuracy of each model is evaluated using accuracy scores and confusion matrices.

## Model Comparison

The notebook concludes with a comparison of the accuracy achieved by each model.

| Estimators             | Accuracy |
|------------------------|----------|
| Logistic Regression    | 0.86     |
| K-Nearest Neighbor     | 0.86     |
| Decision Tree          | 0.77     |
| Support Vector Machine | 0.88     |

The Support Vector Machine model outperformed the other models in predicting heart disease.

## Conclusion
The project demonstrates the application of machine learning techniques to predict heart attacks. The choice of the best algorithm may depend on the specific characteristics of the dataset and the goals of the analysis.


