
# Predict Customer Churn

## Overview
This project aims to predict customer churn using machine learning based on customer-related features.

## Dataset
- Customer information (demographics, service usage, etc.)
- Target variable: `Churn` (whether the customer leaves or not)

## Method

### Preprocessing
- One-Hot Encoding for categorical features

### Model
- LightGBM

### Validation
- Stratified K-Fold Cross Validation

### Evaluation Metric
- ROC-AUC

## Result
- ROC-AUC (CV): **0.91589**

## Project Structure
