# Heart-Disease-Risk-Prediction-Analysis
A complete machine learning pipeline for predicting heart disease risk using the UCI Heart Disease dataset. Demonstrates practical application of data science in healthcare screening and risk assessment.
Dataset

Source: UCI Heart Disease Dataset (Kaggle)
Size: 1,025 patient records
Balance: 48.7% no disease, 51.3% with disease
Features: Demographics, clinical measurements, cardiac assessments, diagnostic indicators

Technical Implementation

Algorithm: Logistic Regression (chosen for interpretability in healthcare)
Preprocessing: StandardScaler, missing value analysis, outlier detection
Validation: 70-30 train-test split with stratification
Evaluation: Comprehensive metrics including AUC-ROC, sensitivity, specificity

Results

Testing Accuracy: 81.8%
AUC-ROC Score: 0.925 (excellent discrimination)
Sensitivity: 89.2% (crucial for medical screening)
Specificity: 74.0%
Most Important Features: Chest pain type, sex, vessel blockages, ST depression

Key Insights

Model achieves clinically relevant performance suitable for screening
High sensitivity enables effective identification of at-risk patients
Uses only routine clinical measurements available in most healthcare settings
Results align with established medical knowledge, supporting model validity

Technologies Used

Python: pandas, numpy, scikit-learn, matplotlib, seaborn
Statistical Analysis: scipy.stats for t-tests and chi-square tests
Machine Learning: Logistic regression with proper validation
Data Visualization: Comprehensive EDA and model evaluation plots

Clinical Applications

Primary care screening tool
Risk stratification using standard blood work and ECG
Early detection system for preventive interventions
Decision support for healthcare providers

Repository Structure

├── heart-disease-prediction/
│   ├── analysis.py
│   ├── heart.csv
│   ├── results/
│   │   ├── visualizations/
│   │   └── performance_metrics.txt
│   ├── README.md
│   └── requirements.txt
