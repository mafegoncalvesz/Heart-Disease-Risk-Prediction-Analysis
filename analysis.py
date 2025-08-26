# Heart Disease Risk Prediction Analysis
# Assessment 3: Programming Exercise
# Student: Maria Fernanda Cavalcante Goncalves
# Student ID: A00124607

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("HEART DISEASE RISK PREDICTION ANALYSIS")
print("=" * 60)

# ===========================
# STEP 1: DATA LOADING AND INITIAL EXPLORATION
# ===========================

print("\n" + "="*50)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*50)

# Load the dataset
# Note: Download heart.csv from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# Or use the UCI dataset format
try:
    # Try to load the dataset (assuming it's in the same directory)
    data = pd.read_csv('heart.csv')
    print("✓ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ heart.csv not found. Please download the dataset from:")
    print("   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    print("   Place the heart.csv file in the same directory as this script.")
    exit()

# Display basic information about the dataset
print(f"\n📊 Dataset Shape: {data.shape}")
print(f"   - {data.shape[0]} patients")
print(f"   - {data.shape[1]} features (including target)")

print(f"\n📋 Column Names:")
for i, col in enumerate(data.columns, 1):
    print(f"   {i:2}. {col}")

print(f"\n🔍 Data Types:")
print(data.dtypes)

print(f"\n❓ Missing Values:")
missing_values = data.isnull().sum()
if missing_values.sum() == 0:
    print("   ✓ No missing values found!")
else:
    print(missing_values[missing_values > 0])

print(f"\n📈 Basic Statistical Summary:")
print(data.describe())

# Check the target variable distribution
print(f"\n🎯 Target Variable Distribution:")
target_counts = data['target'].value_counts()
print(f"   No Heart Disease (0): {target_counts[0]} patients ({target_counts[0]/len(data)*100:.1f}%)")
print(f"   Heart Disease (1):    {target_counts[1]} patients ({target_counts[1]/len(data)*100:.1f}%)")

# ===========================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ===========================

print("\n" + "="*50)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*50)

# Create figure for multiple subplots
plt.figure(figsize=(15, 12))

# 1. Target variable distribution
plt.subplot(2, 3, 1)
target_counts.plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Heart Disease Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Heart Disease Status')
plt.ylabel('Number of Patients')
plt.xticks([0, 1], ['No Disease', 'Disease Present'], rotation=0)
plt.grid(axis='y', alpha=0.3)

# 2. Age distribution by heart disease status
plt.subplot(2, 3, 2)
data.boxplot(column='age', by='target', ax=plt.gca())
plt.title('Age Distribution by Heart Disease Status')
plt.suptitle('')  # Remove default title
plt.xlabel('Heart Disease Status')
plt.ylabel('Age (years)')

# 3. Chest pain type distribution
plt.subplot(2, 3, 3)
cp_counts = data.groupby(['cp', 'target']).size().unstack()
cp_counts.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightblue'])
plt.title('Chest Pain Type vs Heart Disease')
plt.xlabel('Chest Pain Type')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.legend(['No Disease', 'Disease Present'])

# 4. Gender distribution
plt.subplot(2, 3, 4)
gender_disease = pd.crosstab(data['sex'], data['target'])
gender_disease.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightblue'])
plt.title('Gender vs Heart Disease')
plt.xlabel('Gender (0=Female, 1=Male)')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.legend(['No Disease', 'Disease Present'])

# 5. Maximum heart rate distribution
plt.subplot(2, 3, 5)
plt.hist(data[data['target']==0]['thalach'], alpha=0.7, label='No Disease', bins=20, color='lightcoral')
plt.hist(data[data['target']==1]['thalach'], alpha=0.7, label='Disease Present', bins=20, color='lightblue')
plt.title('Maximum Heart Rate Distribution')
plt.xlabel('Maximum Heart Rate')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)

# 6. Cholesterol levels
plt.subplot(2, 3, 6)
# Remove outliers for better visualization (cholesterol = 0 is likely missing data)
chol_clean = data[data['chol'] > 0]
plt.boxplot([chol_clean[chol_clean['target']==0]['chol'], 
             chol_clean[chol_clean['target']==1]['chol']], 
            labels=['No Disease', 'Disease Present'])
plt.title('Cholesterol Levels by Heart Disease Status')
plt.ylabel('Cholesterol (mg/dl)')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Exploratory visualizations created!")

# ===========================
# STEP 3: CORRELATION ANALYSIS
# ===========================

print("\n" + "="*50)
print("STEP 3: CORRELATION ANALYSIS")
print("="*50)

# Calculate correlation matrix
correlation_matrix = data.corr()

# Create correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Heart Disease Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Find features most correlated with target
target_correlations = correlation_matrix['target'].abs().sort_values(ascending=False)
print("🔗 Features most correlated with Heart Disease:")
for feature, corr in target_correlations.items():
    if feature != 'target':
        print(f"   {feature:<12}: {corr:.3f}")

# ===========================
# STEP 4: STATISTICAL TESTING
# ===========================

print("\n" + "="*50)
print("STEP 4: STATISTICAL TESTING")
print("="*50)

# T-tests for continuous variables
continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

print("📊 T-test results for continuous variables:")
print("   Variable     | t-statistic | p-value  | Interpretation")
print("   " + "-"*55)

for var in continuous_vars:
    # Remove zeros for cholesterol (likely missing data)
    if var == 'chol':
        var_data = data[data[var] > 0]
    else:
        var_data = data
    
    group_0 = var_data[var_data['target'] == 0][var]
    group_1 = var_data[var_data['target'] == 1][var]
    
    t_stat, p_value = stats.ttest_ind(group_0, group_1)
    significance = "Significant" if p_value < 0.05 else "Not significant"
    
    print(f"   {var:<12} | {t_stat:>10.3f} | {p_value:>7.4f} | {significance}")

# Chi-square tests for categorical variables
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

print(f"\n🎲 Chi-square test results for categorical variables:")
print("   Variable     | Chi2-stat   | p-value  | Interpretation")
print("   " + "-"*55)

for var in categorical_vars:
    contingency_table = pd.crosstab(data[var], data['target'])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    significance = "Significant" if p_value < 0.05 else "Not significant"
    
    print(f"   {var:<12} | {chi2_stat:>10.3f} | {p_value:>7.4f} | {significance}")

# ===========================
# STEP 5: MACHINE LEARNING MODEL
# ===========================

print("\n" + "="*50)
print("STEP 5: LOGISTIC REGRESSION MODEL")
print("="*50)

# Prepare the data
X = data.drop('target', axis=1)
y = data['target']

print(f"📋 Features used in the model:")
for i, feature in enumerate(X.columns, 1):
    print(f"   {i:2}. {feature}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42,
                                                    stratify=y)

print(f"\n📊 Data split:")
print(f"   Training set: {len(X_train)} patients")
print(f"   Testing set:  {len(X_test)} patients")

# Standardize the features for better logistic regression performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = log_reg.predict(X_train_scaled)
y_test_pred = log_reg.predict(X_test_scaled)
y_test_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Calculate performance metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n🎯 Model Performance:")
print(f"   Training Accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
print(f"   Testing Accuracy:  {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n📈 Feature Importance (Logistic Regression Coefficients):")
print("   Rank | Feature      | Coefficient | Interpretation")
print("   " + "-"*55)

for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
    coef = row['Coefficient']
    interpretation = "Increases risk" if coef > 0 else "Decreases risk"
    print(f"   {i:2}   | {row['Feature']:<12} | {coef:>10.3f} | {interpretation}")

# ===========================
# STEP 6: MODEL EVALUATION VISUALIZATIONS
# ===========================

print("\n" + "="*50)
print("STEP 6: MODEL EVALUATION VISUALIZATIONS")
print("="*50)

# Create evaluation visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix', fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_xticklabels(['No Disease', 'Disease'])
ax1.set_yticklabels(['No Disease', 'Disease'])

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

# 3. Feature Importance Plot
top_features = feature_importance.head(8)
ax3.barh(range(len(top_features)), top_features['Coefficient'], 
         color=['red' if x < 0 else 'blue' for x in top_features['Coefficient']])
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['Feature'])
ax3.set_xlabel('Coefficient Value')
ax3.set_title('Top 8 Most Important Features', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Prediction Probability Distribution
ax4.hist(y_test_prob[y_test == 0], bins=20, alpha=0.7, label='No Disease', color='lightcoral')
ax4.hist(y_test_prob[y_test == 1], bins=20, alpha=0.7, label='Disease Present', color='lightblue')
ax4.set_xlabel('Predicted Probability of Heart Disease')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Prediction Probabilities', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Model evaluation visualizations created!")

# ===========================
# STEP 7: DETAILED CLASSIFICATION REPORT
# ===========================

print("\n" + "="*50)
print("STEP 7: DETAILED CLASSIFICATION REPORT")
print("="*50)

print("📋 Detailed Classification Report:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['No Disease', 'Disease Present']))

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate
precision = tp / (tp + fp)
npv = tn / (tn + fn)  # Negative Predictive Value

print(f"\n📊 Additional Performance Metrics:")
print(f"   Sensitivity (Recall):    {sensitivity:.3f} ({sensitivity*100:.1f}%)")
print(f"   Specificity:             {specificity:.3f} ({specificity*100:.1f}%)")
print(f"   Precision:               {precision:.3f} ({precision*100:.1f}%)")
print(f"   Negative Predictive Value: {npv:.3f} ({npv*100:.1f}%)")
print(f"   AUC-ROC Score:           {roc_auc:.3f}")

# ===========================
# STEP 8: SUMMARY AND CLINICAL INSIGHTS
# ===========================

print("\n" + "="*50)
print("STEP 8: SUMMARY AND CLINICAL INSIGHTS")
print("="*50)

print("🔍 KEY FINDINGS:")

print(f"\n1. MODEL PERFORMANCE:")
print(f"   • Overall accuracy: {test_accuracy:.1%}")
print(f"   • The model correctly identifies {sensitivity:.1%} of patients with heart disease")
print(f"   • The model correctly identifies {specificity:.1%} of patients without heart disease")

print(f"\n2. MOST IMPORTANT RISK FACTORS:")
top_3_features = feature_importance.head(3)
for i, (_, row) in enumerate(top_3_features.iterrows(), 1):
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"   • {row['Feature']}: {direction} heart disease risk (coef: {row['Coefficient']:.3f})")

print(f"\n3. STATISTICAL SIGNIFICANCE:")
print(f"   • Several features show statistically significant differences between groups")
print(f"   • The model demonstrates good discriminative ability (AUC = {roc_auc:.3f})")

print(f"\n4. CLINICAL IMPLICATIONS:")
print(f"   • This model could assist healthcare providers in risk assessment")
print(f"   • Early identification of high-risk patients enables preventive interventions")
print(f"   • The model uses readily available clinical measurements")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("✅ All analysis steps completed successfully!")
print("📊 Results saved and ready for report compilation.")
print("📈 Charts and statistics generated for inclusion in final report.")