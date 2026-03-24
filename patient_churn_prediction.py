# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('r'D:\Data Analytics Projects\patient_churn_dataset.csv'')
df.head()
# %%
import seaborn as sns
# %%
df.columns
# %%
df.isna().sum()
# %%
df.info()
# %%
df.describe()
# %%
df['Last_Interaction_Date']=pd.to_datetime(df['Last_Interaction_Date'])
# %%
df.info()
# %%
#Univariate Analysis
df['Churned'].value_counts().plot(kind='bar')
plt.title('Churned vs Not Churned')
# %%
sns.kdeplot(df['Age'], fill=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Density')
plt.grid(True)
# %%
sns.boxplot(x=df['Distance_To_Facility_Miles'])
plt.title('Distance to Facility Distribution')
plt.xlabel('Distance (miles)')

# %%
df['Specialty'].value_counts().plot(kind='pie', autopct='%1.1f%%')
# %%
df['Insurance_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
# %%
df['Portal_Usage'].value_counts().plot(kind='bar')
# %%
#Bivariate Analysis
df.groupby('Churned')['Missed_Appointments'].sum().plot(kind='bar')
plt.xlabel('Churned')
plt.ylabel('Total Missed Appointments')
plt.title('Missed Appointments by Churn Status')
# %%
df.groupby('Churned')['Overall_Satisfaction'].mean().plot(kind='bar')
plt.xlabel('Churned')
plt.ylabel('Average Overall Satisfaction')
plt.title('Overall Satisfaction by Churn Status')
# %%
df.groupby('Insurance_Type')['Churned'].sum().sort_values(ascending=False).plot(kind='bar')
plt.xlabel('Insurance Type')
plt.ylabel('Total Churned')
plt.title('Churned by Insurance Type')
# %%
df.groupby('Churned')['Distance_To_Facility_Miles'].mean().plot(kind='bar')
plt.xlabel('Churned')
plt.ylabel('Average Distance to Facility')
plt.title('Distance to Facility by Churn Status')
# %%
df.groupby('Portal_Usage')['Churned'].sum().plot(kind='bar')
plt.xlabel('Portal Usage')
plt.ylabel('Total Churned')
plt.title('Churned by Portal Usage')
# %%
df.groupby('Churned')['Avg_Out_Of_Pocket_Cost'].mean().plot(kind='bar')
plt.xlabel('Churned')
plt.ylabel('Average Out-of-Pocket Cost')
plt.title('Out-of-Pocket Cost by Churn Status')
# %%
df.groupby('Churned')['Wait_Time_Satisfaction'].mean().plot(kind='bar')
plt.xlabel('Churned')
plt.ylabel('Average Wait Time Satisfaction')
plt.title('Wait Time Satisfaction by Churn Status')

# %%
#Correlation Analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5)

plt.title('Correlation Matrix for Patient Churn Factors')
plt.show()
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Features and target
X = df.drop('Churned', axis=1)
y = df['Churned']


if 'CustomerID' in X.columns:
    X = X.drop('CustomerID', axis=1)

X = X.select_dtypes(exclude=['datetime64[ns]'])


X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
print("Train Accuracy:", accuracy_score(y_train, y_train_pred) * 100, "%")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred) * 100, "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
            xticklabels=['Stayed (0)', 'Churned (1)'],
            yticklabels=['Stayed (0)', 'Churned (1)'])

plt.title('Confusion Matrix - Patient Churn Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
# %%
## Patient Churn Prediction Project

## Overview

#This project analyzes patient data to understand the key factors influencing churn and builds a Logistic Regression model to predict whether a patient will leave or continue with the healthcare facility.



## Dataset

#The dataset includes:

#* Demographics (Age, Distance to facility)
#* Behavioral factors (Missed appointments, Portal usage)
#* Financial details (Out-of-pocket cost, Insurance type)
#* Satisfaction metrics (Overall satisfaction, Wait time satisfaction)
#* Target: **Churned (0 = Stayed, 1 = Churned)**



## Workflow

### 1. Data Loading & Inspection

#* Loaded dataset using pandas
#* Checked structure, data types, and missing values

### 2. Data Preprocessing

#* Converted date column to datetime format
#* Removed irrelevant columns like IDs
#* Converted categorical variables using one-hot encoding
#* Scaled features using StandardScaler

### 3. Exploratory Data Analysis

#### Univariate Analysis

#* Distribution of churn
#* Age distribution
#* Distance spread
#* Category distributions (insurance, specialty, portal usage)

#### Bivariate Analysis

#* Relationship between churn and:

  #* Missed appointments
  #* Satisfaction levels
  #* Costs
  #* Distance
  #* Portal usage

#### Correlation Analysis

#* Heatmap to identify relationships between numerical features



## Model Building

#* Model: Logistic Regression
#* Train-test split: 80-20
#* Feature scaling applied
#* Model trained and predictions generated



## Model Evaluation

#* Accuracy (Train & Test)
#* Confusion Matrix
#* Classification Report (Precision, Recall, F1-score)



## Key Insights

#* **Missed Appointments:** A higher number of missed visits strongly indicates that the patient is likely to leave. It reflects reduced engagement with the healthcare service.

#* **Wait Time Satisfaction:** Patients who are dissatisfied with waiting times tend to drop off more. Efficient service plays a major role in retaining patients.

#* **Out-of-Pocket Cost:** Higher personal expenses increase the chances of churn. Patients prefer affordable and predictable healthcare costs.

#* **Portal Usage:** Patients who actively use digital platforms (like appointment booking portals) are more likely to stay, indicating better engagement.

#* **Billing Issues:** Frequent problems in billing reduce trust and push patients away. A smooth billing system is important for retention.

#* **Distance to Facility:** Patients living farther away show a higher tendency to churn due to convenience factors.

#* **Staff & Provider Ratings:** Good experience with both doctors and staff is necessary. Poor support interactions can lead to churn even if clinical care is good.

#* **Tenure:** New patients are more likely to leave compared to long-term patients. Early-stage engagement is critical for retention.

#---

## Requirements

#* pandas
#* numpy
#* matplotlib
#* seaborn
#* scikit-learn



## Future Improvements

#* Handle class imbalance
#* Try advanced models (Random Forest, XGBoost)
#* Perform hyperparameter tuning
#* Feature selection
#* Deploy as an application/dashboard



## Conclusion

#This project demonstrates a complete pipeline from data analysis to predictive modeling. It highlights the most influential factors behind patient churn and provides a base for improving retention strategies.
