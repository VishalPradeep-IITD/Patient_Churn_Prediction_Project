Patient Churn Prediction Project

Overview

This project analyzes patient data to understand the key factors influencing churn and builds a Logistic Regression model to predict whether a patient will leave or continue with the healthcare facility.



Dataset

The dataset includes:

* Demographics (Age, Distance to facility)
* Behavioral factors (Missed appointments, Portal usage)
* Financial details (Out-of-pocket cost, Insurance type)
* Satisfaction metrics (Overall satisfaction, Wait time satisfaction)
* Target: **Churned (0 = Stayed, 1 = Churned)**



 Workflow

1. Data Loading & Inspection

* Loaded dataset using pandas
* Checked structure, data types, and missing values

2. Data Preprocessing

* Converted date column to datetime format
* Removed irrelevant columns like IDs
* Converted categorical variables using one-hot encoding
* Scaled features using StandardScaler

# 3. Exploratory Data Analysis

# Univariate Analysis

* Distribution of churn
* Age distribution
* Distance spread
* Category distributions (insurance, specialty, portal usage)

# Bivariate Analysis

* Relationship between churn and:

  * Missed appointments
  * Satisfaction levels
  * Costs
  * Distance
  * Portal usage

# Correlation Analysis

* Heatmap to identify relationships between numerical features



# Model Building

* Model: Logistic Regression
* Train-test split: 80-20
* Feature scaling applied
* Model trained and predictions generated



# Model Evaluation

* Accuracy (Train & Test)
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)



## Key Insights

* **Missed Appointments:** A higher number of missed visits strongly indicates that the patient is likely to leave. It reflects reduced engagement with the healthcare service.

* **Wait Time Satisfaction:** Patients who are dissatisfied with waiting times tend to drop off more. Efficient service plays a major role in retaining patients.

* **Out-of-Pocket Cost:** Higher personal expenses increase the chances of churn. Patients prefer affordable and predictable healthcare costs.

* **Portal Usage:** Patients who actively use digital platforms (like appointment booking portals) are more likely to stay, indicating better engagement.

* **Billing Issues:** Frequent problems in billing reduce trust and push patients away. A smooth billing system is important for retention.

* **Distance to Facility:** Patients living farther away show a higher tendency to churn due to convenience factors.

* **Staff & Provider Ratings:** Good experience with both doctors and staff is necessary. Poor support interactions can lead to churn even if clinical care is good.

* **Tenure:** New patients are more likely to leave compared to long-term patients. Early-stage engagement is critical for retention.

# Requirements

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

# Future Improvements

* Handle class imbalance
* Try advanced models (Random Forest, XGBoost)
* Perform hyperparameter tuning
* Feature selection
* Deploy as an application/dashboard




## Conclusion

#This project demonstrates a complete pipeline from data analysis to predictive modeling. It highlights the most influential factors behind patient churn and provides a base for improving retention strategies.
