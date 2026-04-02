# Patient Churn Prediction Project

# Overview

This project analyzes patient data to understand the key factors influencing churn and builds a Logistic Regression model to predict whether a patient will leave or continue with the healthcare facility.



# Dataset

The dataset includes:

* Demographics (Age, Distance to facility)
* Behavioral factors (Missed appointments, Portal usage)
* Financial details (Out-of-pocket cost, Insurance type)
* Satisfaction metrics (Overall satisfaction, Wait time satisfaction)
* Target: **Churned (0 = Stayed, 1 = Churned)**



# Workflow

1. Data Loading & Inspection

* Loaded dataset using pandas
* Checked structure, data types, and missing values

2. Data Preprocessing

* Converted date column to datetime format
* Removed irrelevant columns like IDs
* Converted categorical variables using one-hot encoding
* Scaled features using StandardScaler

# Exploratory Data Analysis

# Univariate Analysis

* Distribution of churn
* Age distribution
* Distance spread
* Category distributions (insurance, specialty, portal usage)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>State</th>
      <th>Tenure_Months</th>
      <th>Specialty</th>
      <th>Insurance_Type</th>
      <th>Visits_Last_Year</th>
      <th>Missed_Appointments</th>
      <th>Days_Since_Last_Visit</th>
      <th>...</th>
      <th>Overall_Satisfaction</th>
      <th>Wait_Time_Satisfaction</th>
      <th>Staff_Satisfaction</th>
      <th>Provider_Rating</th>
      <th>Avg_Out_Of_Pocket_Cost</th>
      <th>Billing_Issues</th>
      <th>Portal_Usage</th>
      <th>Referrals_Made</th>
      <th>Distance_To_Facility_Miles</th>
      <th>Churned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C20000</td>
      <td>41</td>
      <td>Female</td>
      <td>PA</td>
      <td>62</td>
      <td>Pediatrics</td>
      <td>Medicaid</td>
      <td>1</td>
      <td>0</td>
      <td>564</td>
      <td>...</td>
      <td>3.5</td>
      <td>4.9</td>
      <td>3.8</td>
      <td>4.2</td>
      <td>306</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>21.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C20001</td>
      <td>43</td>
      <td>Female</td>
      <td>GA</td>
      <td>44</td>
      <td>Internal Medicine</td>
      <td>Self-Pay</td>
      <td>7</td>
      <td>4</td>
      <td>254</td>
      <td>...</td>
      <td>2.6</td>
      <td>3.1</td>
      <td>4.7</td>
      <td>4.3</td>
      <td>1851</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47.6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C20002</td>
      <td>21</td>
      <td>Male</td>
      <td>MI</td>
      <td>120</td>
      <td>Internal Medicine</td>
      <td>Medicaid</td>
      <td>15</td>
      <td>5</td>
      <td>89</td>
      <td>...</td>
      <td>1.6</td>
      <td>4.4</td>
      <td>2.1</td>
      <td>4.7</td>
      <td>391</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>7.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C20003</td>
      <td>65</td>
      <td>Male</td>
      <td>FL</td>
      <td>118</td>
      <td>General Practice</td>
      <td>Private</td>
      <td>10</td>
      <td>3</td>
      <td>135</td>
      <td>...</td>
      <td>2.6</td>
      <td>4.3</td>
      <td>4.3</td>
      <td>4.9</td>
      <td>808</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11.6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C20004</td>
      <td>18</td>
      <td>Female</td>
      <td>CA</td>
      <td>70</td>
      <td>Cardiology</td>
      <td>Medicaid</td>
      <td>5</td>
      <td>4</td>
      <td>696</td>
      <td>...</td>
      <td>2.2</td>
      <td>4.0</td>
      <td>4.1</td>
      <td>4.4</td>
      <td>866</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10.3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
*
* # Bivariate Analysis

* Relationship between churn and:

  * Missed appointments
  * Satisfaction levels
  * Costs
  * Distance
  * Portal usage

# Correlation Analysis

* Heatmap to identify relationships between numerical features



#  Model Building

* Model: Logistic Regression
* Train-test split: 80-20
* Feature scaling applied
* Model trained and predictions generated



  # Model Evaluation

* Accuracy (Train & Test)
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)



# Key Insights

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


# Conclusion

#This project demonstrates a complete pipeline from data analysis to predictive modeling. It highlights the most influential factors behind patient churn and provides a base for improving retention strategies.
