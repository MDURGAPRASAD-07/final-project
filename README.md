# 🩺 Predicting Diabetes Early Using Ensemble Machine Learning

## 📖 Overview
This project applies **ensemble machine learning methods** to predict diabetes early using health indicators from the CDC **Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset**.  
The work was completed as part of the MSc Data Science program at the **University of Hertfordshire**.  

Diabetes is one of the fastest-growing chronic diseases worldwide, and early detection is critical to reducing complications such as cardiovascular disease, kidney failure, and blindness.  
By leveraging ensemble and boosting algorithms, this project provides a scalable and accurate predictive framework for early diabetes risk detection.

---

## 📂 Repository Structure

---

## 📊 Dataset
- **Source:** [CDC BRFSS 2015 (via Kaggle)](https://www.cdc.gov/brfss/)  
- **Records:** 253,680 survey responses  
- **Features:** 21 health-related variables (BMI, Age, Blood Pressure, Cholesterol, Smoking, etc.)  
- **Target Variable:** `Diabetes_012`
  - `0` → No diabetes  
  - `1` → Prediabetes  
  - `2` → Diabetes  

⚠️ The dataset is highly **imbalanced**, requiring careful handling during preprocessing.

---

## ⚙️ Methodology
This project follows the **CRISP-ML(Q)** framework (Cross-Industry Standard Process for Machine Learning with Quality assurance):

1. **Data Acquisition & Understanding**
   - Loaded and cleaned BRFSS 2015 dataset
   - Conducted EDA: distributions, correlations, imbalance checks, outliers

2. **Data Preparation**
   - Outlier removal (z-score method)
   - Scaling (StandardScaler)
   - Class imbalance handled via **ADASYN** oversampling

3. **Modeling**
   - Compared multiple ensemble classifiers:
     - ExtraTrees, Random Forest, Gradient Boosting
     - XGBoost, LightGBM, AdaBoost
   - Hyperparameter tuning via **GridSearchCV** and **RandomizedSearchCV**

4. **Evaluation**
   - Accuracy, Balanced Accuracy
   - Precision, Recall, F1-score
   - ROC and PR curves
   - Confusion Matrices (both proper & diagnostic balanced)
   - Feature importance analysis

5. **Deployment Preparation**
   - Best model & scaler saved as `.pkl` files using **joblib**
   - Small prediction demo included in `Project.py`

---

## 📈 Results
- **Best-performing model:** ExtraTrees (high recall & F1-score)  
- **Feature Importance (ExtraTrees):**
  - Top predictors: **Age, BMI, General Health, Income, Education**
  - Lifestyle factors (e.g., smoking, physical activity) had smaller influence  

### Key Insights:
- **LightGBM & XGBoost** achieved the best raw accuracy (~87%), but struggled with minority classes.  
- **ExtraTrees** achieved the **highest recall (0.89)** and **F1-score (0.87)**, making it ideal for **screening use-cases** where missing diabetic patients must be minimized.  
- Class imbalance remained the biggest challenge despite resampling.  

---

