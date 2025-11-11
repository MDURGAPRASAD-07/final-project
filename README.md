# Predicting Diabetes Early Using Machine Learning

This project predicts whether a person is **non-diabetic**, **prediabetic**, or **diabetic** using health and lifestyle data from the **BRFSS 2015** public survey dataset.  
The model used here is an **ensemble machine learning approach**, and class imbalance is handled using **ADASYN** to improve detection of prediabetic and diabetic cases.

---

## Dataset
- Source: BRFSS 2015 (CDC)
- Records: ~253,000 people
- Target: `Diabetes_012` (0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes)

The dataset path in the code is already set:
