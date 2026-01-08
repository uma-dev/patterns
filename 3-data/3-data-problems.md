## Introduction
In the field of machine learning and data mining, datasets present different types of complexity that affect the quality of models. Among the most relevant are **class imbalance** and **missing data**, problems that can compromise the accuracy and generalization of classification algorithms.  

---

## Development

### 1. Types of Data Complexity
- **Class imbalance:** occurs when one class has many more instances than the other(s).  
- **Missing data:** null or absent values that prevent proper model training.  
- **Noise:** outliers or inconsistent values.  
- **High dimensionality:** too many attributes that complicate modeling.  
- **Redundancy:** repeated or irrelevant attributes.  

### 2. Class Imbalance
**Problem:** models tend to favor the majority class.  
**Possible solutions:**  
- Resampling (oversampling the minority class, undersampling the majority class).  
- Specialized algorithms (SMOTE, ADASYN).  
- Using proper metrics (F1-score, AUC-ROC instead of just accuracy).  

### 3. Missing Data
**Problem:** missing values distort the available information.  
**Possible solutions:**  
- Removing instances or attributes with too many null values.  
- Value imputation (mean, median, mode, or advanced techniques like KNN-imputer).  
- Using models robust to missing data.  

---

## Conclusion
Properly handling class imbalance and missing data is essential to ensure fair and reliable results. Applying resampling techniques, imputation methods, and appropriate evaluation metrics helps reduce negative effects.  

---
