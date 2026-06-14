# Heart Disease Prediction

A machine learning web app that predicts the likelihood of heart disease based on clinical patient data.

Built as **Task 4** of the CodeAlpha ML Internship.

---

## Problem Statement

Heart disease is one of the leading causes of death globally. Early detection using patient vitals and clinical markers can significantly improve outcomes. This project trains multiple ML models on the Cleveland Heart Disease dataset and deploys the best-performing model as an interactive web app.

---

## Dataset

- **Source:** [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (via Kaggle)
- **Rows:** 1,025 patients
- **Target:** `1` = Heart Disease, `0` = No Heart Disease
- **Class balance:** 526 positive / 499 negative (near-balanced)

### Features

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 1 = Male, 0 = Female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = Yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0–2) |
| ca | Number of major vessels colored by fluoroscopy (0–4) |
| thal | Thalassemia type (0–3) |

---

## Models Trained

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.81 | 0.76 | 0.91 | 0.83 | 0.93 |
| SVM | 0.93 | 0.92 | 0.94 | 0.93 | 0.98 |
| Random Forest | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* |
| **XGBoost** | **0.98** | **0.98** | **0.98** | **0.98** | — |

> *Random Forest showed signs of overfitting on the training split (1.00 on test set is a red flag on a dataset this size). XGBoost with regularization (`max_depth=4`, `subsample=0.8`) was selected as the final model for being both high-performing and more reliable.

---

## Tech Stack

- **Language:** Python 3.x
- **ML:** scikit-learn, XGBoost
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **App:** Streamlit
- **Model persistence:** joblib
- **Training environment:** Google Colab

---

## Project Structure

```
heart-disease-predictor/
├── app.py                    # Streamlit web app
├── heart.csv                 # Dataset
├── heart_disease_model.pkl   # Trained XGBoost model
├── scaler.pkl                # StandardScaler (used for LR/SVM reference)
├── requirements.txt          # Dependencies
└── README.md
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Mukilkani23/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## requirements.txt

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
```

---

## How It Works

1. User enters 13 clinical features through the Streamlit UI
2. Input is converted to a DataFrame matching the training schema
3. XGBoost model predicts: `0` (No Disease) or `1` (Heart Disease)
4. `predict_proba` returns the probability score, displayed as a percentage
5. Result is shown with a color-coded risk label and advice

---

## Key Learnings

- **Overfitting detection:** Perfect 1.00 scores on a small dataset = red flag. Random Forest memorized the training data. XGBoost with depth/subsample constraints generalized better.
- **Data leakage:** Scaler was fit only on training data, then applied to test data — never fit on test split.
- **Scaling matters by model:** LR and SVM need scaled input; tree-based models (RF, XGBoost) work fine on raw data.
- **predict with `.values`:** XGBoost trained on NumPy arrays doesn't handle DataFrames with column names gracefully — pass `.values` at inference time.

---

## Author

**Mukilkani R P**
B.E. CSE, Anna University (2028)
GitHub: [Mukilkani23](https://github.com/Mukilkani23)

CodeAlpha ML Internship — Task 4
