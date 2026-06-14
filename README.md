# CodeAlpha ML Internship — Project Portfolio

**Intern:** Mukilkani R P
**GitHub:** [Mukilkani23](https://github.com/Mukilkani23)
**LinkedIn:** [mukilkani-r-p](https://www.linkedin.com/in/mukilkani-r-p-62831b369/)
**Internship:** CodeAlpha Machine Learning Internship

---

This repository contains all completed tasks from the CodeAlpha ML Internship. Each task is a standalone end-to-end ML project covering data processing, model training, evaluation, and Streamlit deployment.

---

## Repository Structure

```
CodeAlpha-ML-Internship/
├── Task_1_Credit_Scoring/
│   ├── train.py
│   ├── app.py
│   ├── requirements.txt
│   ├── .gitignore
│   ├── model/
│   │   ├── credit_model_pipeline.pkl
│   │   ├── label_encoders.pkl
│   │   ├── feature_cols.pkl
│   │   └── model_info.txt
│   └── plots/
│       ├── 01_class_balance.png
│       ├── 02_correlation_heatmap.png
│       ├── 03_feature_distributions.png
│       ├── 04_default_by_purpose.png
│       ├── 05_confusion_matrix.png
│       ├── 06_roc_curves.png
│       └── 07_feature_importance.png
│
├── Task_3_Handwritten_Recognition/
│   ├── app.py
│   ├── requirements.txt
│   └── model/
│       └── emnist_letters_cnn.h5
│
├── Task_4_Disease_Prediction/
│   ├── app.py
│   ├── Disease_Prediction_model.ipynb
│   ├── heart.csv
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   └── requirements.txt
│
└── README.md
```

---

## Task 1 — Credit Scoring Model

**Problem:** Given a loan applicant's financial profile, predict whether they will default on the loan (binary classification).

**Dataset:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
32,581 records · 12 raw features → 14 after feature engineering · ~22% default rate

### Features

| Feature | Type | Description |
|---|---|---|
| person_age | int | Applicant age |
| person_income | float | Annual income |
| person_home_ownership | cat | RENT / OWN / MORTGAGE / OTHER |
| person_emp_length | float | Employment length (years) |
| loan_intent | cat | Loan purpose |
| loan_grade | cat | Bank-assigned grade A–G |
| loan_amnt | float | Loan amount requested |
| loan_int_rate | float | Interest rate (%) |
| loan_percent_income | float | Loan as % of income |
| cb_person_default_on_file | cat | Historical default Y/N |
| cb_person_cred_hist_length | int | Years of credit history |
| loan_status | int | Target — 0 (No Default) / 1 (Default) |

**Engineered features:** `debt_to_income`, `is_young`, `high_interest`

### Models Trained

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

*(Run `train.py` to populate these results)*

### Tech Stack

Python · scikit-learn · XGBoost · pandas · numpy · matplotlib · seaborn · joblib · Streamlit

### Key Design Decisions

- `sklearn Pipeline` bundles preprocessing with each model to prevent train/inference feature mismatch
- Class imbalance handled via `class_weight='balanced'` (sklearn) and `scale_pos_weight` (XGBoost)
- ROC-AUC used as primary metric over accuracy for imbalanced classification

### How to Run

```bash
cd Task_1_Credit_Scoring
pip install -r requirements.txt
# Place credit_risk_dataset.csv in this folder
python train.py
streamlit run app.py
```

---

## Task 3 — Handwritten Character Recognition

**Problem:** Recognize handwritten capital letters (A–Z) from a drawn canvas input in real time.

**Dataset:** [EMNIST Letters — Kaggle](https://www.kaggle.com/datasets/crawford/emnist)
88,800 training samples · 26 classes (A–Z) · 28×28 grayscale images

### Model Architecture

```
Input (28×28 grayscale)
→ Conv2D → BatchNormalization → MaxPooling → Dropout
→ Conv2D → BatchNormalization → MaxPooling → Dropout
→ Flatten → Dense(256) → Dropout
→ Dense(26, softmax)
```

Optimizer: Adam · Loss: Sparse Categorical Crossentropy

### Results

| Metric | Score |
|---|---|
| Training Accuracy | ~94% |
| Validation Accuracy | ~92% |

### Features

- Drawable canvas — draw directly in the browser using `streamlit-drawable-canvas`
- Top-3 predictions displayed with confidence percentages
- EMNIST rotation correction applied automatically (dataset images are rotated 90° + mirrored)
- Clear & Draw Again button to reset canvas

### Tech Stack

Python · TensorFlow / Keras · Streamlit · streamlit-drawable-canvas · NumPy · Pillow · OpenCV

### How to Run

```bash
cd Task_3_Handwritten_Recognition
python -m venv venv311
venv311\Scripts\activate       # Windows
# source venv311/bin/activate  # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

---

## Task 4 — Heart Disease Prediction

**Problem:** Predict the likelihood of heart disease in a patient based on 13 clinical attributes (binary classification).

**Dataset:** [Heart Disease Dataset — UCI / Kaggle](https://www.kaggle.com/)
1,025 patients · 13 features · 526 positive cases, 499 negative

### Features

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia type |

### Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8098 | 0.7619 | 0.9143 | 0.8312 | 0.9298 |
| SVM | 0.9268 | 0.9167 | 0.9429 | 0.9296 | 0.9771 |
| Random Forest | 0.8780 | 0.8679 | 0.9048 | 0.8860 | 0.9440 |
| XGBoost | 0.8927 | 0.8868 | 0.9143 | 0.9003 | 0.9524 |

**Final model: XGBoost** — best balance of F1 and ROC-AUC after regularization (`max_depth`, `min_samples_split`, `subsample`)

### Tech Stack

Python · scikit-learn · XGBoost · pandas · numpy · matplotlib · seaborn · joblib · Streamlit

### Key Design Decisions

- Stratified train/test split to preserve class balance
- `StandardScaler` fit only on training data to prevent data leakage
- Regularization applied to tree-based models to prevent perfect-score overfitting

### How to Run

```bash
cd Task_4_Disease_Prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Learnings Across All Tasks

**sklearn Pipelines** — bundling preprocessing inside a Pipeline prevents train/inference feature mismatch. Critical for any deployed model.

**Evaluation metrics** — ROC-AUC and F1 tell the real story in imbalanced datasets. Accuracy alone is misleading.

**Overfitting detection** — a perfect score (1.0000) on tree models is a red flag, not a win. Regularize.

**Data leakage** — scalers and encoders must be fit on training data only. Fitting on the full dataset contaminates results.

**CNN preprocessing** — the EMNIST dataset ships with images rotated 90° and mirrored. Always inspect dataset quirks before training.

**Feature engineering** — derived features (`debt_to_income`, `is_young`, `high_interest`) add signal beyond raw columns.

---

*CodeAlpha ML Internship · Mukilkani R P · 2026*
