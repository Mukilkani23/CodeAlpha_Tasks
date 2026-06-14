# Handwritten Character Recognition System

A deep learning web app that recognizes handwritten letters (A–Z) in real time using a CNN trained on the EMNIST Letters dataset.

---


---

## Project Overview

This project is Task 3 of the CodeAlpha ML Internship. It covers the full ML pipeline — dataset preparation, CNN architecture design, model training, and deployment as an interactive Streamlit web app.

---

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- Streamlit
- streamlit-drawable-canvas
- NumPy, Pillow, OpenCV

---

## Model Architecture

- Input: 28×28 grayscale image
- Conv2D → BatchNormalization → MaxPooling → Dropout (repeated blocks)
- Dense output layer with 26 units (A–Z)
- Trained on EMNIST Letters dataset (88,800 samples)
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy

---

## Features

- Drawable canvas — draw directly in the browser
- Top-3 predictions with confidence percentages
- EMNIST rotation correction applied automatically
- Clear & Draw Again button to reset canvas
- Deployed on Streamlit Cloud

---

## How to Run Locally

```bash
git clone https://github.com/Mukilkani23/CodeAlpha-Handwritten-Character-Recognition.git
cd CodeAlpha-Handwritten-Character-Recognition
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

EMNIST Letters — [Kaggle](https://www.kaggle.com/datasets/crawford/emnist)

26 classes (A–Z), 28×28 grayscale images, 88,800 training samples.

---

## Project Structure
```
Task_3/
├── model/
│   └── emnist_letters_cnn.h5
├── app.py
├── requirements.txt
└── README.md
```

---

## Results

- Training Accuracy: ~94%
- Validation Accuracy: ~92%
- Top-3 prediction displayed with confidence scores

---

## Author

Mukilkani R P 
[GitHub](https://github.com/Mukilkani23)
[LinkedIn](https://www.linkedin.com/in/mukilkani-r-p-62831b369/)
CodeAlpha ML Internship | Task 3
