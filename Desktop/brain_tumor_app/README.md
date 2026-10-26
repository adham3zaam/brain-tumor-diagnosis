# 🧠 Brain Tumor MRI Diagnosis

An AI-powered web application for detecting and classifying brain tumors from MRI scans using deep learning.

## 🎯 Features

- Upload brain MRI images and get instant diagnosis
- Detects 4 classes: Glioma, Meningioma, Pituitary Tumor, and No Tumor
- Professional medical-style interface built with Streamlit
- Detailed probability analysis for each class
- ~96% accuracy on test data

## 🏗️ Model Architecture

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Method:** Transfer Learning with fine-tuning
- **Input Size:** 224×224 pixels
- **Framework:** TensorFlow / Keras

## 📊 Dataset

Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-diagnosis.git
cd brain-tumor-diagnosis
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
brain-tumor-diagnosis/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── class_names.json          # Class labels
├── models/
│   └── brain_tumor_model.keras   # Trained model
└── README.md

احفظ (`Ctrl + S`).

---

### 🐙 الخطوة 3: اعمل الريبو على GitHub

1. افتح [github.com/new](https://github.com/new)
2. املأ:
   - **Repository name**: `brain-tumor-diagnosis`
   - **Description**: `AI-powered brain tumor detection from MRI scans`
   - **Visibility**: **Public** (أو Private لو عايز)
   - ⚠️ **متعلمش** على أي حاجة تحت (README, .gitignore, license)
3. اضغط **Create repository**

---

### 💻 الخطوة 4: في Terminal في VS Code

**اتأكد إنك جوه فولدر المشروع** (المسار في الـ Terminal يبان فيه `brain_tumor_app`).

شغّل الأوامر دي **بالترتيب**:

```bash
git init