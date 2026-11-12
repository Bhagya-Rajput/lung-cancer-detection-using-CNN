# 🫁 Lung Cancer Detection Using CNN

An educational deep learning project that uses a **Convolutional Neural Network (CNN)** to classify lung CT scan images into three categories: **Benign, Malignant, and Normal**, using the IQ-OTHNCCD dataset.

The project includes a custom CNN architecture, image preprocessing, model training, evaluation, and a Streamlit-based web application for image classification.

> **Medical Disclaimer:** This project is developed for educational and research purposes only. It is not a clinically validated medical device and must not be used to diagnose cancer, rule out cancer, or make medical or treatment decisions.

---

## 📌 Project Overview

Lung cancer is one of the major health challenges worldwide. This project explores how deep learning and computer vision can be used to classify lung CT scan images.

The model uses a custom CNN architecture with GroupNormalization to learn visual features from CT images and predict one of three dataset classes.

### 🎯 Objectives

- Develop a custom CNN for lung CT image classification.
- Improve model generalization through image preprocessing and augmentation.
- Address class imbalance during training.
- Evaluate performance using classification reports and confusion matrices.
- Build an interactive Streamlit application.
- Explore out-of-distribution (OOD) detection and model limitations.

---

## 📂 Dataset

**Dataset:** IQ-OTHNCCD (Iraq-Oncology Teaching Hospital and National Center for Cancer Diseases)

The dataset contains lung CT scan images divided into three categories.

### Dataset Folder Structure

Place the dataset in the root directory of the repository using the following structure:

```text
Lung-Cancer-Detection-Using-CNN/
│
├── Augmented IQ-OTHNCCD lung cancer dataset/
│   ├── Benign cases/
│   ├── Malignant cases/
│   └── Normal cases/
│
├── models/
├── main.py
├── app.py
├── create_model.py
├── requirements.txt
└── README.md
```

**Classes:**

| Class | Description |
|---|---|
| Benign | Non-cancerous cases |
| Malignant | Cancerous cases |
| Normal | Normal lung CT images |

During training, original unaugmented CT images are reserved for validation, while the remaining original images and augmented images are used for training.

This separation helps reduce data leakage between the training and validation sets.

---

## 🧠 Model Architecture

This project uses a **custom 5-block CNN architecture with GroupNormalization**.

### Key Components

- Convolutional layers for feature extraction.
- GroupNormalization for training stability.
- ReLU activation for non-linearity.
- MaxPooling for spatial dimension reduction.
- Dropout for regularization.
- Global Average Pooling for feature aggregation.
- Dense layers for classification.
- Softmax activation for three-class output.

### Model Configuration

| Parameter | Value |
|---|---|
| Model | Custom CNN |
| Input Image Size | 160 × 160 |
| Input Channels | 3 (RGB) |
| Normalization | GroupNormalization |
| Output Classes | 3 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Activation | ReLU / Softmax |

---

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV, NumPy
- **Data Visualization:** Matplotlib
- **Model Evaluation:** Scikit-learn
- **Web Application:** Streamlit
- **Development Environment:** Visual Studio Code

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Bhagya-Rajput/lung-cancer-detection-using-CNN.git
```

### 2. Navigate to the Project Directory

```bash
cd lung-cancer-detection-using-CNN
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

Run the following command from the repository root:

```bash
python main.py
```

The training pipeline performs:

1. Dataset loading and class identification.
2. Image resizing to 160 × 160 pixels.
3. RGB conversion and pixel normalization.
4. Data augmentation.
5. Training and validation splitting.
6. CNN model construction.
7. Model training and validation.
8. Model evaluation using classification metrics.
9. Saving the best model checkpoint.

### Saved Model Files

After training, the following files are generated:

```text
models/
├── lung_cancer_model.keras
└── class_names.json
```

The model checkpoint and class-name mapping are used by the Streamlit application.

---

## 🌐 Run the Streamlit Application

Start the web application using:

```bash
python -m streamlit run app.py
```

The application allows users to:

- Upload lung CT scan images.
- Preview uploaded images.
- Perform image classification.
- View predicted classes and model scores.
- Receive a warning when the model's output confidence is low.

Supported image formats include JPG, JPEG, PNG, BMP, and TIFF.

**Important:** The application must use the same image preprocessing as the training pipeline, including 160 × 160 resizing, RGB conversion, and pixel normalization to the range [0, 1].

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Per-class metrics are important because a model may achieve reasonable overall accuracy while performing poorly on a particular class.

The classification report and confusion matrix help identify incorrect predictions and class-specific weaknesses.

---

## 🔍 Out-of-Distribution (OOD) Detection

A standard three-class softmax classifier always produces scores for Benign, Malignant, and Normal, even when the input image is unrelated to the training dataset.

Therefore, a high softmax score does not guarantee that an image belongs to one of the trained classes.

This project explores conservative confidence-based warnings as an initial safeguard.

For more reliable OOD detection, further work is required:

- Collect external OOD images.
- Test the model on unrelated images and other medical scans.
- Evaluate unfamiliar CT images from different sources.
- Investigate energy-based scoring or feature-based OOD detection.
- Calibrate and validate rejection thresholds using separate datasets.

**Note:** OOD detection is not guaranteed by a simple softmax threshold. The model should not be considered reliable for unfamiliar medical images without further validation.

---

## 📁 Project Files

| File / Folder | Description |
|---|---|
| `main.py` | Dataset loading, CNN construction, training and evaluation |
| `app.py` | Streamlit web application |
| `create_model.py` | Prevents accidental creation of a dummy untrained model |
| `requirements.txt` | Required Python libraries |
| `models/` | Saved trained model and class mapping |
| `Augmented IQ-OTHNCCD lung cancer dataset/` | Dataset directory |
| `README.md` | Project documentation |

---

## 🔮 Future Improvements

- Improve classification performance across all three classes.
- Investigate incorrect predictions and class imbalance.
- Implement and evaluate more reliable OOD detection.
- Test generalization on independent external datasets.
- Improve the Streamlit interface and user experience.
- Explore explainable AI techniques such as Grad-CAM.

---

## 👨‍💻 Author

**Bhagya Kumar Rajput**

B.Tech CSE – Data Science  
Aspiring Data Scientist | Machine Learning Enthusiast

- **GitHub:** [Bhagya-Rajput](https://github.com/Bhagya-Rajput)
- **LinkedIn:** [Bhagya Kumar Rajput](https://www.linkedin.com/in/bhagya-kumar-rajput-124849359/)

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ If you find this project interesting, consider giving the repository a star!