# 🫁 Pneumonia Detection using Deep Learning

A deep learning-based system for automated detection of pneumonia from chest X-ray images. This project compares multiple models including a custom CNN, ResNet50, and MobileNetV2 to identify the most effective architecture for medical image classification.

---

## 📌 Overview

Pneumonia is a serious lung infection that requires early detection for effective treatment. This project leverages deep learning techniques to classify chest X-ray images into **Normal** and **Pneumonia** categories.

The system is trained on the **Kaggle Chest X-ray dataset** and evaluated using standard classification metrics.

---

## 🎯 Objectives

- Develop an automated pneumonia detection system  
- Compare CNN with transfer learning models  
- Improve model performance using preprocessing and augmentation  
- Minimize false negatives (critical in medical diagnosis)  

---

## 🧠 Models Used

- ✅ Custom Convolutional Neural Network (CNN)  
- ✅ ResNet50 (Transfer Learning + Fine-Tuning)  
- ✅ MobileNetV2 (Transfer Learning)  

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Image resizing (224×224)
   - Normalization (1/255 scaling)

2. **Data Augmentation**
   - Rotation (20°)
   - Zoom (0.2)
   - Horizontal Flip

3. **Training Strategy**
   - CNN trained from scratch  
   - ResNet50 & MobileNetV2 trained using:
     - Feature extraction (frozen layers)
     - Fine-tuning (unfrozen top layers)

---

## 📊 Results

| Model         | Accuracy | Precision | Recall | F1-Score |
|--------------|---------|----------|--------|----------|
| CNN          | **0.92** | 0.92     | 0.92   | 0.92     |
| ResNet50     | 0.87     | 0.87     | 0.87   | 0.86     |
| MobileNetV2  | 0.80     | 0.85     | 0.80   | 0.78     |

🔍 **Key Insight:**  
The custom CNN outperformed pretrained models, showing better adaptation to domain-specific features.

---

## 📈 Visualizations

- Loss Curves  
- Accuracy Curves  
- Confusion Matrices  

*(Add images here if uploaded)*

---

## 🏥 Key Observations

- CNN achieved the highest accuracy (**92%**)  
- High recall for Pneumonia (critical for diagnosis)  
- ResNet50 improved after fine-tuning  
- MobileNetV2 showed class imbalance issues  

---

## 🚀 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Matplotlib  
- Google Colab (GPU – NVIDIA T4)

---

## 📂 Dataset

- Kaggle Chest X-ray (Pneumonia) Dataset  
- Contains ~5,800 labeled images  

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/pneumonia-detection.git

# Navigate to project folder
cd pneumonia-detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
