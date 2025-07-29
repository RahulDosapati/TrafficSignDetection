# 🚦 Traffic Sign Detection

This project presents a deep learning-based image classification system to detect and classify traffic signs using camera-captured images. Leveraging both a custom Convolutional Neural Network (CNN) and the MobileNet architecture, this project aims to support autonomous driving systems by accurately recognizing road signs.

## 🌐 Project Overview

The classification models are trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset containing thousands of traffic sign images across 43 categories. The pipeline includes:

- Image preprocessing
- Model training using both CNN and MobileNet
- Performance evaluation based on key metrics

## 📁 Dataset

- **Source**: [GTSRB - German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- **Description**:
  - Over 50,000 labeled images of traffic signs
  - 43 distinct classes (e.g., Speed limits, Stop, Yield, No Entry)
  - Includes variations in lighting, angle, and occlusion to simulate real driving scenarios

## 🧠 Models

- **CNN (Built from Scratch)**:
  - Layers: Conv2D, MaxPooling2D, Dropout, Flatten, Dense
  - Trained from scratch using raw traffic sign images

- **MobileNet (Transfer Learning)**:
  - Uses pre-trained MobileNet (on ImageNet)
  - Fine-tuned on the GTSRB dataset
  - Faster training and higher accuracy with fewer parameters

## 🧼 Data Preprocessing

- Image resizing and normalization
- One-hot encoding of labels
- Train-test split using `sklearn`

## 📈 Evaluation Metrics

- Accuracy and loss plots over training epochs
- Confusion matrix to visualize misclassifications
- Classification report including:
  - Precision
  - Recall
  - F1-score

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/traffic-sign-detection.git
   cd traffic-sign-detection
   ```
2. **Install dependencies**

  ```bash
    pip install -r requirements.txt
  ```
3. **Download the dataset**

- Download GTSRB from Kaggle or the official source

- Place it inside a data/ folder

4. **Run the CNN model**

  ```bash
    python src/train_cnn.py
  ```
5. **Run the MobileNet model**

  ```bash
    python src/train_mobilenet.py
  ```
# 🖼️ Sample Results
- Visualizations of training accuracy and loss

- Confusion matrix showing predicted vs actual classes

- MobileNet model achieves higher accuracy with reduced training time


