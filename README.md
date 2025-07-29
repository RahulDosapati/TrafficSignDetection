🚦 Traffic Sign Detection
This project builds a deep learning-based image classification system to detect and classify traffic signs from camera-captured images. Using both a custom Convolutional Neural Network (CNN) and a fine-tuned MobileNet model, it supports key tasks in autonomous driving by reliably identifying road signs.

🌐 Project Overview
The models are trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset, which includes over 50,000 labeled images spanning 43 different traffic sign classes. The project covers the full pipeline:

Image preprocessing

Training CNN and MobileNet models

Evaluating performance using standard classification metrics

📁 Dataset
Source: GTSRB - German Traffic Sign Recognition Benchmark

Description:

50,000+ labeled images

43 traffic sign classes (e.g., speed limits, stop, yield, no entry)

Includes real-world variation in lighting, orientation, and occlusion

🧠 Models
🔹 CNN (Built from Scratch)
Layers: Conv2D, MaxPooling2D, Dropout, Flatten, Dense

Trained from scratch using the GTSRB dataset

Provides a lightweight, interpretable baseline

🔹 MobileNet (Transfer Learning)
Pre-trained on ImageNet, then fine-tuned on traffic signs

Faster training, fewer parameters, higher accuracy

Suitable for real-time or edge applications

🧼 Data Preprocessing
Image resizing and normalization

One-hot encoding of labels

Train–test split using train_test_split from sklearn

📈 Evaluation Metrics
Accuracy & Loss Curves over training epochs

Confusion Matrix to visualize predictions

Classification Report with precision, recall, F1-score

⚙️ Tech Stack
Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

🚀 How to Run
Clone the repo

bash
Copy
Edit
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset

Get the GTSRB dataset from Kaggle or the official site

Place it inside a data/ folder

Run the CNN model

bash
Copy
Edit
python src/train_cnn.py
Run the MobileNet model

bash
Copy
Edit
python src/train_mobilenet.py
🖼️ Sample Outputs
Plots showing training accuracy/loss

Confusion matrix of predicted vs actual classes

Performance comparison of CNN vs MobileNet
