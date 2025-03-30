# Handwritten Digit Recognition Using CNN

## 📄 Project Overview
This project implements a **Convolutional Neural Network (CNN)** model to classify handwritten digits using the **MNIST dataset**. The goal is to create an accurate and efficient digit recognition system, suitable for real-world applications like **postal sorting, bank check processing, and form digitization**.  

---

## 🚀 Features
- **Deep Learning Model:** Utilizes a CNN architecture inspired by **LeNet-5** for efficient feature extraction and classification.  
- **Data Augmentation:** Applies techniques such as rotation, zooming, and flipping to expand the dataset and enhance model generalization.  
- **High Accuracy:** Achieves reliable predictions on handwritten digits, making it suitable for automation tasks.  
- **Performance Optimization:** Uses **RMSProp** optimizer and **ReduceLROnPlateau** to dynamically adjust the learning rate, preventing stagnation.  
- **GPU Support:** Leverages **Kaggle’s GPU** for faster training and reduced computation time.  

---

## 🛠️ Technologies Used
- **Python**: Programming language for model development and data processing.  
- **TensorFlow & Keras**: Deep learning frameworks used to build and train the CNN model.  
- **Pandas & NumPy**: For data manipulation and preprocessing.  
- **Matplotlib & Seaborn**: For visualizing the data distribution and model performance.  

---

## 📊 Dataset  
- **MNIST Handwritten Digit Dataset**: Contains **60,000 training images** and **10,000 testing images** of handwritten digits (0-9), with each image having a **28x28** grayscale pixel resolution.  

---

## ⚙️ Model Architecture
The CNN model follows the **LeNet-5** structure:  


- **Conv2D Layers:** Extract features from the images.  
- **MaxPooling Layers:** Reduce spatial dimensions, preventing overfitting.  
- **Dropout Layers:** Improve generalization by randomly deactivating neurons.  
- **Flatten Layer:** Converts feature maps into a vector.  
- **Dense Layers:** Perform the final classification.  

---

## 🔥 Model Evaluation
- **Training Accuracy:** Consistently improves over epochs.  
- **Validation Accuracy:** Closely aligns with training accuracy, indicating minimal overfitting.  
- **Confusion Matrix:** Reveals strong classification performance with minimal misclassifications.  

---

## 📈 Results
- The model achieves **high accuracy** on both training and validation datasets.  
- **Predictions** on test data are saved in a CSV file, making it suitable for submission in competitions or deployment.  

---

## 📚 Installation & Usage

### 1. Clone the Repository  
```bash```
git clone https://github.com/YourUsername/Handwritten-Digit-Recognition-CNN.git

pip install -r requirements.txt


/data  
    ├── train.csv            # Training dataset  
    ├── test.csv             # Test dataset  
/src  
    ├── CNN_DLM_PROJECT_055007.ipynb  # Jupyter notebook with the model code  
/output  
    ├── predictions.csv      # Model predictions saved in CSV format  
/README.md                   # Project documentation 

Author:
Ashish Michael Chauhan (055007)
