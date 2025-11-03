# Face-Mask-Detection-Using-CNN
## 📘 Project Overview
The **Face Mask Detection** project is a deep learning-based system designed to automatically detect whether a person is wearing a face mask or not.  
Using a **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras**, the model classifies facial images into two categories — **With Mask** and **Without Mask**.  

The model is deployed as a simple **Streamlit web app**, where users can upload an image and instantly get the prediction result.

---

## 🧠 Features
- Detects if a person is wearing a face mask or not  
- Uses a trained **CNN model** for high accuracy  
- **Streamlit** interface for easy web deployment  
- Works efficiently on CPU or GPU  
- Simple to train and extend with new data  

---

## 🧩 Technologies Used
- **Programming Language:** Python  
- **Deep Learning Frameworks:** TensorFlow, Keras  
- **Other Libraries:** NumPy, OpenCV, Streamlit, Matplotlib  
- **Dataset:** [Face Mask Detection Dataset](https://www.kaggle.com/datasets)  

---

## 📂 Project Structure
ace Mask Detection/
│
├── data/
│ ├── with_mask/
│ └── without_mask/
│
├── mask_detector_model.h5 # Trained CNN model
├── train_model.py # Model training script
├── app.py # Streamlit web app
