# 🏰 Google Landmark Detection v2 – Implementation

This is a **self-learning project** using the **Google Landmark Detection Dataset v2** to build a landmark prediction model.  
It combines **deep learning** with a **Flask web app** that allows users to upload images and get predictions of famous landmarks.

**🌍 Live Application:** [Landmark Detection App](https://landmark-app.onrender.com)

---

## 📖 Overview
The application leverages a **Convolutional Neural Network (CNN)** trained on Google’s landmark dataset to classify and predict landmarks from images.  
Users can interact with the model through a clean and simple **web interface**.

---

## ✨ Features
- 📸 Upload an image for landmark recognition  
- ⚡ Real-time predictions using a trained deep learning model  
- 🖥️ User-friendly web interface  
- 🚀 Deployed online for easy access  

---

## 🛠️ Technologies Used
- **Backend:** Python, Flask  
- **Deep Learning:** TensorFlow, Keras  
- **Frontend:** HTML, CSS  
- **Deployment:** Render  

---

## ⚙️ How It Works
1. **Image Upload:** User selects an image from their device.  
2. **Prediction:** The CNN model processes the image and predicts the landmark.  
3. **Result Display:** The predicted landmark name is shown in the web app.  

---

---

> **⚠️ Note:**  
> This project uses only **~2GB of the Google Landmark Detection Dataset v2** (the original dataset is ~500GB).  
> Because of this limitation:  
> - The model recognizes only a **small subset of landmarks**.  
> - Predictions may sometimes repeat or seem “stuck” on a few classes.  
> - Running on **free-tier hosting (Render)** also means inference may take a few seconds.  
>
> This is expected behavior and part of the learning-focused nature of this project.
> Currently working to resolve this issue....

---

---

## 📂 Project Structure
Google_landmark_dataset_v2_project/
+-- app/                      
|   +-- static/               
|   |   +-- images/          
|   |       +-- profilepic.jpeg 
|   +-- templates/            
|   |   +-- index.html        
|   +-- main.py              
+-- model_loader.py          
+-- final_model.keras         
+-- requirements.txt         
+-- Procfile                  


---

## 🖼️ Screenshots & Example Predictions

**🏠 Home Page / About the Project**  
![Home Page](./screenshots/d6f9c1cb-9570-4dd0-a3a6-e25e1aebb763.png)

**📤 Try It Yourself / Image Upload**  
![Try It Yourself](./screenshots/31ad0ab1-809c-4720-b18a-67a955c5287e.png)

**✅ Prediction Example**  
![Prediction Example](./screenshots/84e7c957-f88f-459d-9d37-c4277c2203d9.png)

---

---

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/AgrawalAdarsh/Google_landmark_dataset_v2_project.git
   cd Google_landmark_dataset_v2_project

---   

---

## 📊 Dataset

This project uses the Google Landmark Detection Dataset v2, one of the largest publicly available landmark recognition datasets.

---

---

## 🔮 Future Work

1. Improve accuracy using transfer learning (EfficientNet, ResNet).
2. Add Top-5 predictions with confidence scores.
and many more...

---

---

## 👨‍💻 Contributor & Developer **Adarsh Agrawal**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/adarsh-agrawal-3b0a76268/)

---
