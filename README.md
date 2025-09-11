# GoogleLandmarkDetectionv2_implementation

This is a self-learning project using the **Google Landmark Detection Dataset v2** to build a landmark prediction model. The project combines deep learning with a web interface to allow users to upload images and get predictions of famous landmarks.

## Overview

The application leverages a **Convolutional Neural Network (CNN)** trained on Google’s landmark dataset to classify and predict landmarks from images. Users can interact with the model via a simple web interface.

**Live application:** [Landmark Detection App](https://landmark-app.onrender.com)

## Features

- Upload an image for landmark recognition.  
- Real-time predictions using a trained deep learning model.  
- User-friendly web interface for easy access.  

## Technologies Used

- **Backend:** Python, Flask  
- **Deep Learning:** TensorFlow, Keras  
- **Frontend:** HTML, CSS
- **Deployment:** Render  

## How It Works

1. **Image Upload:** User selects an image from their device.  
2. **Prediction:** The CNN model predicts the landmark.  
3. **Result Display:** The predicted landmark is displayed in the web interface.  

## Project Structure
Google_landmark_dataset_v2_project/
│
├─ app/
│ ├─ static/
│ │ └─ images/
│ │ └─ profilepic.jpeg
│ └─ templates/
│ └─ index.html
├─ main.py
├─ model_loader.py
├─ final_model.keras
├─ requirements.txt
├─ Procfile
├─ runtime.txt



## Screenshots & Example Predictions

**Home Page / About the Project:**  

![Home Page](./screenshots/d6f9c1cb-9570-4dd0-a3a6-e25e1aebb763.png)

**Try It Yourself / Image Upload:**  

![Try It Yourself](./screenshots/31ad0ab1-809c-4720-b18a-67a955c5287e.png)

**Prediction Example:**  

![Prediction Example](./screenshots/84e7c957-f88f-459d-9d37-c4277c2203d9.png)

> Make sure to place your images inside a folder named `screenshots` in your repo.

---

## Video Demonstration

You can add a video to show the working of your project. Place your video inside a folder named `videos` (or any folder you like). Here's how to include it:

**Option 1: Using HTML `<video>` tag (GitHub supported for local videos in repos)**

<video width="640" height="360" controls>
  <source src="./videos/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Option 2: Linking to a video hosted online (YouTube, Google Drive, etc.)**

[Watch Demo Video](https://www.youtube.com/watch?v=YOUR_VIDEO_LINK)

