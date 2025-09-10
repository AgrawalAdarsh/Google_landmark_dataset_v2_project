from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
import pandas as pd

app = Flask(__name__)

# Path to model
MODEL_PATH = os.path.join(os.getcwd(), "final_model.keras")
TRAIN_CSV = os.path.join(os.getcwd(), "train.csv")

# Google Drive file ID (set your model ID here)
GDRIVE_FILE_ID = "1UGfgPYFZvwq3jmDpfTNJ65nQKFQzpGFa"

# Global variable for lazy loading
model = None
df = None   # dataframe for labels

# Preprocess image
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# Function to load the model (lazy)
def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model using gdown...")
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            print("Download complete!")
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")

# Load train.csv for label counts
def load_labels():
    global df
    if df is None and os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "Error: No file uploaded", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "Error: No file selected", 400
    
    try:
        # Lazy load model + labels
        load_model()
        load_labels()

        img = preprocess_image(file)
        preds = model.predict(img)
        predicted_index = int(np.argmax(preds, axis=1)[0])

        # Count samples in that class (from train.csv)
        sample_cnt = None
        if df is not None:
            sample_cnt = (df["landmark_id"] == predicted_index).sum()

        # Format output
        result = f"Predicted Label: {predicted_index}"
        if sample_cnt is not None:
            result += f"\nSamples in class {predicted_index}: {sample_cnt}"

        return result

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/healthz")
def health():
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
