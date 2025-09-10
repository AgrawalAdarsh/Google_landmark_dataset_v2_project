from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
import pandas as pd

app = Flask(__name__)

# ======================
# Paths and model setup
# ======================
MODEL_PATH = os.path.join(os.getcwd(), "final_model.keras")
CSV_PATH = os.path.join(os.getcwd(), "train.csv")  # train.csv with id, landmark_id
GDRIVE_FILE_ID = "1UGfgPYFZvwq3jmDpfTNJ65nQKFQzpGFa"

# Globals
model = None
class_counts = None

# ======================
# Preprocess image
# ======================
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ======================
# Lazy load model + CSV
# ======================
def load_model():
    global model, class_counts
    if model is None:
        # Download model if not present
        if not os.path.exists(MODEL_PATH):
            print("Downloading model using gdown...")
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            print("Download complete!")

        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")

        # Load train.csv and compute counts
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            class_counts = df["landmark_id"].value_counts().to_dict()
            print("Class counts loaded.")
        else:
            print("Warning: train.csv not found, sample counts unavailable.")

# ======================
# Routes
# ======================
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
        # Load model and counts if needed
        load_model()

        img = preprocess_image(file)
        preds = model.predict(img)
        predicted_index = int(np.argmax(preds, axis=1)[0])

        # Lookup class count
        sample_count = class_counts.get(predicted_index, "Unknown") if class_counts else "Unavailable"

        # Return nice formatted string
        return (
            f"Label: {predicted_index}\n"
            f"Classified as: {predicted_index}\n"
            f"Samples in class {predicted_index}: {sample_count}"
        )

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/healthz")
def health():
    return "OK"

# ======================
# Run
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
