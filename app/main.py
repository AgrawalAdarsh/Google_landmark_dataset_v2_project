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
BASE_DIR = os.getcwd()  # Root of project
MODEL_PATH = os.path.join(BASE_DIR, "final_model.keras")
CSV_PATH = os.path.join(BASE_DIR, "train.csv")  # train.csv with id, landmark_id

# Google Drive IDs
MODEL_GDRIVE_ID = "1UGfgPYFZvwq3jmDpfTNJ65nQKFQzpGFa"
CSV_GDRIVE_ID = "15e4OKT1sMt4ESUpYzGZiRIS9D7FimR1x"

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
# Download file if missing
# ======================
def download_if_missing(file_path, gdrive_id):
    if not os.path.exists(file_path):
        print(f"Downloading {os.path.basename(file_path)} from Google Drive...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, file_path, quiet=False)
        print(f"{os.path.basename(file_path)} download complete!")

# ======================
# Load model + CSV
# ======================
def load_model():
    global model, class_counts

    if model is None:
        # Ensure model and CSV exist
        download_if_missing(MODEL_PATH, MODEL_GDRIVE_ID)
        download_if_missing(CSV_PATH, CSV_GDRIVE_ID)

        # Load model
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")

        # Load CSV for class counts
        df = pd.read_csv(CSV_PATH)
        class_counts = df["landmark_id"].value_counts().to_dict()
        print("Class counts loaded.")

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
