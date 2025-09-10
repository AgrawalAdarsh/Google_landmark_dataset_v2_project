from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown

app = Flask(__name__)

# Path to model
MODEL_PATH = os.path.join(os.getcwd(), "final_model.keras")

# Google Drive file ID (set your model ID here)
GDRIVE_FILE_ID = "1UGfgPYFZvwq3jmDpfTNJ65nQKFQzpGFa"

# Global variable for lazy loading
model = None

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    
    try:
        # Lazy load model if not loaded
        load_model()

        img = preprocess_image(file)
        preds = model.predict(img)
        predicted_class = int(np.argmax(preds, axis=1)[0])

        # Return only number
        return jsonify(predicted_class=predicted_class)

    except Exception as e:
        return jsonify(error=str(e))

@app.route("/healthz")
def health():
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
