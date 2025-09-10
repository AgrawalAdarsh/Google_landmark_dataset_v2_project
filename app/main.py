from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import requests

# Initialize Flask app
app = Flask(__name__)

# Path to local model
MODEL_PATH = os.path.join(os.getcwd(), "final_model.keras")

# Download the model if not present (for cloud deployment)
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?export=download&id=1UGfgPYFZvwq3jmDpfTNJ65nQKFQzpGFa"  # direct download link
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete!")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


# Preprocess uploaded image
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # must match training size
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


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
        img = preprocess_image(file)
        preds = model.predict(img)
        predicted_class = int(np.argmax(preds, axis=1)[0])
        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/healthz")
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
