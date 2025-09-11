from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd

app = Flask(__name__)

# ======================
# Paths and model setup
# ======================
MODEL_PATH = os.path.join(os.getcwd(), "final_model.keras")
CSV_PATH = os.path.join(os.getcwd(), "train.csv")  # train.csv with id, landmark_id

# Globals
model = None
class_counts = None

# ======================
# Preprocess image
# ======================
def preprocess_image(file):
    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize
    img = cv2.resize(img, (224, 224))
    # Normalize
    img = img.astype("float32") / 255.0
    # Add batch dimension
    return np.expand_dims(img, axis=0)

# ======================
# Load model and CSV
# ======================
def load_model_and_data():
    global model, class_counts
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please download it first.")
    
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    # Load train.csv for class counts
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        class_counts = df["landmark_id"].value_counts().to_dict()
        print("Class counts loaded.")
    else:
        print("Warning: train.csv not found, class counts unavailable.")

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
        print("Preprocessing image...")
        img = preprocess_image(file)

        print("Predicting...")
        preds = model.predict(img)
        predicted_index = int(np.argmax(preds, axis=1)[0])
        print("Prediction done!")

        # Lookup class count
        sample_count = class_counts.get(predicted_index, "Unknown") if class_counts else "Unavailable"

        # Return formatted string
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
    # Force CPU to avoid GPU allocation issues if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Load model & CSV once at startup
    load_model_and_data()

    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=True)
