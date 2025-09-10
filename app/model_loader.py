import tensorflow as tf

MODEL_PATH = "final_model.keras"


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
