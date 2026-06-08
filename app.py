from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# =========================
# Load Models
# =========================

gender_model = tf.keras.models.load_model("model/gender_model.keras")
age_model = tf.keras.models.load_model("model/age_model.keras")
race_model = tf.keras.models.load_model("model/race_model.keras")

labels = ["Male", "Female"]
race_labels = ["White", "Black", "Asian", "Indian", "Others"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Health Check Route
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Gender Detection API is live"
    })

# =========================
# Prediction Route
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "No image provided"
            }), 400

        image_data = data["image"]

        # Remove base64 header if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "error": "Invalid image"
            }), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            return jsonify({
                "gender": "No Face Detected",
                "age": None,
                "race": None,
                "confidence": 0
            })

        # Use first detected face
        x, y, w, h = faces[0]

        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = face.reshape(1, 64, 64, 3)

        # Predictions
        gender_prediction = gender_model.predict(face, verbose=0)
        age_prediction = age_model.predict(face, verbose=0)
        race_prediction = race_model.predict(face, verbose=0)

        gender_index = int(np.argmax(gender_prediction))
        race_index = int(np.argmax(race_prediction))

        result = {
            "gender": labels[gender_index],
            "age": int(age_prediction[0][0]),
            "race": race_labels[race_index],
            "confidence": round(
                float(np.max(gender_prediction)) * 100,
                2
            )
        }

        print("Prediction:", result)

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# Run App
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)