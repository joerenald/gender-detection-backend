from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

# Enable CORS
CORS(app)

# Load Models Once at Startup
print("Loading models...")

gender_model = tf.keras.models.load_model(
    "model/gender_model.keras",
    compile=False
)

age_model = tf.keras.models.load_model(
    "model/age_model.keras",
    compile=False
)

print("Models loaded successfully.")

labels = ["Male", "Female"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Health Check Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Gender Age Detection API is live"
    })


# Test Route
@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "success",
        "message": "Backend working properly"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No data received"
            }), 400

        if "image" not in data:
            return jsonify({
                "error": "Image field missing"
            }), 400

        image_data = data["image"]

        # Remove Base64 Header
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode Image
        img_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(
            img_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({
                "error": "Invalid image"
            }), 400

        # Face Detection
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            return jsonify({
                "gender": "No Face Detected",
                "age": "",
                "confidence": 0
            })

        # Use First Face
        x, y, w, h = faces[0]

        face = img[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (64, 64)
        )

        face = face.astype("float32") / 255.0

        face = np.expand_dims(
            face,
            axis=0
        )

        # Predictions
        gender_prediction = gender_model.predict(
            face,
            verbose=0
        )

        age_prediction = age_model.predict(
            face,
            verbose=0
        )

        gender_index = int(
            np.argmax(gender_prediction)
        )

        confidence = float(
            np.max(gender_prediction)
        ) * 100

        age = int(
            age_prediction[0][0]
        )

        result = {
            "gender": labels[gender_index],
            "age": age,
            "confidence": round(confidence, 2)
        }

        print("Prediction:", result)

        return jsonify(result)

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    app.run(
        host="0.0.0.0",
        port=port
    )