from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

# CORS
CORS(
    app,
    resources={r"/*": {"origins": "*"}}
)

print("Loading models...")

# Load models
gender_model = tf.keras.models.load_model("model/gender_model.keras")
age_model = tf.keras.models.load_model("model/age_model.keras")

print("Models loaded successfully.")

labels = ["Male", "Female"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Home Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Gender Age Detection API is live"
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:

        print("Prediction request received")

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON received"
            }), 400

        if "image" not in data:
            return jsonify({
                "error": "No image key found"
            }), 400

        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",")[1]

        print("Base64 decoded")

        img_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "error": "Image decoding failed"
            }), 400

        print("Image decoded")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        print(f"Faces found: {len(faces)}")

        if len(faces) == 0:
            return jsonify({
                "gender": "No Face Detected",
                "age": "",
                "confidence": 0
            })

        x, y, w, h = faces[0]

        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (64, 64))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)

        print("Running Gender Model")

        gender_prediction = gender_model.predict(
            face,
            verbose=0
        )

        print("Running Age Model")

        age_prediction = age_model.predict(
            face,
            verbose=0
        )

        gender_index = int(np.argmax(gender_prediction))

        confidence = float(
            np.max(gender_prediction)
        ) * 100

        age = int(age_prediction[0][0])

        result = {
            "gender": labels[gender_index],
            "age": age,
            "confidence": round(confidence, 2)
        }

        print("Result:", result)

        return jsonify(result)

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )