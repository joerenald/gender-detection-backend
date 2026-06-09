from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os

app = Flask(__name__)

# Enable CORS
CORS(app)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Health Check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Gender Age Detection API Test Mode"
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

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
                "age": "",
                "confidence": 0
            })

        # Sample fixed response
        result = {
            "gender": "Male",
            "age": 23,
            "confidence": 95.5
        }

        print("TEST RESPONSE:", result)

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