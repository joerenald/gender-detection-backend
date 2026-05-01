from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load trained model
gender_model = tf.keras.models.load_model("model/gender_model.keras")
age_model = tf.keras.models.load_model("model/age_model.keras")
race_model = tf.keras.models.load_model("model/race_model.keras")

labels = ["Male", "Female"]
race_labels = ["White", "Black", "Asian", "Indian", "Others"]
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]

        if "," in data:
         image_data = data.split(",")[1]
        else:
         image_data = data
        img_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)


        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
         return jsonify({
          "gender": "No Face Detected",
          "confidence": 0
        })

        # Crop first detected face
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]

        img = cv2.resize(face, (64,64))

        img = img / 255.0

        img = img.reshape(1,64,64,3)

        gender_prediction = gender_model.predict(img, verbose=0)
        age_prediction = age_model.predict(img, verbose=0)
        race_prediction = race_model.predict(img, verbose=0)

        index = int(np.argmax(gender_prediction))
        race_index = int(np.argmax(race_prediction))
        race = race_labels[race_index]

        confidence = float(np.max(gender_prediction)) * 100

        age = int(age_prediction[0][0])

        result = {
    "gender": labels[index],
    "age": age,
    "race": race,   # 👈 NEW
    "confidence": round(confidence,2)
}

        print("Prediction:", result)

        return jsonify(result)

    except Exception as e:

        print("ERROR:", e)

        return jsonify({"error": str(e)}),500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)