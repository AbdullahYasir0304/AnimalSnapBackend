from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

decode = tf.keras.applications.mobilenet_v2.decode_predictions

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    img_bytes = image.read()

    try:
        processed = preprocess_image(img_bytes)
        predictions = model.predict(processed)
        decoded = decode(predictions, top=1)[0][0]

        label = decoded[1]
        confidence = float(decoded[2])

        return jsonify({
            "animal": label.replace('_', ' ').title(),
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
