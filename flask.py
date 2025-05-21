# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import numpy as np
# import cv2
# import logging

# app = Flask(__name__)
# CORS(app)

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# # Load your model
# MODEL_PATH = "soil_texture_mobilenetv2.keras"
# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
#     logging.info("✅ Model loaded successfully.")
# except Exception as e:
#     logging.error(f"❌ Error loading model: {str(e)}")
#     model = None


# def preprocess_image(file):
#     try:
#         in_memory_file = file.read()
#         npimg = np.frombuffer(in_memory_file, np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (224, 224))
#         img = img / 255.0
#         return np.expand_dims(img, axis=0)
#     except Exception as e:
#         raise ValueError(f"Invalid image format: {str(e)}")


# @app.route("/analyze", methods=["POST"])
# def analyze():
#     if not model:
#         return jsonify({"success": False, "message": "Model not loaded"}), 500

#     if "image" not in request.files:
#         return jsonify({"success": False, "message": "No image uploaded"}), 400

#     file = request.files["image"]
#     try:
#         processed = preprocess_image(file)
#         prediction = model.predict(processed)[0][0]
#         result = "Suitable" if prediction > 0.5 else "Not Suitable"

#         return jsonify({
#             "success": True,
#             "result": result,
#             "confidence": float(prediction)
#         })

#     except ValueError as ve:
#         return jsonify({"success": False, "message": str(ve)}), 400
#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return jsonify({"success": False, "message": "Internal server error"}), 500


# @app.route("/")
# def index():
#     return jsonify({"message": "Welcome to AgriSense Soil API!"})


# if __name__ == "__main__":
#     app.run(debug=True)
