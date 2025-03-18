from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = load_model('../soil_texture_mobilenetv2.keras')

# Image size for preprocessing
IMG_SIZE = (224, 224)

# Function to preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the texture
def predict_texture(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    texture_classes = ['sandy', 'loamy', 'clayey', 'alluvial']
    return texture_classes[predicted_class[0]]

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_path = 'uploaded_image.png'
    img_file.save(img_path)
    predicted_texture = predict_texture(img_path)
    return jsonify({'predicted_texture': predicted_texture})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)