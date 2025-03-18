from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)

model = load_model('soil_texture_mobilenetv2.keras')

IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_texture(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    texture_classes = ['sandy', 'loamy', 'clayey', 'alluvial']
    return texture_classes[predicted_class[0]]

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_path = 'uploaded_image.png'
    img_file.save(img_path)
    predicted_texture = predict_texture(img_path)
    return jsonify({'predicted_texture': predicted_texture})

if __name__ == '__main__':
    app.run(debug=True)


    from flask import Flask, request, jsonify
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('my_model')  # Replace 'my_model' with your model path

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json
    input_data = data['input']

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)