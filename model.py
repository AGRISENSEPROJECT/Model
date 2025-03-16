import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
import cv2

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# --- Soil Texture Prediction (Image) ---
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Train directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

textures = ['sandy', 'loamy', 'clayey', 'alluvial']

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(textures), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Save the trained model
model.save('soil_texture_mobilenetv2.keras')

# Load the saved model for predictions
model = tf.keras.models.load_model('soil_texture_mobilenetv2.keras')

def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_texture(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    texture_classes = list(train_generator.class_indices.keys())
    return texture_classes[predicted_class[0]]

# --- Crop Recommendation ---
def recommend_crop(soil_texture, temperature, humidity):
    # Define crop recommendations based on soil texture and environmental conditions
    crop_mapping = {
        'sandy': ['Tomatoes', 'Carrots', 'Radishes', 'Sweet Potatoes', 'Melons'],
        'loamy': ['Wheat', 'Corn (Maize)', 'Beans', 'Cucumbers', 'Lettuce'],
        'clayey': ['Cabbage', 'Broccoli', 'Cauliflower', 'Peas', 'Potatoes'],
        'alluvial': ['Rice', 'Sugarcane', 'Bananas', 'Cotton', 'Jute']
    }
    # Filter crops based on scope (Irish Potatoes, Tomatoes, Beans)
    scope_crops = ['Irish Potatoes', 'Tomatoes', 'Beans']
    recommended = crop_mapping.get(soil_texture, ['Unknown'])
    return [crop for crop in recommended if crop in scope_crops]

# --- Fertilizer Recommendation ---
def predict_fertilizer(input_data):
    # Load the model and preprocessing objects
    model = joblib.load('fertilizer_predictor.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    for column in ['Soil Type', 'Crop Type']:
        input_df[column] = label_encoders[column].transform([input_df[column]])

    # Standardize the features
    input_scaled = scaler.transform(input_df)

    # Predict the fertilizer
    prediction = model.predict(input_scaled)
    fertilizer = label_encoders['Fertilizer Name'].inverse_transform(prediction)

    return fertilizer[0]

# --- Flask API ---
@app.route('/predict', methods=['POST'])
def predict():
    # Get image and static data from request
    img_file = request.files['image']
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    soil_type = request.form['soil_type']

    # Save the uploaded image
    img_path = 'uploaded_image.png'
    img_file.save(img_path)

    # Predict soil texture from image
    predicted_texture = predict_texture(img_path)

    # Recommend crop
    recommended_crops = recommend_crop(predicted_texture, temperature, humidity)

    # Predict fertilizer for the first recommended crop
    if recommended_crops:
        crop_type = recommended_crops[0]
        fertilizer_input = {
            'Temparature': temperature,
            'Humidity ': humidity,
            'Moisture': 50,  # Static for now
            'Soil Type': soil_type,
            'Crop Type': crop_type,
            'Nitrogen': 120,  # Static for now
            'Potassium': 35,  # Static for now
            'Phosphorous': 10  # Static for now
        }
        recommended_fertilizer = predict_fertilizer(fertilizer_input)
    else:
        recommended_fertilizer = 'Unknown'

    # Return results
    return jsonify({
        'predicted_texture': predicted_texture,
        'recommended_crops': recommended_crops,
        'recommended_fertilizer': recommended_fertilizer
    })

if __name__ == '__main__':
    app.run(debug=True)