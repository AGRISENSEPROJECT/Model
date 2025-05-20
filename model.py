import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
import os

# Model paths
MODEL_PATH = 'soil_texture_mobilenetv2.keras'
FERTILIZER_MODEL_PATH = 'fertilizer_predictor.pkl'
LABEL_ENCODERS_PATH = 'label_encoders.pkl'
SCALER_PATH = 'scaler.pkl'
#this is
# Image size for preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Function to train the model
def train_model():
    print("Training new soil texture model...")
    
    # Define data directories
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    
    # Check if data directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        print(f"Created training directory: {train_dir}")
        # Create subdirectories for each texture class
        for texture in ['sandy', 'loamy', 'clayey', 'alluvial']:
            os.makedirs(os.path.join(train_dir, texture), exist_ok=True)
        print("Please add training images to the respective soil texture folders in the 'data/train' directory")
    
    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)
        print(f"Created validation directory: {val_dir}")
        # Create subdirectories for each texture class
        for texture in ['sandy', 'loamy', 'clayey', 'alluvial']:
            os.makedirs(os.path.join(val_dir, texture), exist_ok=True)
        print("Please add validation images to the respective soil texture folders in the 'data/validation' directory")
    
    # Check if there are images in the training directory
    has_images = False
    for texture in ['sandy', 'loamy', 'clayey', 'alluvial']:
        texture_dir = os.path.join(train_dir, texture)
        if os.path.exists(texture_dir) and len(os.listdir(texture_dir)) > 0:
            has_images = True
            break
    
    if not has_images:
        print("No training images found. Please add images to the training directories.")
        return None
    
    textures = ['sandy', 'loamy', 'clayey', 'alluvial']
    
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
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
    
    # Create the model
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
    print("Starting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )
    
    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model

# Function to preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the texture
def predict_texture(img_path, model):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    texture_classes = ['sandy', 'loamy', 'clayey', 'alluvial']
    return texture_classes[predicted_class[0]]

# Crop Recommendation function
def recommend_crop(soil_texture, temperature, humidity, rainfall):
    # Define crop recommendations based on soil texture and environmental conditions
    # Focusing on rice, Irish potatoes, and tomatoes
    crop_suitability = {
        'rice': {
            'soil_texture': ['alluvial', 'clayey'],
            'temperature': {'min': 20, 'max': 35},
            'humidity': {'min': 60, 'max': 90},
            'rainfall': {'min': 800, 'max': 2000}
        },
        'Irish Potatoes': {
            'soil_texture': ['loamy', 'sandy'],
            'temperature': {'min': 15, 'max': 25},
            'humidity': {'min': 60, 'max': 85},
            'rainfall': {'min': 500, 'max': 700}
        },
        'Tomatoes': {
            'soil_texture': ['sandy', 'loamy'],
            'temperature': {'min': 18, 'max': 29},
            'humidity': {'min': 50, 'max': 70},
            'rainfall': {'min': 400, 'max': 600}
        }
    }
    
    # Calculate suitability scores for each crop
    scores = {}
    for crop, requirements in crop_suitability.items():
        # Soil texture match (binary: 1 if matched, 0 if not)
        soil_score = 1 if soil_texture in requirements['soil_texture'] else 0
        
        # Temperature suitability (0 to 1 scale)
        temp_min = requirements['temperature']['min']
        temp_max = requirements['temperature']['max']
        if temperature < temp_min:
            temp_score = max(0, 1 - (temp_min - temperature) / temp_min)
        elif temperature > temp_max:
            temp_score = max(0, 1 - (temperature - temp_max) / temp_max)
        else:
            temp_score = 1
        
        # Humidity suitability (0 to 1 scale)
        hum_min = requirements['humidity']['min']
        hum_max = requirements['humidity']['max']
        if humidity < hum_min:
            hum_score = max(0, 1 - (hum_min - humidity) / hum_min)
        elif humidity > hum_max:
            hum_score = max(0, 1 - (humidity - hum_max) / hum_max)
        else:
            hum_score = 1
            
        # Rainfall suitability (0 to 1 scale)
        rain_min = requirements['rainfall']['min']
        rain_max = requirements['rainfall']['max']
        if rainfall < rain_min:
            rain_score = max(0, 1 - (rain_min - rainfall) / rain_min)
        elif rainfall > rain_max:
            rain_score = max(0, 1 - (rainfall - rain_max) / rain_max)
        else:
            rain_score = 1
        
        # Overall score (weighted average)
        scores[crop] = (soil_score * 0.4) + (temp_score * 0.2) + (hum_score * 0.2) + (rain_score * 0.2)
    
    # Sort crops by score
    recommended_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return crops and their suitability scores
    return [{"crop": crop, "suitability_score": round(score * 100, 1)} for crop, score in recommended_crops]

# Fertilizer Recommendation function
def predict_fertilizer(soil_texture, crop_type, nitrogen, phosphorus, potassium, temperature, humidity):
    # Create a structured dataset for fertilizer recommendations
    fertilizer_data = {
        'rice': {
            'sandy': {'NPK 10-10-10': 'Low nitrogen content. Good for initial growth stages.'},
            'loamy': {'Urea': 'High in nitrogen, good for vegetative growth of rice in loamy soil.'},
            'clayey': {'Ammonium Sulfate': 'Provides nitrogen and sulfur, good for clayey soils.'},
            'alluvial': {'DAP': 'High phosphorus content helps in root development in alluvial soils.'}
        },
        'Irish Potatoes': {
            'sandy': {'NPK 5-10-10': 'Low nitrogen, high potassium for tuber development in sandy soil.'},
            'loamy': {'NPK 10-20-20': 'Balanced for overall growth, with focus on tuber development.'},
            'clayey': {'Triple Super Phosphate': 'High phosphorus helps with root development in heavy soil.'},
            'alluvial': {'MOP': 'High potassium content for improving quality and disease resistance.'}
        },
        'Tomatoes': {
            'sandy': {'NPK 5-10-5': 'Balanced nutrients with focus on phosphorus for flowering.'},
            'loamy': {'NPK 8-32-16': 'High phosphorus promotes flowering and fruiting in tomatoes.'},
            'clayey': {'Single Super Phosphate': 'Good for breaking down clayey soil and providing phosphorus.'},
            'alluvial': {'NPK 12-12-17': 'Higher potassium content for fruit development and quality.'}
        }
    }
    
    # N-P-K level classification
    n_level = "Low" if nitrogen < 50 else "Medium" if nitrogen < 100 else "High"
    p_level = "Low" if phosphorus < 25 else "Medium" if phosphorus < 50 else "High"
    k_level = "Low" if potassium < 25 else "Medium" if potassium < 50 else "High"
    
    # Get base fertilizer recommendation
    try:
        base_fertilizer = list(fertilizer_data[crop_type][soil_texture].keys())[0]
        base_description = fertilizer_data[crop_type][soil_texture][base_fertilizer]
    except KeyError:
        # If exact combination not found, provide a general recommendation
        base_fertilizer = "NPK 14-14-14"
        base_description = "General purpose balanced fertilizer suitable for most crops."
    
    # Additional recommendations based on N-P-K levels
    additional_recommendations = []
    
    if n_level == "Low":
        additional_recommendations.append("Consider adding urea or ammonium sulfate to increase nitrogen levels.")
    if p_level == "Low":
        additional_recommendations.append("Add bone meal or rock phosphate to improve phosphorus content.")
    if k_level == "Low":
        additional_recommendations.append("Apply potash or wood ash to increase potassium levels.")
    
    return {
        "recommended_fertilizer": base_fertilizer,
        "description": base_description,
        "soil_npk_status": f"N: {n_level}, P: {p_level}, K: {k_level}",
        "additional_recommendations": additional_recommendations
    }

# Load models function
def load_models():
    models = {}
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}")
        response = input("Do you want to rebuild the model? (y/n): ")
        if response.lower() == 'y':
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
                print(f"Deleted existing model at {MODEL_PATH}")
            model = train_model()
            if model:
                models['texture'] = model
            else:
                return None, "Failed to train the model"
        else:
            try:
                models['texture'] = load_model(MODEL_PATH)
                print("Model loaded successfully")
            except Exception as e:
                return None, f"Error loading model: {str(e)}"
    else:
        print(f"No model found at {MODEL_PATH}")
        model = train_model()
        if model:
            models['texture'] = model
        else:
            return None, "Failed to train the model"
    
    return models, True

# Main function to run when the script is executed directly
if __name__ == "__main__":
    models, status = load_models()
    if not models:
        print(f"Error: {status}")
    else:
        print("Models loaded successfully and ready for use.")
        print("You can now run app.py to start the API server.")
