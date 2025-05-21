import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from flask import Flask, request, jsonify


MODEL_PATH = 'soil_texture_mobilenetv2.keras'
CROP_MODEL_PATH = 'crop_predictor.pkl'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

app = Flask(__name__)

def generate_synthetic_data():
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
    data = []
    for crop, reqs in crop_suitability.items():
        for _ in range(1000):
            soil_texture = np.random.choice(reqs['soil_texture'])
            temperature = np.random.uniform(reqs['temperature']['min'] - 5, reqs['temperature']['max'] + 5)
            humidity = np.random.uniform(reqs['humidity']['min'] - 10, reqs['humidity']['max'] + 10)
            rainfall = np.random.uniform(reqs['rainfall']['min'] - 100, reqs['rainfall']['max'] + 100)
            data.append([soil_texture, temperature, humidity, rainfall, crop])
    return pd.DataFrame(data, columns=['soil_texture', 'temperature', 'humidity', 'rainfall', 'crop'])

def train_crop_model():
    df = generate_synthetic_data()
    label_encoder = LabelEncoder()
    df['soil_texture'] = label_encoder.fit_transform(df['soil_texture'])
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    X = df[['soil_texture', 'temperature', 'humidity', 'rainfall']]
    y = df['crop']
    scaler = StandardScaler()
    X[['temperature', 'humidity', 'rainfall']] = scaler.fit_transform(X[['temperature', 'humidity', 'rainfall']])
    joblib.dump(scaler, SCALER_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, CROP_MODEL_PATH)
    return model

def train_texture_model():
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        for texture in ['sandy', 'loamy', 'clayey', 'alluvial']:
            os.makedirs(os.path.join(train_dir, texture), exist_ok=True)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)
        for texture in ['sandy', 'loamy', 'clayey', 'alluvial']:
            os.makedirs(os.path.join(val_dir, texture), exist_ok=True)
    has_images = False
    for texture in ['sandy', 'loamy', 'clayey', 'alluvial']:
        texture_dir = os.path.join(train_dir, texture)
        if os.path.exists(texture_dir) and len(os.listdir(texture_dir)) > 0:
            has_images = True
            break
    if not has_images:
        return None
    textures = ['sandy', 'loamy', 'clayey', 'alluvial']
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
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
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(textures), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=20,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[lr_scheduler]
    )
    model.save(MODEL_PATH)
    return model

def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_texture(img_path, model):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    texture_classes = ['sandy', 'loamy', 'clayey', 'alluvial']
    return texture_classes[predicted_class[0]]

def recommend_crop(soil_texture, temperature, humidity, rainfall):
    if not os.path.exists(CROP_MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        crop_model = train_crop_model()
    else:
        crop_model = joblib.load(CROP_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    input_data = pd.DataFrame([[soil_texture, temperature, humidity, rainfall]], 
                             columns=['soil_texture', 'temperature', 'humidity', 'rainfall'])
    input_data['soil_texture'] = label_encoder.transform([soil_texture])[0]
    input_data[['temperature', 'humidity', 'rainfall']] = scaler.transform(input_data[['temperature', 'humidity', 'rainfall']])
    prediction = crop_model.predict(input_data)
    return [{"crop": prediction[0], "suitability_score": 100.0}]

def recommend_fertilizer(soil_texture, crop_type, nitrogen, phosphorus, potassium):
    fertilizer_recommendations = {
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
    n_level = "Low" if nitrogen < 50 else "Medium" if nitrogen < 100 else "High"
    p_level = "Low" if phosphorus < 25 else "Medium" if phosphorus < 50 else "High"
    k_level = "Low" if potassium < 25 else "Medium" if potassium < 50 else "High"
    try:
        base_fertilizer = list(fertilizer_recommendations[crop_type][soil_texture].keys())[0]
        base_description = fertilizer_recommendations[crop_type][soil_texture][base_fertilizer]
    except KeyError:
        return {
            "fertilizer": "Unknown",
            "reason": "No fertilizer recommendation found for the given crop and soil texture combination."
        }
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

def load_models():
    models = {}
    if os.path.exists(MODEL_PATH):
        try:
            models['texture'] = load_model(MODEL_PATH)
        except Exception as e:
            model = train_texture_model()
            if model:
                models['texture'] = model
            else:
                return None, "Failed to train the texture model"
    else:
        model = train_texture_model()
        if model:
            models['texture'] = model
        else:
            return None, "Failed to train the texture model"
    if not os.path.exists(CROP_MODEL_PATH):
        train_crop_model()
    return models, True

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    if not content:
        return jsonify({"error": "No JSON data provided"}), 400
    image_path = content.get('image')
    temperature = content.get('temperature')
    humidity = content.get('humidity')
    rainfall = content.get('rainfall')
    crop_type = content.get('crop_type')
    nitrogen = content.get('nitrogen')
    phosphorus = content.get('phosphorus')
    potassium = content.get('potassium')
    if not all([image_path, temperature, humidity, rainfall, crop_type, nitrogen, phosphorus, potassium]):
        return jsonify({"error": "Missing required parameters"}), 400
    if not os.path.exists(image_path):
        return jsonify({"error": "Image file not found"}), 400
    models, status = load_models()
    if not models:
        return jsonify({"error": status}), 500
    soil_texture = predict_texture(image_path, models['texture'])
    crop_recommendations = recommend_crop(soil_texture, temperature, humidity, rainfall)
    fertilizer_recommendation = recommend_fertilizer(soil_texture, crop_type, nitrogen, phosphorus, potassium)
    return jsonify({
        "soil_texture": soil_texture,
        "crop_recommendations": crop_recommendations,
        "fertilizer_recommendation": fertilizer_recommendation
    })

if __name__ == "__main__":
    models, status = load_models()
    if not models:
        print(f"Error: {status}")
    else:
        app.run(debug=True)