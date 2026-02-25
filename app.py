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
from flask_cors import CORS
from flasgger import Swagger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import uuid
from data import CROP_SUITABILITY, FERTILIZER_RECOMMENDATIONS

MODEL_PATH = 'soil_texture_mobilenetv2.keras'
CROP_MODEL_PATH = 'crop_predictor.pkl'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_synthetic_data():
    data = []
    for crop, reqs in CROP_SUITABILITY.items():
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
    model.fit(
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
    supported_textures = ['sandy', 'loamy', 'clayey', 'alluvial']
    if soil_texture not in supported_textures:
        return {
            "error": f"Unsupported soil texture: '{soil_texture}'. Our system currently supports: {', '.join(supported_textures)}.",
            "message": "For specialized soil types not in our dataset, we recommend consulting with local agricultural advisors.",
            "supported_textures": supported_textures
        }

    if not all(os.path.exists(p) for p in [CROP_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]):
        crop_model = train_crop_model()
    else:
        crop_model = joblib.load(CROP_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

    all_crops = list(CROP_SUITABILITY.keys())
    recommendations = []

    for crop in all_crops:
        reqs = CROP_SUITABILITY[crop]
        score = 0.0

        # Soil texture suitability (40% weight) - MUST match for high score
        if soil_texture in reqs['soil_texture']:
            score += 40
        else:
            score += 0  # No points if soil texture doesn't match

        # Temperature suitability (20% weight) - MUST be in range
        if reqs['temperature']['min'] <= temperature <= reqs['temperature']['max']:
            score += 20
        else:
            score += 0  # No points if outside range

        # Humidity suitability (20% weight) - MUST be in range
        if reqs['humidity']['min'] <= humidity <= reqs['humidity']['max']:
            score += 20
        else:
            score += 0  # No points if outside range

        # Rainfall suitability (20% weight) - MUST be in range
        if reqs['rainfall']['min'] <= rainfall <= reqs['rainfall']['max']:
            score += 20
        else:
            score += 0  # No points if outside range

        recommendations.append({
            "crop": crop,
            "suitability_score": score
        })

    # Sort by suitability score (highest first)
    recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)

    return recommendations

def recommend_fertilizer(soil_texture, crop_type, nitrogen, phosphorus, potassium):
    n_level = "Low" if nitrogen < 50 else "Medium" if nitrogen < 100 else "High"
    p_level = "Low" if phosphorus < 25 else "Medium" if phosphorus < 50 else "High"
    k_level = "Low" if potassium < 25 else "Medium" if potassium < 50 else "High"

    try:
        base_fertilizer = list(FERTILIZER_RECOMMENDATIONS[crop_type][soil_texture].keys())[0]
        base_description = FERTILIZER_RECOMMENDATIONS[crop_type][soil_texture][base_fertilizer]
    except KeyError:
        return {
            "fertilizer": "Unknown",
            "reason": "No recommendation found for this combination."
        }

    additional = []
    if n_level == "Low": additional.append("Add urea or ammonium sulfate.")
    if p_level == "Low": additional.append("Add bone meal or rock phosphate.")
    if k_level == "Low": additional.append("Apply potash or wood ash.")

    return {
        "recommended_fertilizer": base_fertilizer,
        "description": base_description,
        "soil_npk_status": f"N: {n_level}, P: {p_level}, K: {k_level}",
        "additional_recommendations": additional
    }

def load_models():
    models = {}
    if os.path.exists(MODEL_PATH):
        try:
            models['texture'] = load_model(MODEL_PATH)
        except:
            models['texture'] = train_texture_model()
    else:
        models['texture'] = train_texture_model()

    if not models['texture']:
        return None, "Texture model error"

    if not os.path.exists(CROP_MODEL_PATH):
        train_crop_model()
    return models, True

@app.route('/predict', methods=['POST'])
def predict():
    """
    Soil Analysis and Crop Recommendation API
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            image:
              type: string
            temperature:
              type: number
            humidity:
              type: number
            rainfall:
              type: number
            nitrogen:
              type: number
            phosphorus:
              type: number
            potassium:
              type: number
            crop_type:
              type: string
              enum: ['rice', 'Irish Potatoes', 'Tomatoes']
              description: Optional - only needed for fertilizer recommendations
    responses:
      200:
        description: Success
    """
    if request.is_json:
        data = request.get_json() or {}
        image_path = data.get('image')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        rainfall = data.get('rainfall')
        crop_type = data.get('crop_type')
        nitrogen = data.get('nitrogen')
        phosphorus = data.get('phosphorus')
        potassium = data.get('potassium')

        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Invalid image path"}), 400
    else:
        if 'image' not in request.files:
            return jsonify({"error": "No image"}), 400

        file = request.files['image']
        try:
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            rainfall = float(request.form.get('rainfall'))
            crop_type = request.form.get('crop_type')
            nitrogen = float(request.form.get('nitrogen'))
            phosphorus = float(request.form.get('phosphorus'))
            potassium = float(request.form.get('potassium'))
        except:
            return jsonify({"error": "Invalid parameters"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
        file.save(image_path)

    try:
        models, status = load_models()
        if not models:
            return jsonify({"error": status}), 500

        soil_texture = predict_texture(image_path, models['texture'])
        crop_recommendations = recommend_crop(soil_texture, temperature, humidity, rainfall)

        if isinstance(crop_recommendations, dict) and "error" in crop_recommendations:
            return jsonify(crop_recommendations), 400

        fertilizer_recommendation = None
        if crop_type:
            fertilizer_recommendation = recommend_fertilizer(soil_texture, crop_type, nitrogen, phosphorus, potassium)
        else:
            fertilizer_recommendation = {"message": "Provide crop_type for fertilizer recommendations"}

        return jsonify({
            "soil_texture": soil_texture,
            "crop_recommendations": crop_recommendations,
            "fertilizer_recommendation": fertilizer_recommendation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    models, status = load_models()
    if models:
        app.run(debug=True)
