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
import requests
from datetime import datetime, timedelta

from data.crop_data import CROP_SUITABILITY
from data.fertilizer_data import FERTILIZER_RECOMMENDATIONS
from data.disease_data import DISEASE_PATTERNS, CROP_DISEASES
from data.irrigation_data import CROP_WATER_NEEDS, SOIL_MOISTURE_THRESHOLDS

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

def detect_disease(image_path, crop_type):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Disease detection feature - WAITING FOR SATELLITE DATA
        # Currently no dataset available - using placeholder logic
        # Once satellite integration is active, real CNN predictions will work

        return {
            'status': 'satellite_integration_pending',
            'message': 'Disease detection waiting for satellite data integration',
            'current_capability': 'placeholder_only',
            'available_diseases': CROP_DISEASES.get(crop_type, []),
            'satellite_status': {
                'data_source': 'pending_satellite_imagery',
                'prediction_accuracy': 'will_improve_with_real_data',
                'eta': 'Available once satellite API is integrated'
            },
            'placeholder_info': {
                'note': 'Currently showing disease patterns database',
                'real_predictions': 'Requires satellite imagery dataset'
            }
        }
    except Exception as e:
        return {'error': f'Disease detection failed: {str(e)}'}

def recommend_irrigation(soil_moisture, crop_type, temperature, humidity, rainfall):
    try:
        if crop_type not in CROP_WATER_NEEDS:
            return {'error': f'Crop type {crop_type} not supported for irrigation recommendations'}

        crop_needs = CROP_WATER_NEEDS[crop_type]
        optimal = crop_needs['optimal_moisture']
        critical = crop_needs['critical_moisture']

        # Calculate irrigation need
        if soil_moisture <= critical:
            urgency = 'immediate'
            next_irrigation = 'Today, 6 AM'
            water_amount = crop_needs['daily_water_mm'] * 1.5
        elif soil_moisture < optimal:
            urgency = 'soon'
            next_irrigation = 'Tomorrow, 6 AM'
            water_amount = crop_needs['daily_water_mm']
        else:
            urgency = 'none'
            next_irrigation = 'No immediate irrigation needed'
            water_amount = 0

        # Adjust for weather conditions
        if rainfall > 10:
            water_amount *= 0.7  # Reduce if recent rainfall

        if temperature > 35:
            water_amount *= 1.2  # Increase in extreme heat

        return {
            'status': urgency,
            'next_irrigation': next_irrigation,
            'recommended_water_mm': round(water_amount, 1),
            'soil_moisture': soil_moisture,
            'optimal_moisture': optimal,
            'weather_adjustment': f"Reduced by 30% due to rainfall" if rainfall > 10 else "Increased by 20% due to heat" if temperature > 35 else "No adjustment"
        }
    except Exception as e:
        return {'error': f'Irrigation recommendation failed: {str(e)}'}

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

@app.route('/comprehensive-analyze', methods=['POST'])
def comprehensive_analyze():
    """
    Comprehensive Agricultural Analysis API
    ---
    tags:
      - Analysis
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Photo of the soil (jpg/png). Used to classify soil texture.
      - name: temperature
        in: formData
        type: number
        required: true
        description: Air temperature in degrees Celsius
      - name: humidity
        in: formData
        type: number
        required: true
        description: Relative humidity in percent
      - name: rainfall
        in: formData
        type: number
        required: true
        description: Rainfall in mm
      - name: nitrogen
        in: formData
        type: number
        required: true
        description: Soil nitrogen level (kg/ha)
      - name: phosphorus
        in: formData
        type: number
        required: true
        description: Soil phosphorus level (kg/ha)
      - name: potassium
        in: formData
        type: number
        required: true
        description: Soil potassium level (kg/ha)
      - name: soil_moisture
        in: formData
        type: number
        required: false
        default: 50
        description: Soil moisture in percent (defaults to 50)
      - name: crop_type
        in: formData
        type: string
        required: false
        enum: ['rice', 'Irish Potatoes', 'Tomatoes']
        description: Required for fertilizer recommendations, otherwise optional
    responses:
      200:
        description: Comprehensive analysis results
        schema:
          type: object
          properties:
            soil_analysis:
              type: object
              properties:
                texture:
                  type: string
                  enum: ['sandy', 'loamy', 'clayey', 'alluvial']
                moisture:
                  type: number
            crop_recommendations:
              type: array
              items:
                type: object
                properties:
                  crop:
                    type: string
                  suitability_score:
                    type: number
            disease_analysis:
              type: object
            irrigation_recommendation:
              type: object
            fertilizer_recommendation:
              type: object
            weather_forecast:
              type: object
            satellite_integration:
              type: object
            timestamp:
              type: string
        examples:
          application/json:
            soil_analysis: {texture: "loamy", moisture: 45}
            crop_recommendations:
              - {crop: "Irish Potatoes", suitability_score: 100.0}
              - {crop: "Tomatoes", suitability_score: 80.0}
              - {crop: "rice", suitability_score: 40.0}
            disease_analysis:
              status: "satellite_integration_pending"
              message: "Disease detection waiting for satellite data integration"
            irrigation_recommendation:
              status: "soon"
              next_irrigation: "Tomorrow, 6 AM"
              recommended_water_mm: 5.0
              soil_moisture: 45
              optimal_moisture: 60
              weather_adjustment: "No adjustment"
            fertilizer_recommendation:
              recommended_fertilizer: "NPK 17-17-17"
              description: "Balanced fertilizer for loamy soil"
              soil_npk_status: "N: Low, P: Medium, K: High"
              additional_recommendations: ["Add urea or ammonium sulfate."]
            weather_forecast:
              today: {temp: 22.0, humidity: 70.0, rainfall: 5.0}
              tomorrow: {temp: 24.0, humidity: 65.0, rainfall: 0}
              next_3_days: "Partly cloudy with chance of rain"
            satellite_integration:
              disease_detection: "pending_satellite_data"
              irrigation_monitoring: "ready_with_ground_sensors"
            timestamp: "2026-07-31T18:00:00.000000"
      400:
        description: Missing/invalid parameters or unsupported soil texture
      500:
        description: Model loading or prediction error
    """
    if request.is_json:
        data = request.get_json() or {}
        image_path = data.get('image')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        rainfall = data.get('rainfall')
        nitrogen = data.get('nitrogen')
        phosphorus = data.get('phosphorus')
        potassium = data.get('potassium')
        soil_moisture = data.get('soil_moisture', 50)  # Default if not provided
        crop_type = data.get('crop_type')
    else:
        if 'image' not in request.files:
            return jsonify({"error": "No image"}), 400

        file = request.files['image']
        try:
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            rainfall = float(request.form.get('rainfall'))
            nitrogen = float(request.form.get('nitrogen'))
            phosphorus = float(request.form.get('phosphorus'))
            potassium = float(request.form.get('potassium'))
            soil_moisture = float(request.form.get('soil_moisture', 50))
            crop_type = request.form.get('crop_type')
        except:
            return jsonify({"error": "Invalid parameters"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
        file.save(image_path)

    try:
        models, status = load_models()
        if not models:
            return jsonify({"error": status}), 500

        # Soil texture analysis
        soil_texture = predict_texture(image_path, models['texture'])

        # Crop recommendations
        crop_recommendations = recommend_crop(soil_texture, temperature, humidity, rainfall)
        if isinstance(crop_recommendations, dict) and "error" in crop_recommendations:
            return jsonify(crop_recommendations), 400

        # Disease detection
        disease_analysis = detect_disease(image_path, crop_recommendations[0]['crop'] if crop_recommendations else 'unknown')

        # Irrigation recommendations
        irrigation_recommendation = recommend_irrigation(soil_moisture, crop_recommendations[0]['crop'] if crop_recommendations else 'unknown', temperature, humidity, rainfall)

        # Fertilizer recommendations
        fertilizer_recommendation = None
        if crop_type:
            fertilizer_recommendation = recommend_fertilizer(soil_texture, crop_type, nitrogen, phosphorus, potassium)
        else:
            fertilizer_recommendation = {"message": "Provide crop_type for fertilizer recommendations"}

        # Weather forecast (placeholder - would integrate with real API)
        weather_forecast = {
            "today": {"temp": temperature, "humidity": humidity, "rainfall": rainfall},
            "tomorrow": {"temp": temperature + 2, "humidity": humidity - 5, "rainfall": max(0, rainfall - 5)},
            "next_3_days": "Partly cloudy with chance of rain"
        }

        # Satellite integration status
        satellite_status = {
            "disease_detection": "pending_satellite_data",
            "irrigation_monitoring": "ready_with_ground_sensors",
            "crop_health_monitoring": "pending_ndvi_integration",
            "data_sources": {
                "current": ["ground_sensors", "user_input", "manual_images"],
                "planned": ["sentinel_2", "landsat_8", "modis"],
                "eta": "once_apis_integrated"
            }
        }

        return jsonify({
            "soil_analysis": {
                "texture": soil_texture,
                "moisture": soil_moisture
            },
            "crop_recommendations": crop_recommendations,
            "disease_analysis": disease_analysis,
            "irrigation_recommendation": irrigation_recommendation,
            "fertilizer_recommendation": fertilizer_recommendation,
            "weather_forecast": weather_forecast,
            "satellite_integration": satellite_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Soil Analysis and Crop Recommendation API with Satellite Integration
    ---
    tags:
      - Prediction
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Photo of the soil (jpg/png). Used to classify soil texture.
      - name: temperature
        in: formData
        type: number
        required: true
        description: Air temperature in degrees Celsius
      - name: humidity
        in: formData
        type: number
        required: true
        description: Relative humidity in percent
      - name: rainfall
        in: formData
        type: number
        required: true
        description: Rainfall in mm
      - name: nitrogen
        in: formData
        type: number
        required: true
        description: Soil nitrogen level (kg/ha)
      - name: phosphorus
        in: formData
        type: number
        required: true
        description: Soil phosphorus level (kg/ha)
      - name: potassium
        in: formData
        type: number
        required: true
        description: Soil potassium level (kg/ha)
      - name: soil_moisture
        in: formData
        type: number
        required: false
        default: 50
        description: Soil moisture in percent (defaults to 50)
      - name: crop_type
        in: formData
        type: string
        required: false
        enum: ['rice', 'Irish Potatoes', 'Tomatoes']
        description: Optional - AI will auto-detect best crop if not provided
      - name: lat
        in: formData
        type: number
        required: false
        description: Optional field latitude for satellite data
      - name: lon
        in: formData
        type: number
        required: false
        description: Optional field longitude for satellite data
    responses:
      200:
        description: Prediction results with satellite integration
        schema:
          type: object
          properties:
            recommendations:
              type: array
              description: One block per category (crop, irrigation, disease, fertilizer, weather)
              items:
                type: object
                properties:
                  category:
                    type: string
                  icon:
                    type: string
                  data:
                    type: object
            soil_analysis:
              type: object
              properties:
                texture:
                  type: string
                  enum: ['sandy', 'loamy', 'clayey', 'alluvial']
                moisture:
                  type: number
            satellite_integration:
              type: object
            timestamp:
              type: string
        examples:
          application/json:
            recommendations:
              - category: "Crop recommends"
                icon: "plant"
                data:
                  - {crop: "Irish Potatoes", suitability_score: 100.0}
                  - {crop: "rice", suitability_score: 40.0}
                best_crop: "Irish Potatoes"
                confidence: 100.0
              - category: "Irrigation recommends"
                icon: "water_drop"
                data:
                  status: "soon"
                  next_irrigation: "Tomorrow, 6 AM"
                  recommended_water_mm: 5.0
                  soil_moisture: 45
                  optimal_moisture: 60
                  weather_adjustment: "No adjustment"
                status: "soon"
                next_irrigation: "Tomorrow, 6 AM"
              - category: "Disease recommends"
                icon: "person_with_magnifying_glass"
                data:
                  status: "satellite_integration_pending"
                  message: "Disease detection waiting for satellite data integration"
              - category: "Fertilizer recommends"
                icon: "fertilizer_bag"
                data:
                  recommended_fertilizer: "NPK 17-17-17"
                  description: "Balanced fertilizer for loamy soil"
                  soil_npk_status: "N: Low, P: Medium, K: High"
                  additional_recommendations: ["Add urea or ammonium sulfate."]
                recommended_fertilizer: "NPK 17-17-17"
                npk_status: "N: Low, P: Medium, K: High"
              - category: "Weather recommends"
                icon: "cloud_with_rain"
                data:
                  today: {temp: 22.0, humidity: 70.0, rainfall: 5.0}
                  tomorrow: {temp: 24.0, humidity: 65.0, rainfall: 0}
                  next_3_days: "Partly cloudy with chance of rain"
            soil_analysis: {texture: "loamy", moisture: 45}
            satellite_integration: null
            timestamp: "2026-07-31T18:00:00.000000"
      400:
        description: Missing/invalid parameters or unsupported soil texture
      500:
        description: Model loading or prediction error
    """
    if request.is_json:
        data = request.get_json() or {}
        image_path = data.get('image')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        rainfall = data.get('rainfall')
        nitrogen = data.get('nitrogen')
        phosphorus = data.get('phosphorus')
        potassium = data.get('potassium')
        soil_moisture = data.get('soil_moisture', 50)
        crop_type = data.get('crop_type')
        coordinates = data.get('coordinates')

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
            nitrogen = float(request.form.get('nitrogen'))
            phosphorus = float(request.form.get('phosphorus'))
            potassium = float(request.form.get('potassium'))
            soil_moisture = float(request.form.get('soil_moisture', 50))
            crop_type = request.form.get('crop_type')
            # Get coordinates from form if available
            lat = request.form.get('lat')
            lon = request.form.get('lon')
            coordinates = {'lat': float(lat), 'lon': float(lon)} if lat and lon else None
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

        best_crop = crop_recommendations[0]['crop'] if crop_recommendations else 'unknown'

        disease_analysis = detect_disease(image_path, best_crop)

        # Irrigation recommendations (use best crop)
        irrigation_recommendation = recommend_irrigation(soil_moisture, best_crop, temperature, humidity, rainfall)

        # Fertilizer recommendations (auto-use best crop)
        fertilizer_recommendation = recommend_fertilizer(soil_texture, best_crop, nitrogen, phosphorus, potassium)

        # Satellite data integration
        satellite_data = None
        if coordinates:
            satellite_data = get_satellite_data(coordinates)

        return jsonify({
            "recommendations": [
                {
                    "category": "Crop recommends",
                    "icon": "plant",
                    "data": crop_recommendations,
                    "best_crop": crop_recommendations[0]['crop'] if crop_recommendations else "unknown",
                    "confidence": crop_recommendations[0]['suitability_score'] if crop_recommendations else 0
                },
                {
                    "category": "Irrigation recommends",
                    "icon": "water_drop",
                    "data": irrigation_recommendation,
                    "status": irrigation_recommendation.get('status', 'unknown'),
                    "next_irrigation": irrigation_recommendation.get('next_irrigation', 'Not available')
                },
                {
                    "category": "Disease recommends",
                    "icon": "person_with_magnifying_glass",
                    "data": disease_analysis,
                    "health_status": disease_analysis.get('health_status', 'unknown'),
                    "detected_diseases": disease_analysis.get('detected_diseases', {})
                },
                {
                    "category": "Fertilizer recommends",
                    "icon": "fertilizer_bag",
                    "data": fertilizer_recommendation,
                    "recommended_fertilizer": fertilizer_recommendation.get('recommended_fertilizer', 'Unknown') if isinstance(fertilizer_recommendation, dict) else 'Unknown',
                    "npk_status": fertilizer_recommendation.get('soil_npk_status', 'N: Unknown') if isinstance(fertilizer_recommendation, dict) else 'N: Unknown'
                },
                {
                    "category": "Weather recommends",
                    "icon": "cloud_with_rain",
                    "data": {
                        "today": {"temp": temperature, "humidity": humidity, "rainfall": rainfall},
                        "tomorrow": {"temp": temperature + 2, "humidity": humidity - 5, "rainfall": max(0, rainfall - 5)},
                        "next_3_days": "Partly cloudy with chance of rain"
                    }
                }
            ],
            "soil_analysis": {
                "texture": soil_texture,
                "moisture": soil_moisture
            },
            "satellite_integration": satellite_data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_satellite_data(coordinates):
    """Get satellite data for field coordinates"""
    try:
        # Placeholder for satellite API integration
        # In production, this would call Sentinel-2, Landsat-8 APIs

        return {
            "status": "integration_ready",
            "coordinates": coordinates,
            "data_sources": ["sentinel_2", "landsat_8", "modis"],
            "current_capability": "placeholder",
            "available_data": {
                "ndvi": "pending_api_integration",
                "field_health": "pending_ndvi_analysis",
                "historical_imagery": "available_once_connected"
            },
            "api_status": {
                "sentinel_2": "free_registration_required",
                "landsat_8": "free_account_needed",
                "modis": "direct_access_available"
            },
            "next_steps": "Configure API keys for real satellite imagery"
        }
    except Exception as e:
        return {"error": f"Satellite data failed: {str(e)}"}

if __name__ == "__main__":
    models, status = load_models()
    if models:
        app.run(debug=True)
