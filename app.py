# from flask import Flask, request, jsonify
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import load_img, img_to_array
# import numpy as np

# # Initialize the Flask app
# app = Flask(__name__)

# # Load the model
# model = load_model('../soil_texture_mobilenetv2.keras')

# # Image size for preprocessing
# IMG_SIZE = (224, 224)

# # Function to preprocess the image
# def preprocess_image(img_path):
#     img = load_img(img_path, target_size=IMG_SIZE)
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Function to predict the texture
# def predict_texture(img_path):
#     img_array = preprocess_image(img_path)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
#     texture_classes = ['sandy', 'loamy', 'clayey', 'alluvial']
#     return texture_classes[predicted_class[0]]

# # Define the root route
# @app.route('/')
# def home():
#     return "Welcome to the AgriSense AI Backend!"

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the JSON payload
#     data = request.json
#     if not data or 'image_path' not in data:
#         return jsonify({'error': 'Missing image_path in JSON payload'}), 400

#     # Get the image path from the JSON payload
#     img_path = data['image_path']

#     # Predict the texture
#     try:
#         predicted_texture = predict_texture(img_path)
#         return jsonify({'predicted_texture': predicted_texture})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# import os
# import model as agrimodel

# # Initialize the Flask app
# app = Flask(__name__)

# # Load models at startup
# models, model_status = agrimodel.load_models()
# if not isinstance(model_status, bool):
#     print(f"Error loading models: {model_status}")
#     print("Starting without models - they will be loaded on the first request")
#     models = None

# # Define the root route
# @app.route('/')
# def home():
#     return "Welcome to the AgriSense AI Backend! API is running."

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     global models
    
#     try:
#         # Check if models are loaded
#         if not models:
#             models, model_status = agrimodel.load_models()
#             if not isinstance(model_status, bool):
#                 return jsonify({'error': f'Failed to load models: {model_status}'}), 500
        
#         # Get JSON data
#         data = request.json
#         if not data:
#             return jsonify({'error': 'Missing JSON payload'}), 400
        
#         # Extract parameters
#         if 'image_path' not in data:
#             return jsonify({'error': 'Missing image_path in JSON payload'}), 400
        
#         img_path = data['image_path']
#         temperature = float(data.get('temperature', 25.0))
#         humidity = float(data.get('humidity', 60.0))
#         rainfall = float(data.get('rainfall', 800.0))
#         nitrogen = float(data.get('nitrogen', 50.0))
#         phosphorus = float(data.get('phosphorus', 30.0))
#         potassium = float(data.get('potassium', 30.0))
        
#         # Check if image path exists
#         if not os.path.exists(img_path):
#             return jsonify({'error': f'Image path not found: {img_path}'}), 400
        
#         # Predict soil texture
#         predicted_texture = agrimodel.predict_texture(img_path, models['texture'])
        
#         # Recommend crops
#         recommended_crops = agrimodel.recommend_crop(
#             predicted_texture, temperature, humidity, rainfall
#         )
        
#         # Get best crop (highest suitability)
#         best_crop = recommended_crops[0]['crop'] if recommended_crops else None
        
#         # Recommend fertilizer for the best crop
#         if best_crop:
#             fertilizer_recommendation = agrimodel.predict_fertilizer(
#                 predicted_texture, best_crop, nitrogen, phosphorus, potassium, 
#                 temperature, humidity
#             )
#         else:
#             fertilizer_recommendation = {
#                 "recommended_fertilizer": "Unknown",
#                 "description": "No suitable crop found for the given conditions."
#             }
        
#         # Return results
#         return jsonify({
#             'soil_analysis': {
#                 'predicted_texture': predicted_texture
#             },
#             'crop_recommendation': recommended_crops,
#             'fertilizer_recommendation': fertilizer_recommendation
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Health check endpoint
# @app.route('/health', methods=['GET'])
# def health_check():
#     if models:
#         return jsonify({'status': 'healthy', 'models_loaded': True})
#     else:
#         return jsonify({'status': 'unhealthy', 'models_loaded': False}), 500

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8080)


from flask import Flask, request, jsonify
import os
import model as agrimodel

# Initialize the Flask app
app = Flask(__name__)

# Load models at startup without interactive prompts
# This line is modified to use a patched version of load_models or set environment variable
os.environ["REBUILD_MODEL"] = "n"  # Set default to not rebuild model
models, model_status = agrimodel.load_models()
if not isinstance(model_status, bool):
    print(f"Error loading models: {model_status}")
    print("Starting without models - they will be loaded on the first request")
    models = None

# Define the root route
@app.route('/')
def home():
    return "Welcome to the AgriSense AI Backend! API is running."

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    global models
    
    try:
        # Check if models are loaded
        if not models:
            models, model_status = agrimodel.load_models()
            if not isinstance(model_status, bool):
                return jsonify({'error': f'Failed to load models: {model_status}'}), 500
        
        # Get JSON data
        data = request.json
        if not data:
            return jsonify({'error': 'Missing JSON payload'}), 400
        
        # Extract parameters
        if 'image_path' not in data:
            return jsonify({'error': 'Missing image_path in JSON payload'}), 400
        
        img_path = data['image_path']
        temperature = float(data.get('temperature', 25.0))
        humidity = float(data.get('humidity', 60.0))
        rainfall = float(data.get('rainfall', 800.0))
        nitrogen = float(data.get('nitrogen', 50.0))
        phosphorus = float(data.get('phosphorus', 30.0))
        potassium = float(data.get('potassium', 30.0))
        
        # Check if image path exists
        if not os.path.exists(img_path):
            return jsonify({'error': f'Image path not found: {img_path}'}), 400
        
        # Predict soil texture
        predicted_texture = agrimodel.predict_texture(img_path, models['texture'])
        
        # Recommend crops
        recommended_crops = agrimodel.recommend_crop(
            predicted_texture, temperature, humidity, rainfall
        )
        
        # Get best crop (highest suitability)
        best_crop = recommended_crops[0]['crop'] if recommended_crops else None
        
        # Recommend fertilizer for the best crop
        if best_crop:
            fertilizer_recommendation = agrimodel.predict_fertilizer(
                predicted_texture, best_crop, nitrogen, phosphorus, potassium, 
                temperature, humidity
            )
        else:
            fertilizer_recommendation = {
                "recommended_fertilizer": "Unknown",
                "description": "No suitable crop found for the given conditions."
            }
        
        # Return results
        return jsonify({
            'soil_analysis': {
                'predicted_texture': predicted_texture
            },
            'crop_recommendation': recommended_crops,
            'fertilizer_recommendation': fertilizer_recommendation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    if models:
        return jsonify({'status': 'healthy', 'models_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'models_loaded': False}), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)