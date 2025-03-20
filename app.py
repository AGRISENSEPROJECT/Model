# # from flask import Flask, request, jsonify
# # import os
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.utils import load_img, img_to_array
# # import numpy as np

# # # Initialize the Flask app
# # app = Flask(__name__)

# # # Load the model
# # model = load_model('../soil_texture_mobilenetv2.keras')

# # # Image size for preprocessing
# # IMG_SIZE = (224, 224)

# # # Function to preprocess the image
# # def preprocess_image(img_path):
# #     img = load_img(img_path, target_size=IMG_SIZE)
# #     img_array = img_to_array(img) / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)
# #     return img_array

# # # Function to predict the texture
# # def predict_texture(img_path):
# #     img_array = preprocess_image(img_path)
# #     prediction = model.predict(img_array)
# #     predicted_class = np.argmax(prediction, axis=1)
# #     texture_classes = ['sandy', 'loamy', 'clayey', 'alluvial']
# #     return texture_classes[predicted_class[0]]

# # # Define the root route
# # @app.route('/')
# # def home():
# #     return "Welcome to the AgriSense AI Backend!"

# # # Define the prediction route
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get the JSON payload
# #     data = request.json
# #     if not data or 'image_path' not in data:
# #         return jsonify({'error': 'Missing image_path in JSON payload'}), 400

# #     # Get the image path from the JSON payload
# #     img_path = data['image_path']

# #     # Predict the texture
# #     try:
# #         predicted_texture = predict_texture(img_path)
# #         return jsonify({'predicted_texture': predicted_texture})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # # Run the app
# # if __name__ == '__main__':
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify
# # import os
# # import model as agrimodel

# # # Initialize the Flask app
# # app = Flask(__name__)

# # # Load models at startup
# # models, model_status = agrimodel.load_models()
# # if not isinstance(model_status, bool):
# #     print(f"Error loading models: {model_status}")
# #     print("Starting without models - they will be loaded on the first request")
# #     models = None

# # # Define the root route
# # @app.route('/')
# # def home():
# #     return "Welcome to the AgriSense AI Backend! API is running."

# # # Define the prediction route
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     global models
    
# #     try:
# #         # Check if models are loaded
# #         if not models:
# #             models, model_status = agrimodel.load_models()
# #             if not isinstance(model_status, bool):
# #                 return jsonify({'error': f'Failed to load models: {model_status}'}), 500
        
# #         # Get JSON data
# #         data = request.json
# #         if not data:
# #             return jsonify({'error': 'Missing JSON payload'}), 400
        
# #         # Extract parameters
# #         if 'image_path' not in data:
# #             return jsonify({'error': 'Missing image_path in JSON payload'}), 400
        
# #         img_path = data['image_path']
# #         temperature = float(data.get('temperature', 25.0))
# #         humidity = float(data.get('humidity', 60.0))
# #         rainfall = float(data.get('rainfall', 800.0))
# #         nitrogen = float(data.get('nitrogen', 50.0))
# #         phosphorus = float(data.get('phosphorus', 30.0))
# #         potassium = float(data.get('potassium', 30.0))
        
# #         # Check if image path exists
# #         if not os.path.exists(img_path):
# #             return jsonify({'error': f'Image path not found: {img_path}'}), 400
        
# #         # Predict soil texture
# #         predicted_texture = agrimodel.predict_texture(img_path, models['texture'])
        
# #         # Recommend crops
# #         recommended_crops = agrimodel.recommend_crop(
# #             predicted_texture, temperature, humidity, rainfall
# #         )
        
# #         # Get best crop (highest suitability)
# #         best_crop = recommended_crops[0]['crop'] if recommended_crops else None
        
# #         # Recommend fertilizer for the best crop
# #         if best_crop:
# #             fertilizer_recommendation = agrimodel.predict_fertilizer(
# #                 predicted_texture, best_crop, nitrogen, phosphorus, potassium, 
# #                 temperature, humidity
# #             )
# #         else:
# #             fertilizer_recommendation = {
# #                 "recommended_fertilizer": "Unknown",
# #                 "description": "No suitable crop found for the given conditions."
# #             }
        
# #         # Return results
# #         return jsonify({
# #             'soil_analysis': {
# #                 'predicted_texture': predicted_texture
# #             },
# #             'crop_recommendation': recommended_crops,
# #             'fertilizer_recommendation': fertilizer_recommendation
# #         })
        
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # # Health check endpoint
# # @app.route('/health', methods=['GET'])
# # def health_check():
# #     if models:
# #         return jsonify({'status': 'healthy', 'models_loaded': True})
# #     else:
# #         return jsonify({'status': 'unhealthy', 'models_loaded': False}), 500

# # # Run the app
# # if __name__ == '__main__':
# #     app.run(debug=True, host='0.0.0.0', port=8080)


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

# import os
# import tempfile
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import logging

# # Import functions from your model.py file
# from model import predict_texture, recommend_crop, predict_fertilizer

# # Configure logging
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the model
# MODEL_PATH = 'soil_texture_mobilenetv2.keras'

# # Load the model at startup
# try:
#     texture_model = load_model(MODEL_PATH)
#     logger.info(f"Model loaded successfully from {MODEL_PATH}")
# except Exception as e:
#     logger.error(f"Error loading model: {str(e)}")
#     texture_model = None

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint to verify the API is running."""
#     return jsonify({"status": "healthy", "model_loaded": texture_model is not None})

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Endpoint to predict soil texture, recommend crops, and fertilizers.
#     Accepts image file upload and additional parameters.
#     """
#     # Check if image is in the request
#     if 'image' not in request.files:
#         logger.warning("No image part in the request")
#         return jsonify({"error": "No image part in the request"}), 400
        
#     img_file = request.files['image']
#     if img_file.filename == '':
#         logger.warning("No image selected")
#         return jsonify({"error": "No image selected"}), 400
    
#     # Get parameters from the request with default values
#     try:
#         temperature = float(request.form.get('temperature', 25))
#         humidity = float(request.form.get('humidity', 60))
#         rainfall = float(request.form.get('rainfall', 500))
#         nitrogen = float(request.form.get('nitrogen', 50))
#         phosphorus = float(request.form.get('phosphorus', 25))
#         potassium = float(request.form.get('potassium', 25))
#         soil_type = request.form.get('soil_type', 'unknown')  # Optional parameter
#     except ValueError as e:
#         logger.error(f"Invalid parameter format: {str(e)}")
#         return jsonify({"error": f"Invalid parameter format: {str(e)}"}), 400
    
#     # Ensure the model is loaded
#     if texture_model is None:
#         logger.error("Model not loaded")
#         return jsonify({"error": "Model not loaded. Please check server logs."}), 500
    
#     # Create a temporary file to save the uploaded image
#     temp_dir = tempfile.gettempdir()
    
#     try:
#         # Use a temporary file with context manager
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=temp_dir) as temp:
#             temp_path = temp.name
#             img_file.save(temp_path)
#             logger.info(f"Image saved to temporary path: {temp_path}")
            
#             # Predict soil texture
#             try:
#                 predicted_texture = predict_texture(temp_path, texture_model)
#                 logger.info(f"Predicted soil texture: {predicted_texture}")
#             except Exception as e:
#                 logger.error(f"Error predicting texture: {str(e)}")
#                 return jsonify({"error": f"Error predicting texture: {str(e)}"}), 500
            
#             # If soil_type is provided, use it to override the predicted texture
#             if soil_type != 'unknown':
#                 logger.info(f"Using provided soil type: {soil_type} instead of predicted: {predicted_texture}")
#                 soil_texture_for_recommendations = soil_type.lower()
#             else:
#                 soil_texture_for_recommendations = predicted_texture
            
#             # Recommend crops
#             recommended_crops = recommend_crop(soil_texture_for_recommendations, temperature, humidity, rainfall)
            
#             # Recommend fertilizer for the first crop
#             fertilizer_recommendation = None
#             if recommended_crops:
#                 top_crop = recommended_crops[0]['crop']
#                 fertilizer_recommendation = predict_fertilizer(
#                     soil_texture_for_recommendations, 
#                     top_crop, 
#                     nitrogen, 
#                     phosphorus, 
#                     potassium, 
#                     temperature, 
#                     humidity
#                 )
            
#             # Prepare the response
#             response = {
#                 "predicted_texture": predicted_texture,
#                 "recommended_crops": recommended_crops,
#                 "fertilizer_recommendation": fertilizer_recommendation
#             }
            
#             return jsonify(response)
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
#     finally:
#         # Clean up - delete the temporary file
#         if 'temp_path' in locals() and os.path.exists(temp_path):
#             try:
#                 os.unlink(temp_path)
#                 logger.info(f"Temporary file deleted: {temp_path}")
#             except Exception as e:
#                 logger.warning(f"Failed to delete temporary file: {str(e)}")

# if __name__ == '__main__':
#     # Get port from environment variable or default to 5000
#     port = int(os.environ.get('PORT', 5000))
    
#     # In production, you might want to use a production WSGI server
#     # This is just for development/testing
#     app.run(host='0.0.0.0', port=port, debug=False)