
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model and preprocessing objects
MODEL_DIR = 'models'
model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for model and feature names
model = None
feature_names = []

def load_model():
    global model, feature_names
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        logger.info("Model and feature names loaded successfully!")
        logger.info(f"Using model: {type(model).__name__}")
        logger.info(f"Features: {feature_names}")
        return True
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Load the model at startup
load_model()

@app.route('/')
def home():
    if model is None:
        model_status = "Model not loaded. Please ensure model files exist."
        return render_template('index.html', feature_names=feature_names, 
                              error_message=model_status)
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                              prediction_text="Error: Model not loaded", 
                              feature_names=feature_names)
    
    try:
        # Get data from form
        features = {}
        for feature in feature_names:
            # Get the value from the form, with better error handling
            value = request.form.get(feature, '')
            if value.strip() == '':
                features[feature] = 0.0
            else:
                features[feature] = float(value)

        # Create a dataframe with the features
        input_df = pd.DataFrame([features])
        
        logger.info(f"Prediction input: {features}")

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Format prediction for display (convert from $100,000s to full dollar amount)
        output = prediction * 100000
        
        # Create a timestamp for this prediction
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Prediction result: ${output:,.2f}")

        return render_template('index.html',
                               prediction_text=f'Estimated House Price: ${output:,.2f}',
                               prediction_value=f'${output:,.2f}',
                               timestamp=timestamp,
                               feature_names=feature_names,
                               input_values=features)

    except ValueError as e:
        logger.error(f"Invalid input value: {e}")
        return render_template('index.html',
                               prediction_text='Error: Please enter valid numeric values for all fields',
                               feature_names=feature_names)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('index.html',
                               prediction_text=f'Error in prediction: {str(e)}',
                               feature_names=feature_names)

@app.route('/clear', methods=['POST'])
def clear():
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port)
