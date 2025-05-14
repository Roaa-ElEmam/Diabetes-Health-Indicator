from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import traceback
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model (you'll need to train and save it first)
try:
    model = joblib.load('diabetes_model.pkl')
    with open('feature_order.pkl', 'rb') as f:
        feature_order = pickle.load(f)
    logger.info("Model and feature order loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or feature order: {str(e)}")
    logger.error(traceback.format_exc())
    model = None
    feature_order = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            logger.error("Model is None - not loaded properly")
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please ensure diabetes_model.pkl exists.'
            })

        # Log the incoming form data
        logger.debug(f"Received form data: {request.form}")
        
        # Get values from the form
        data = {
            'HighBP': float(request.form['HighBP']),
            'HighChol': float(request.form['HighChol']),
            'BMI': float(request.form['BMI']),
            'Smoker': float(request.form['Smoker']),
            'Stroke': float(request.form['Stroke']),
            'HeartDiseaseorAttack': float(request.form['HeartDiseaseorAttack']),
            'PhysActivity': float(request.form['PhysActivity']),
            'Fruits': float(request.form['Fruits']),
            'Veggies': float(request.form['Veggies']),
            'HvyAlcoholConsump': float(request.form['HvyAlcoholConsump']),
            'AnyHealthcare': float(request.form['AnyHealthcare']),
            'CholCheck': float(request.form['CholCheck']),
            'NoDocbcCost': float(request.form['NoDocbcCost']),
            'GenHlth': float(request.form['GenHlth']),
            'MentHlth': float(request.form['MentHlth']),
            'PhysHlth': float(request.form['PhysHlth']),
            'DiffWalk': float(request.form['DiffWalk']),
            'Sex': float(request.form['Sex']),
            'Age': float(request.form['Age']),
            'Education': float(request.form['Education']),
            'Income': float(request.form['Income'])
        }
        
        logger.debug(f"Processed data: {data}")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        logger.debug(f"DataFrame shape before reorder: {df.shape}")
        # Reorder columns to match training
        df = df[feature_order]
        logger.debug(f"DataFrame shape after reorder: {df.shape}")
        
        # Make prediction
        prediction = model.predict(df)[0]
        logger.debug(f"Raw prediction: {prediction}")
        
        # Map prediction to status
        status_map = {
            0: "Healthy",
            1: "Pre-diabetic",
            2: "Diabetic"
        }
        
        result = status_map.get(prediction, "Unknown")
        logger.info(f"Final prediction: {result}")
        
        return jsonify({
            'status': 'success',
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error during prediction: {str(e)}'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 