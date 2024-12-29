from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = './model/house_price_model.pkl'
model = joblib.load(model_path)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        # Extract features from form inputs
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        stories = int(data['stories'])
        parking = int(data['parking'])

        # Add placeholders for missing features
        # Update the additional features to match the model's training
        additional_features = [0] * 8  # Adjust values based on training data
        
        # Combine the features into a single array
        features = [area, bedrooms, bathrooms, stories, parking] + additional_features

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)

        # Return the prediction
        return jsonify({
            'prediction': f'The predicted house price is: ${prediction[0]:,.2f}'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
