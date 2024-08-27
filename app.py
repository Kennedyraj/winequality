from flask import Flask, render_template, request
import pickle
import pandas as pd

# Create a Flask app
app = Flask(__name__)

# Load the trained model and LabelEncoder
with open('wine_quality_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_quality = pickle.load(le_file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    form_data = request.form
    
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'fixed acidity': [float(form_data['fixed_acidity'])],
        'volatile acidity': [float(form_data['volatile_acidity'])],
        'citric acid': [float(form_data['citric_acid'])],
        'residual sugar': [float(form_data['residual_sugar'])],
        'chlorides': [float(form_data['chlorides'])],
        'free sulfur dioxide': [float(form_data['free_sulfur_dioxide'])],
        'total sulfur dioxide': [float(form_data['total_sulfur_dioxide'])],
        'density': [float(form_data['density'])],
        'pH': [float(form_data['ph'])],
        'sulphates': [float(form_data['sulphates'])],
        'alcohol': [float(form_data['alcohol'])]
    })
    
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    
    # Decode the prediction to get the quality label
    prediction_label = label_quality.inverse_transform([prediction_encoded])[0]
    
    # Return result to the template
    return render_template('index.html', prediction_text=f'Predicted Quality: {prediction_label}')

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
