# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    # Extract data from form
    expected_features = model.feature_names_in_
    input_values = [float(request.form[name]) for name in expected_features]

    # Now ensure the input array has the correct shape
    input_array = np.array(input_values).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)
    output = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

    return render_template('index.html', check='The Credit Card Transaction is {}'.format(output))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)