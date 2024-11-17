from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))
# Glass types corresponding to class numbers
glass_types = {
    1: 'Building Windows Float Processed',
    2: 'Building Windows Non-Float Processed',
    3: 'Vehicle Windows Float Processed',
    4: 'Vehicle Windows Non-Float Processed',
    5: 'Containers',
    6: 'Tableware',
    7: 'Headlamps'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [
            float(request.form['RI']),
            float(request.form['Na']),
            float(request.form['Mg']),
            float(request.form['Al']),
            float(request.form['Si']),
            float(request.form['K']),
            float(request.form['Ca']),
            float(request.form['Ba']),
            float(request.form['Fe'])
        ]

        # Convert the features into a numpy array and reshape for prediction
        features = np.array(features).reshape(1, -1)

        # Predict the class number using the model
        prediction_number = model.predict(features)[0]
        
        # Get the name of the glass type
        prediction_name = glass_types.get(prediction_number, 'Unknown')

        return render_template('result.html', prediction_name=prediction_name, prediction_number=prediction_number)

    except Exception as e:
        return render_template('index.html', error="Error in prediction. Please check your input.")

if __name__ == '__main__':
    app.run(debug=True)
