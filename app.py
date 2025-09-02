from flask import Flask, request, render_template_string
import pandas as pd
from Model import train_and_get_components

app = Flask(__name__)

HTML_FORM_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Disease Diagnosis Predictor</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        form { background: #f4f4f4; padding: 1em; border-radius: 5px; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
        label { font-weight: bold; }
        input, select { width: 100%; padding: 8px; box-sizing: border-box; }
        .submit-btn { grid-column: span 2; padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Preliminary Diagnosis Assistant</h1>
    <p>Enter patient data below to get a predicted diagnosis.</p>
    <form action="/predict" method="post">
        <div class="form-grid">
            <div>
                <label for="age">Age:</label>
                <input type="number" id="age" name="Age" required>
            </div>
            <div>
                <label for="gender">Gender:</label>
                <select name="Gender" id="gender">
                    {% for item in genders %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                </select>
            </div>
            <div>
                <label for="symptom1">Symptom 1:</label>
                <select name="Symptom_1" id="symptom1">
                    {% for item in symptoms %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                </select>
            </div>
            <div>
                <label for="symptom2">Symptom 2:</label>
                <select name="Symptom_2" id="symptom2">
                     {% for item in symptoms %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                </select>
            </div>
            <div>
                <label for="symptom3">Symptom 3:</label>
                <select name="Symptom_3" id="symptom3">
                     {% for item in symptoms %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                </select>
            </div>
            <div>
                <label for="heart_rate">Heart Rate (bpm):</label>
                <input type="number" id="heart_rate" name="Heart_Rate_bpm" required>
            </div>
             <div>
                <label for="temp">Body Temp (°C):</label>
                <input type="text" id="temp" name="Body_Temperature_C" required>
            </div>
            <div>
                <label for="oxygen">Oxygen Saturation (%):</label>
                <input type="number" id="oxygen" name="Oxygen_Saturation_%" required>
            </div>
            <div>
                <label for="systolic">Systolic BP (top #):</label>
                <input type="number" id="systolic" name="Systolic_BP" required>
            </div>
            <div>
                <label for="diastolic">Diastolic BP (bottom #):</label>
                <input type="number" id="diastolic" name="Diastolic_BP" required>
            </div>
        </div>
        <br>
        <input class="submit-btn" type="submit" value="Predict Diagnosis">
    </form>
</body>
</html>
"""

HTML_RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
     <style>
        body { font-family: sans-serif; margin: 2em; text-align: center; }
        .result-box { background: #e7f3fe; border-left: 6px solid #2196F3; padding: 20px; margin: 20px auto; max-width: 500px; }
        h2 { color: #2196F3; }
    </style>
</head>
<body>
    <div class="result-box">
        <h1>Predicted Diagnosis</h1>
        <h2>{{ prediction }}</h2>
        <p><em>This is a prediction based on a machine learning model and is not a substitute for professional medical advice.</em></p>
    </div>
    <a href="/">Make another prediction</a>
</body>
</html>
"""

print("Loading data and training model...")
MODEL, FEATURE_COLUMNS, LE_DIAGNOSIS, SYMPTOMS, GENDERS = train_and_get_components('disease_diagnosis.csv')
print("Model trained and ready.")


@app.route('/')
def home():
    return render_template_string(HTML_FORM_TEMPLATE, symptoms=SYMPTOMS, genders=GENDERS)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    input_df = pd.DataFrame(columns=FEATURE_COLUMNS)
    input_df.loc[0] = 0

    for key, value in form_data.items():
        if key in ['Age', 'Heart_Rate_bpm', 'Systolic_BP', 'Diastolic_BP', 'Oxygen_Saturation_%']:
             input_df[key] = int(value)
        elif key == 'Body_Temperature_C':
             input_df[key] = float(value)
        else:
            column_name = f"{key}_{value}"
            if column_name in input_df.columns:
                input_df[column_name] = 1

    prediction_encoded = MODEL.predict(input_df)[0]
    prediction = LE_DIAGNOSIS.inverse_transform([prediction_encoded])[0]

    return render_template_string(HTML_RESULT_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5002)

