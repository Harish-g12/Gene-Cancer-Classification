from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Load saved models and scalers
model = pickle.load(open('cancer_model.pkl', 'rb'))
top50_idx = np.load('top50_idx.npy')  # Top 50 features selected by MI
selected_features_idx = np.load('selected_features_idx.npy')  # Final 10 selected features
scaler = pickle.load(open('scaler_10.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        # Get 10 inputs manually from the form
        genes = [float(request.form[f'gene{i}']) for i in range(1, 11)]
        genes = np.array(genes).reshape(1, -1)

        # Scale the input
        genes_scaled = scaler.transform(genes)

        # Predict
        prediction = model.predict(genes_scaled)[0]
        cancer_name, message, hospital = get_result(prediction)
        return render_template('result.html', prediction=cancer_name, message=message, hospital=hospital)
    except Exception as e:
        return f"Error: {e}"

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    try:
        file = request.files['file']
        data = pd.read_csv(file)

        if data.shape[1] == 20531:
            # Case 1: Full 20,531 gene file uploaded
            # Step 1: Pick top 50
            data_top50 = data.values[:, top50_idx]

            # Step 2: Pick final selected 10 features
            data_selected10 = data_top50[:, selected_features_idx]

            # Step 3: Scale
            data_scaled = scaler.transform(data_selected10)

        elif data.shape[1] == 10:
            # Case 2: Already 10 features file uploaded
            data_scaled = scaler.transform(data.values)

        else:
            return "Uploaded CSV must have either 20531 (full genes) or exactly 10 selected gene columns."

        # Step 4: Predict
        prediction = model.predict(data_scaled)[0]
        cancer_name, message, hospital = get_result(prediction)
        return render_template('result.html', prediction=cancer_name, message=message, hospital=hospital)
    except Exception as e:
        return f"Error: {e}"

def get_result(prediction):
    cancers = {
        0: ('Breast Cancer (BRCA)', "You are stronger than you think! 🌟", "Tata Memorial Hospital, Mumbai"),
        1: ('Colon Cancer (COAD)', "Every step you take is towards recovery. 💪", "AIIMS, New Delhi"),
        2: ('Kidney Cancer (KIRC)', "Believe in the journey of healing. 🌈", "Kidney Cancer Center, CMC Vellore"),
        3: ('Lung Cancer (LUAD)', "You have infinite courage within you. 🕊️", "Global Hospital, Mumbai"),
        4: ('Prostate Cancer (PRAD)', "You are braver than any diagnosis. 🛡️", "Apollo Hospitals, Chennai"),
    }
    return cancers[prediction]

if __name__ == "__main__":
    app.run(debug=True)
