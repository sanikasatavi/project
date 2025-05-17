import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import requests
import numpy as np
import requests

earthquake_data = pd.read_csv("Cleaned_Earthquakes_India.csv")
earthquake_data['year'] = pd.to_datetime(earthquake_data['time'], errors='coerce').dt.year

flood_data = pd.read_csv("Cleaned_Floods_India.csv")
if 'Year' not in flood_data.columns:
    flood_data['Year'] = 2023  

eq_imputer = SimpleImputer(strategy='mean')
X_eq = eq_imputer.fit_transform(earthquake_data[['latitude', 'longitude', 'depth', 'mag', 'year']])
y_eq = earthquake_data['mag'].apply(lambda x: 0 if x < 4 else (1 if x < 5 else (2 if x < 6 else 3)))
X_eq_train, X_eq_test, y_eq_train, y_eq_test = train_test_split(X_eq, y_eq, test_size=0.2, random_state=42)
eq_model = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, eval_metric='mlogloss')
eq_model.fit(X_eq_train, y_eq_train)
joblib.dump(eq_model, "earthquake_model.pkl")

flood_imputer = SimpleImputer(strategy='mean')
X_flood = flood_imputer.fit_transform(flood_data[['Latitude', 'Longitude', 'Year']])
y_flood = flood_data['Severity'].apply(lambda x: 0 if x <= 0 else (1 if x == 1 else (2 if x == 2 else 3)))
X_flood_train, X_flood_test, y_flood_train, y_flood_test = train_test_split(X_flood, y_flood, test_size=0.2, random_state=42)
flood_model = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, eval_metric='mlogloss')
flood_model.fit(X_flood_train, y_flood_train)
joblib.dump(flood_model, "flood_model.pkl")

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Raw request data:", request.data)
        if request.content_type != "application/json":
            return jsonify({"error": "Request must be in JSON format."}), 400

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON data received or malformed JSON."}), 400
        print("Parsed JSON:", data)

        location = data.get('location', '').strip()
        year = data.get('year')

        if not location:
            return jsonify({"error": "Location is required."}), 400
        if year is None:
            return jsonify({"error": "Year must be provided."}), 400

        year = int(year)
        print(f"Converted inputs - Year: {year}")

        headers = {"User-Agent": "DisasterPredictionApp/1.0 (contact@example.com)"}
        response = requests.get(f"https://nominatim.openstreetmap.org/search?format=json&q={location}", headers=headers)
        print("Nominatim API Response:", response.text)
        location_data = response.json()
        if not location_data:
            return jsonify({"error": "Invalid location or API blocked."}), 400

        latitude = float(location_data[0]['lat'])
        longitude = float(location_data[0]['lon'])
        print(f"Retrieved coordinates: {latitude}, {longitude}")

        eq_subset = earthquake_data[(earthquake_data['latitude'].between(latitude - 1, latitude + 1)) &
                                    (earthquake_data['longitude'].between(longitude - 1, longitude + 1))]
        flood_subset = flood_data[(flood_data['Latitude'].between(latitude - 1, latitude + 1)) &
                                  (flood_data['Longitude'].between(longitude - 1, longitude + 1))]
        
        eq_yearly_trend = eq_subset.groupby('year').size()
        flood_yearly_trend = flood_subset.groupby('Year').size()
        
        eq_trend_factor = ((eq_yearly_trend.get(year, 0) + 1) / (eq_yearly_trend.mean() + 1)) if not eq_yearly_trend.empty else 1.0
        flood_trend_factor = ((flood_yearly_trend.get(year, 0) + 1) / (flood_yearly_trend.mean() + 1)) if not flood_yearly_trend.empty else 1.0
        
        print(f"Earthquake trend factor: {eq_trend_factor}, Flood trend factor: {flood_trend_factor}")

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    eq_features = eq_imputer.transform([[latitude, longitude, 10.0, 5.0, year]])
    eq_probs = eq_model.predict_proba(eq_features)[0]
    eq_risk_score = np.dot(eq_probs, [0, 25, 50, 75]) * eq_trend_factor
    eq_risk_score = np.clip(eq_risk_score, 1, 99)  

    flood_features = flood_imputer.transform([[latitude, longitude, year]])
    flood_probs = flood_model.predict_proba(flood_features)[0]
    flood_risk_score = np.dot(flood_probs, [0, 25, 50, 75]) * flood_trend_factor
    flood_risk_score = np.clip(flood_risk_score, 1, 99)

    print(f"Final Earthquake Risk: {eq_risk_score}%, Final Flood Risk: {flood_risk_score}%")

    store_data = {
        "location": location,
        "year": year,
        "earthquake_risk": eq_risk_score,
        "flood_risk": flood_risk_score
    }
    store_response = requests.post("http://127.0.0.1:5001/store", json=store_data)

    if store_response.status_code != 200:
        print("Error storing data:", store_response.text)

    return jsonify({
        "earthquake_risk": f"{eq_risk_score:.2f}%%",
        "flood_risk": f"{flood_risk_score:.2f}%%"
    })
    

if __name__ == '__main__':
    app.run(debug=True)
