from flask import Flask, request, jsonify
import sqlite3
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

earthquake_model = joblib.load("earthquake_model.pkl")
flood_model = joblib.load("flood_model.pkl")

db_path = "disaster_data.db"
def init_db():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT,
                year INTEGER,
                earthquake_risk REAL,
                flood_risk REAL
            )
        ''')
        conn.commit()

init_db()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    location = data.get("location")
    year = data.get("year")
    
    if not location or not year:
        return jsonify({"error": "Missing location or year"}), 400
    
    earthquake_risk = earthquake_model.predict([[year]])[0]  
    flood_risk = flood_model.predict([[year]])[0]
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (location, year, earthquake_risk, flood_risk) VALUES (?, ?, ?, ?)",
                       (location, year, earthquake_risk, flood_risk))
        conn.commit()
    
    return jsonify({"earthquake_risk": earthquake_risk, "flood_risk": flood_risk})

@app.route('/store', methods=['POST'])
def store_data():
    data = request.get_json()
    location = data.get("location")
    year = data.get("year")
    earthquake_risk = data.get("earthquake_risk")
    flood_risk = data.get("flood_risk")
    
    if not location or not year or earthquake_risk is None or flood_risk is None:
        return jsonify({"error": "Missing data"}), 400
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (location, year, earthquake_risk, flood_risk) VALUES (?, ?, ?, ?)",
                       (location, year, earthquake_risk, flood_risk))
        conn.commit()
    
    return jsonify({"message": "Data stored successfully"})

@app.route('/get_data', methods=['GET'])
def get_data():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")
        data = cursor.fetchall()
    return jsonify({"predictions": data})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

