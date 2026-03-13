from flask import Flask, request, jsonify
import joblib
print("Flask backend loaded successfully")
app = Flask(__name__)

model = joblib.load("model/risk_model.pkl")

@app.route("/")
def home():
    return "Backend working!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    attendance = float(data["attendance"])
    marks = float(data["marks"])
    study_hours = float(data["study_hours"])

    features = [[attendance, marks, study_hours]]

    prediction = model.predict(features)[0]

    result = "High Risk" if prediction == 1 else "Low Risk"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
