import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


data = pd.read_csv("../data/student_data.csv")

X = data[["attendance","internal_marks","assignment_score"]]
y = data['risk']

model = RandomForestClassifier()

model.fit(X,y)

os.makedirs("model", exist_ok=True)
joblib.dump(model,"risk_model.pkl")

print("Model trained and saved!")