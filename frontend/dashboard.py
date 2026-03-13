import streamlit as st
import pandas as pd
import joblib
import sys
import os

# allow access to model folder
sys.path.append(os.path.abspath(".."))

from model.columnar import map_columns

# load trained model
model = joblib.load("../model/risk_model.pkl")

st.title("AI Academic Risk Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    # read excel
    df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    # detect columns automatically
    mapping = map_columns(df)

    st.subheader("Detected Columns")
    st.write(mapping)

    name_col = mapping["name"]
    attendance_col = mapping["attendance"]
    marks_col = mapping["marks"]
    study_hours_col = mapping["study_hours"]

    predictions = []
    reasons_list = []

    # prediction loop
    for _, row in df.iterrows():

        attendance = float(row[attendance_col])
        marks = float(row[marks_col])
        study_hours = float(row[study_hours_col])

        features = [[attendance, marks, study_hours]]

        prediction = model.predict(features)[0]

        result = "High Risk" if prediction == 1 else "Low Risk"
        predictions.append(result)

        # determine reason for attention
        reasons = []

        if attendance < 75:
            reasons.append("Low Attendance")

        if marks < 60:
            reasons.append("Low Marks")

        if study_hours < 2:
            reasons.append("Low Study Hours")

        reason_text = ", ".join(reasons) if reasons else "Stable"

        reasons_list.append(reason_text)

    # add results to dataframe
    df["Risk"] = predictions
    df["Attention_Reason"] = reasons_list

    st.subheader("Prediction Results")
    st.dataframe(df)

    # dashboard chart
    st.subheader("Class Risk Overview")
    risk_counts = df["Risk"].value_counts()
    st.bar_chart(risk_counts)

    # students needing attention
    st.subheader("Students Needing Attention")

    high_risk_students = df[df["Risk"] == "High Risk"]

    if len(high_risk_students) > 0:
        st.dataframe(high_risk_students)
    else:
        st.success("No students currently at risk.")

    # download report
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction Report",
        csv,
        "student_risk_report.csv",
        "text/csv"
    )
