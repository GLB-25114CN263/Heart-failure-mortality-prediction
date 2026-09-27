from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/heart_failure_model.pkl")

FEATURE_NAMES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["age"]),
            float(request.form["anaemia"]),
            float(request.form["creatinine_phosphokinase"]),
            float(request.form["diabetes"]),
            float(request.form["ejection_fraction"]),
            float(request.form["high_blood_pressure"]),
            float(request.form["platelets"]),
            float(request.form["serum_creatinine"]),
            float(request.form["serum_sodium"]),
            float(request.form["sex"]),
            float(request.form["smoking"]),
            float(request.form["time"])
        ]

        binary_values = [
            features[1],
            features[3],
            features[5],
            features[9],
            features[10]
        ]

        if any(value not in [0, 1] for value in binary_values):
            return render_template(
                "index.html",
                error="Binary fields must contain only 0 or 1."
            )

        input_data = pd.DataFrame(
            [features],
            columns=FEATURE_NAMES
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "High Risk" if prediction == 1 else "Low Risk"

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{probability:.1%}"
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid values in all fields."
        )


if __name__ == "__main__":
    app.run(debug=True)