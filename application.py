from pathlib import Path
import math
import pickle

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app)

with (ROOT / "LinearRegressionModel.pkl").open("rb") as model_file:
    model = pickle.load(model_file)
car = pd.read_csv(ROOT / "Cleaned_Car_data.csv")


@app.route("/", methods=["GET"])
def index():
    companies = sorted(car["company"].dropna().unique().tolist())
    car_models = sorted(car["name"].dropna().unique().tolist())
    years = sorted(car["year"].dropna().unique().tolist(), reverse=True)
    fuel_types = sorted(car["fuel_type"].dropna().unique().tolist())
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
    )


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    company = request.form.get("company", "").strip()
    car_model = request.form.get("car_models", "").strip()
    year = request.form.get("year", "").strip()
    fuel_type = request.form.get("fuel_type", "").strip()
    driven = request.form.get("kilo_driven", "").strip()

    if not all((company, car_model, year, fuel_type, driven)):
        return jsonify({"error": "Please complete all fields."}), 400

    try:
        year_value = int(year)
        driven_value = float(driven)
    except ValueError:
        return jsonify({"error": "Year and kilometres driven must be valid numbers."}), 400

    if not math.isfinite(driven_value) or driven_value < 0:
        return jsonify({"error": "Kilometres driven must be a non-negative number."}), 400

    valid_car = car[
        (car["name"] == car_model)
        & (car["company"] == company)
        & (car["year"] == year_value)
        & (car["fuel_type"] == fuel_type)
    ]
    if valid_car.empty:
        return jsonify({"error": "The selected car combination is not in the dataset."}), 400

    input_data = pd.DataFrame(
        [[car_model, company, year_value, driven_value, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
    )

    try:
        prediction = float(np.round(model.predict(input_data)[0], 2))
        return jsonify({"prediction": prediction})
    except Exception:
        app.logger.exception("Car price prediction failed")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    app.run()
