# app.py
import pickle
import time
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

from utils.preprocessing import prepare_input, FEATURES
from utils.validation import validate
from explainability.shap_explainer import generate_shap_for_class
from explainability.lime_explainer import generate_lime_for_class
from explainability.global_shap import generate_global_shap
from explainability.global_shap_by_crop import generate_cropwise_shap

app  = Flask(__name__)

model = pickle.load(open("models/decision_tree.pkl", "rb"))
df = pd.read_csv("data/crop_data.csv")
X_train = df.drop("label", axis=1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        error = validate(request.form)
        if error:
            return render_template("predict.html", error=error)
        
        X = prepare_input(request.form)
        proba = model.predict_proba(X)[0]
        classes = model.classes_

        top3_idx = np.argsort(proba)[::-1][:3]
        results = []

        for rank, idx in enumerate(top3_idx, start=1):
            crop_name = classes[idx]

            shap_path = f"static/explain/shap_rank_{rank}.png"
            lime_path = f"static/explain/lime_rank_{rank}.png"

            generate_shap_for_class(
                model=model,
                X_instance=X,
                feature_names=FEATURES,
                class_index=idx,
                class_name=crop_name,
                save_path=shap_path,
                large=(rank == 1)
            )

            generate_lime_for_class(
                model=model,
                X_train=X_train,
                X_instance=X,
                feature_names=FEATURES,
                class_index=idx,
                class_name=crop_name,
                save_path=lime_path,
                large=(rank == 1)
            )

            results.append({
                "rank": rank,
                "crop": crop_name,
                "probability": round(float(proba[idx]), 4),
                "shap_img": shap_path,
                "lime_img": lime_path
            })

        generate_global_shap(
            model, X_train, FEATURES, "static/explain/global_shap.png"
        )

        generate_cropwise_shap(
            model,
            X_train,
            FEATURES,
            "static/explain/global_shap_by_crop.png"
        )

        return render_template(
            "predict.html",
            results=results,
            ts = int(time.time())
        )

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
