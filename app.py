from flask import Flask
from flask import render_template
from flask import request

from prediction import HeartDiseasePrediction


app = Flask(__name__)

predictor = HeartDiseasePrediction()


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        feature_names = [

            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"

        ]

        values = []

        for feature in feature_names:

            value = request.form[
                feature
            ]

            values.append(
                float(value)
            )

        prediction, probability = predictor.predict(
            values
        )

        if prediction == 1:

            result = (
                "Higher likelihood of heart disease"
            )

        else:

            result = (
                "Lower likelihood of heart disease"
            )

        probability = round(
            probability * 100,
            2
        )

        return render_template(
            "result.html",
            result=result,
            probability=probability
        )

    except Exception as e:

        return render_template(
            "result.html",
            result="Error occurred",
            probability=None,
            error=str(e)
        )


if __name__ == "__main__":

    app.run(
        debug=True
    )