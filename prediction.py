import os
import joblib
import numpy as np


class HeartDiseasePrediction:

    def __init__(self):

        model_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "models",
            "scaled_model.pkl"
        )

        data = joblib.load(
            model_path
        )

        self.model = data["model"]

        self.scaler = data["scaler"]

    def predict(self, values):

        values = np.array(
            values,
            dtype=float
        ).reshape(1, -1)

        values = self.scaler.transform(
            values
        )

        prediction = self.model.predict(
            values
        )[0]

        probability = self.model.predict_proba(
            values
        )[0][1]

        return prediction, probability