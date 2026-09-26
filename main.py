import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from log_code import setup_log
from preprocessing import DataPreprocessing
from yeo_timing import YeoJohnsonTransformation
from all_models import AllModels


class HeartDiseaseProject:

    def __init__(self):

        self.base_path = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.data_path = os.path.join(
            self.base_path,
            "heart_disease.csv"
        )

        self.model_path = os.path.join(
            self.base_path,
            "models",
            "scaled_model.pkl"
        )

        self.logger = setup_log(
            "main"
        )

    def run(self):

        self.logger.info(
            "Project started"
        )

        # -------------------------
        # DATA PREPROCESSING
        # -------------------------

        preprocessing = DataPreprocessing(
            self.data_path
        )

        data = preprocessing.load_data()

        self.logger.info(
            "Dataset loaded"
        )

        data = preprocessing.remove_duplicates()

        self.logger.info(
            "Duplicates removed"
        )

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = preprocessing.split_data()

        self.logger.info(
            "Train test split completed"
        )

        # -------------------------
        # SCALING
        # -------------------------

        (
            X_train_scaled,
            X_test_scaled
        ) = preprocessing.scale_data(
            X_train,
            X_test
        )

        self.logger.info(
            "Standard scaling completed"
        )

        # -------------------------
        # YEO-JOHNSON
        # -------------------------

        yeo = YeoJohnsonTransformation()

        (
            X_train_yeo,
            X_test_yeo
        ) = yeo.transform(
            X_train,
            X_test
        )

        self.logger.info(
            "Yeo-Johnson transformation completed"
        )

        # -------------------------
        # ALL MODELS
        # -------------------------

        print("\nMODEL COMPARISON")
        print("----------------")

        models = AllModels(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        models.run_models()

        # -------------------------
        # FINAL MODEL
        # -------------------------

        print("\nFINAL MODEL")
        print("-----------")

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        prediction = model.predict(
            X_test_scaled
        )

        probability = model.predict_proba(
            X_test_scaled
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        auc = roc_auc_score(
            y_test,
            probability
        )

        print(
            "Accuracy:",
            round(accuracy, 4)
        )

        print(
            "ROC-AUC:",
            round(auc, 4)
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            confusion_matrix(
                y_test,
                prediction
            )
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_test,
                prediction
            )
        )

        # -------------------------
        # SAVE MODEL
        # -------------------------

        os.makedirs(
            os.path.dirname(
                self.model_path
            ),
            exist_ok=True
        )

        joblib.dump(
            {
                "model": model,
                "scaler": preprocessing.scaler,
                "features": list(
                    X_train.columns
                )
            },
            self.model_path
        )

        self.logger.info(
            "Scaled model saved"
        )

        self.logger.info(
            "Project completed"
        )


if __name__ == "__main__":

    project = HeartDiseaseProject()

    project.run()