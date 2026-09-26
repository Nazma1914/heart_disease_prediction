from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


class AllModels:

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run_models(self):

        models = {

            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    random_state=42
                ),

            "KNN":
                KNeighborsClassifier(),

            "SVM":
                SVC()
        }

        results = {}

        for name, model in models.items():

            model.fit(
                self.X_train,
                self.y_train
            )

            prediction = model.predict(
                self.X_test
            )

            accuracy = accuracy_score(
                self.y_test,
                prediction
            )

            results[name] = accuracy

            print(
                name,
                "Accuracy:",
                round(accuracy, 4)
            )

        return results