from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class FeatureSelection:

    def __init__(self, k=10):
        self.k = k
        self.selector = SelectKBest(
            score_func=f_classif,
            k=k
        )

    def fit_transform(self, X_train, y_train):

        X_train_selected = self.selector.fit_transform(
            X_train,
            y_train
        )

        return X_train_selected

    def transform(self, X_test):

        X_test_selected = self.selector.transform(
            X_test
        )

        return X_test_selected