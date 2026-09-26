import pandas as pd

from sklearn.preprocessing import PowerTransformer


class YeoJohnsonTransformation:

    def __init__(self):
        self.transformer = PowerTransformer(
            method="yeo-johnson"
        )

    def transform(self, X_train, X_test):

        X_train_yeo = self.transformer.fit_transform(
            X_train
        )

        X_test_yeo = self.transformer.transform(
            X_test
        )

        return X_train_yeo, X_test_yeo