import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):

        self.data = pd.read_csv(self.file_path)

        print("Dataset loaded successfully")
        print("Shape:", self.data.shape)

        return self.data

    def remove_duplicates(self):

        before = self.data.shape[0]

        self.data = self.data.drop_duplicates()

        after = self.data.shape[0]

        print("Duplicates removed:", before - after)

        return self.data

    def split_data(self):

        X = self.data.drop("target", axis=1)

        y = self.data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):

        X_train_scaled = self.scaler.fit_transform(X_train)

        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled