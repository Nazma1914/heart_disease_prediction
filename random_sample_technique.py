import pandas as pd


class RandomSample:

    def __init__(self, file_path):
        self.file_path = file_path

    def get_sample(self, sample_size=20):

        data = pd.read_csv(self.file_path)

        data = data.drop_duplicates()

        sample_size = min(
            sample_size,
            len(data)
        )

        sample = data.sample(
            n=sample_size,
            random_state=42
        )

        return sample


if __name__ == "__main__":

    obj = RandomSample(
        "heart_disease.csv"
    )

    print(
        obj.get_sample()
    )