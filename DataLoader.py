import pandas as pd
import numpy as np


class DataLoader:

    def __init__(self, path):
        self.path = path
        return

    def load_data(self):
        data = pd.read_csv(self.path)

        # Data normalization
        for label, column in data.items():

            if data[label].dtypes in ["int32", "int64", "float32", "float64"]:
                if np.greater(column.max(), 1.0) | np.less(column.min(), 0.0):
                    mean = column.mean()
                    data[label] = data[label].apply(lambda x: x / mean)
                    data[label] = data[label].apply(lambda x: 0.0 if np.isnan(x) else x)

        return data
