from unittest import TestCase
from sklearn.datasets import load_iris, load_digits
import numpy as np
from sklearn.model_selection import train_test_split

from src.maxent import MaxEnt


def preprocess_col(col, min_val, max_val, bin_num):
    result = []
    bin_width = (max_val - min_val) / bin_num
    for item in col:
        if item < min_val:
            item = min_val
        if item > max_val:
            item = max_val

        result.append(np.floor((item - min_val) / bin_width))

    return result


class TestMaxEnt(TestCase):

    def test_predict(self):
        data = load_iris()

        X_data = data['data']
        y_data = data['target']

        preprocessed_x = []
        for i in range(len(X_data[0])):
            x_col = X_data[:, i]
            max_val = np.max(x_col)
            min_val = np.min(x_col)
            preprocessed_x.append(preprocess_col(x_col, min_val, max_val, 10))

        X_data = np.array(preprocessed_x).transpose()

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=15)
        model = MaxEnt(learning_rate=0.01)
        model.fit(X_train, y_train, max_iter=10)
        score = 0
        for X, y in zip(X_test, y_test):
            predict = model.predict_single(X)
            print(X, predict, y)
            if predict == y:
                score += 1
        print(score / len(y_test))
