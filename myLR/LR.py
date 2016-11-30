import numpy as np
import random
import math
import multiprocessing


def get_attrs(attrs, train_line):
    return [1] + list(map(lambda x: train_line[x], attrs))


def get_cost(error):
    mse = sum(error ** 2) / len(error)
    print(np.sqrt(mse))


class Regression:
    def __init__(self, train_data):
        self.attrs = 0
        self.train_data = train_data
        self.weight = 0

    def train_regression(self, attrs, limit, alpha):
        self.attrs = attrs
        train_data = np.array(list(map(lambda x: get_attrs(attrs, x), self.train_data)))
        train_result = np.array(list(map(lambda x: [x[-1]], self.train_data)))
        weight = 0.0000002 * np.random.random((len(attrs) + 1, 1))
        for i in range(limit):
            result = train_data.dot(weight)
            error = train_result - result
            weight += alpha * (2 / len(error) * train_data.T.dot(error))
            if i % 100000 == 0:
                get_cost(error)
        self.weight = weight

    def predict(self,test_data_in):
        test_data = np.array(list(map(lambda x: get_attrs(self.attrs, x), test_data_in)))
        test_result = np.array(list(map(lambda x: [x[-1]], test_data_in)))
        result = test_data.dot(self.weight)
        error = test_result - result
        print("Predict finish!")
        get_cost(error)


def read_train(p, path):
    with open(path) as f:
        lines = f.readlines()
        lines.remove(lines[0])
        divide = math.floor(len(lines) * p)
        data = list()

        for i in range(len(lines)):
            splits = list()
            splits.extend(lines[i].replace('\n', '').split(','))
            splits = list(map(lambda x: float(x), splits))
            data.append(splits)
        train_data = random.sample(data, divide)
        for line in train_data:
            data.remove(line)
        test_data = data

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = read_train(0.9, "./train.csv")
    regression = Regression(train_data)
    regression.train_regression((range(9)), 1000000, 0.00000000000001)
    regression.predict(test_data)
