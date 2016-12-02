import numpy as np
import random
import math
import multiprocessing


def get_attrs(attrs, train_line):
    return [1] + list(map(lambda x: train_line[x], attrs))


def get_cost(error):
    mse = sum(error ** 2) / len(error)
    print(np.sqrt(mse))


def get_dist(x1, x2):
    return sum(list(map(lambda x: (x1[x] - x2[x]) ** 2, range(len(x1) - 1))))


def get_choice(center, line):
    line_copy = line.copy()
    line_copy[-1] = 0
    result = list(map(lambda x: get_dist(x, line), center))
    max_value = max(result)
    return result.index(max_value)


def get_cluster(train_data, k):
    center_old = random.sample(train_data, k)
    center = random.sample(train_data, k)
    cluster = 0
    print("center", center)
    print("centol", center_old)
    while center != center_old:
        cluster = [[]] * k
        print(cluster)
        center_old = center
        for line in train_data:
            cluster[get_choice(center, line)].append(line)
        center = list(map(lambda x: list(sum(np.array(x)) / len(x)), cluster))
        print("center", center)
        print("centol", center_old)
    print(cluster[0] == cluster[1])
    return center,cluster


class Regression:
    def __init__(self, train_data):
        self.attrs = 0
        self.train_data = train_data
        self.weight = 0

    def train_regression(self, attrs, limit, alpha, samp_num):
        self.attrs = attrs
        train_sample = random.sample(self.train_data, samp_num)
        train_data = np.array(list(map(lambda x: get_attrs(attrs, x), train_sample)))
        train_result = np.array(list(map(lambda x: [x[-1]], train_sample)))
        weight = 0.0000002 * np.random.random((np.shape(train_data)[1], 1))
        for i in range(limit):
            result = train_data.dot(weight)
            error = train_result - result
            weight += alpha * (2 / len(error) * train_data.T.dot(error))
            if i % 100000 == 0:
                get_cost(error)
        self.weight = weight

    def predict(self, test_data_in):
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
        max_value = [-9999] * 9
        min_value = [9999] * 9
        for i in range(len(lines)):
            splits = list()
            splits.extend(lines[i].replace('\n', '').split(','))
            splits = list(map(lambda x: float(x), splits))
            for j in range(len(splits) - 1):
                max_value[j] = max(splits[j], max_value[j])
                min_value[j] = min(splits[j], min_value[j])
            data.append(splits)
        for i in range(len(lines)):
            for j in range((np.shape(data)[1]) - 1):
                (data[i])[j] = ((data[i])[j] - min_value[j]) / (max_value[j] - min_value[j])
        get_cluster(data, 3)
        train_data = random.sample(data, divide)
        for line in train_data:
            data.remove(line)
        test_data = data

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = read_train(0.9, "./train.csv")
    print("Finish Read!")
    # regression = Regression(train_data)
    # regression.train_regression(([0, 1, 2, 3, 4, 5, 6, 7, 8]), 100000000, 0.00001, 10000)
    # regression.predict(test_data)
