import json

import numpy as np
import random
import math
import multiprocessing


def get_attrs(attrs, train_line):
    return [1] + list(map(lambda x: train_line[x], attrs))

def get_dist(x1, x2):
    return sum(list(map(lambda x: (x1[x] - x2[x]) ** 2, range(len(x1) - 1))))


def get_choice(center, line):
    result = list(map(lambda x: get_dist(x, line), center))
    min_value = min(result)
    return result.index(min_value)


def train_cluster(train_data, k):
    center_old = random.sample(train_data, k)
    center = random.sample(train_data, k)
    while center != center_old:
        cluster = [list() for x in range(k)]
        center_old = center
        for line in train_data:
            cluster[get_choice(center, line)].append(line)
        center = list(map(lambda x: list(sum(np.array(x)) / len(x)), cluster))
    return center, cluster


def get_cluster(test_data, center):
    cluster = [list() for x in range(len(center))]
    for line in test_data:
        cluster[get_choice(center, line)].append(line)
    return cluster


class Regression:
    def __init__(self, train_data):
        self.attrs = 0
        self.train_data = train_data
        self.weight = 0

    def train_regression(self, attrs, limit, alpha):
        self.attrs = attrs
        train_sample = self.train_data
        train_data = np.array(list(map(lambda x: get_attrs(attrs, x), train_sample)))
        train_result = np.array(list(map(lambda x: [x[-1]], train_sample)))
        weight = 0.0000002 * np.random.random((np.shape(train_data)[1], 1))
        for i in range(limit):
            result = train_data.dot(weight)
            error = train_result - result
            weight += alpha * (2 / len(error) * train_data.T.dot(error))
        self.weight = weight

    def predict(self, test_data_in):
        test_data = np.array([1] + test_data_in[:-1])
        result = test_data.dot(self.weight)
        print(result)


def read_train(train_path, test_path):
    train_data = list()
    test_data = list()
    max_value = [-9999] * 9
    min_value = [9999] * 9

    with open(train_path) as f:
        lines = f.readlines()
        lines.remove(lines[0])

        for i in range(len(lines)):
            splits = list()
            splits.extend(lines[i].replace('\n', '').split(','))
            splits = list(map(lambda x: float(x), splits))
            for j in range(len(splits) - 1):
                max_value[j] = max(splits[j], max_value[j])
                min_value[j] = min(splits[j], min_value[j])
            train_data.append(splits)

    with open(test_path) as f:
        lines = f.readlines()
        lines.remove(lines[0])

        for i in range(len(lines)):
            splits = list()
            splits.extend(lines[i].replace('\n', '').replace('?', '0').split(','))
            splits = list(map(lambda x: float(x), splits))
            for j in range(len(splits) - 1):
                max_value[j] = max(splits[j], max_value[j])
                min_value[j] = min(splits[j], min_value[j])
            test_data.append(splits)

        for i in range(len(train_data)):
            for j in range((np.shape(train_data)[1]) - 1):
                (train_data[i])[j] = ((train_data[i])[j] - min_value[j]) / (max_value[j] - min_value[j])

        for i in range(len(test_data)):
            for j in range((np.shape(test_data)[1]) - 1):
                (test_data[i])[j] = ((test_data[i])[j] - min_value[j]) / (max_value[j] - min_value[j])

    center, train_data = train_cluster(train_data, 1)
    return center, train_data, test_data


def run_test(train_data_i, test_data_i):
    regression = Regression(train_data_i)
    regression.train_regression(([0, 1, 2, 3, 4, 5, 6, 7, 8]), 10000000, 0.00001)
    regression.predict(test_data_i)


if __name__ == "__main__":
    center, train_data, test_data = read_train("./train.csv", "./test.csv")
    print("Finish Read!")
    regressions = list()
    for i in range(len(center)):
        regre = Regression(train_data[i])
        regre.train_regression(([0, 1, 2, 3, 4, 5, 6, 7, 8]), 1000000, 0.00001)
        regressions.append(regre)
    print("Finish Train!")
    for line in test_data:
        regressions[get_choice(center, line)].predict(line)
