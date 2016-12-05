import numpy as np
import math
import random
import multiprocessing
import copy


class KNN:
    def __init__(self, train_data):
        self.train_data = train_data

    def test(self, k, test_data):
        correct = 0
        for line in test_data:
            result = list(map(lambda x: get_dist(x, line), self.train_data))
            result.sort(key=lambda x: x[0])
            result = result[:k]
            emotions = {"anger": 0, "disgust": 0, "fear": 0, "guilt": 0, "joy": 0, "sadness": 0, "shame": 0}
            for res in result:
                # emotions[res[1]] += (1 / res[0]) if res[0] != 0 else 0
                emotions[res[1]] += 1
            predict = sorted(emotions.items(), key=lambda x: x[1])[-1][0]
            print(predict)
        #     if predict == line[0]:
        #         correct += 1
        #         print(correct / len(test_data))
        #
        # print("Test finish! Correct = ", correct)
        return correct


def get_dist(train_line, test_line):
    return len(train_line[1] ^ test_line[1]), train_line[0]


def read_train(train_path, test_path, stop_path):
    with open(stop_path) as f:
        stop_word = set(f.read().split())

    with open(train_path) as f:
        lines = f.readlines()
        train_data = list()
        for line in lines:
            split = line.replace("\n", "").split(",")
            split[1] = set(split[1].split(" "))
            drop = split[1] & stop_word
            # drop = set()
            # for word in split[1]:
            #     if len(word) < 3:
            #         drop.add(word)
            # split[1] ^= drop
            train_data.append(split)

    with open(test_path) as f:
        lines = f.readlines()
        test_data = list()
        for line in lines:
            split = line.replace("\n", "").split(",")
            split[1] = set(split[1].split(" "))
            drop = split[1] & stop_word
            # drop = set()
            # for word in split[1]:
            #     if len(word) < 3:
            #         drop.add(word)
            # split[1] ^= drop
            test_data.append(split)

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = read_train("./train.csv", "./test.csv", "./stop_word.txt")
    print("Read finish!")
    knn = KNN(train_data)
    knn.test(11, test_data)
    print("Multiprocessing finish!")
