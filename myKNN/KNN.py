import numpy as np
import math
import random
import multiprocessing
import copy


class KNN:
    def __init__(self, train_data):
        self.train_data = train_data

    def train(self):
        print("Train finish!")

    def test(self, k, test_data):
        correct = 0
        i = 0
        for line in test_data:
            print(i)
            i += 1
            result = list(map(lambda x: get_dist(x, line), self.train_data))
            result.sort(key=lambda x: x[0])
            result = result[:k]
            emotions = {"anger": 0, "disgust": 0, "fear": 0, "guilt": 0, "joy": 0, "sadness": 0, "shame": 0}
            for res in result:
                emotions[res[1]] += 1 / res[0] if res[0] != 0 else 0
            predict = sorted(emotions.items(), key=lambda x: x[1])[-1][0]
            if predict == line[0]:
                correct += 1
            else:
                pass
        print("Test finish! Correct = ", correct)
        return correct


def get_dist(train_line, test_line):
    x1 = np.array(train_line[1])
    x2 = np.array(test_line[1])
    return sum((x1 - x2) ** 2), train_line[0]


def read_train(path, p):
    # with open("./stop_word.txt") as f:
    #     stop_word = list()
    #     lines = f.readlines()
    #     for line in lines:
    #         splits = line.replace("\n", "").split()
    #         stop_word.extend(splits)

    with open(path) as f:
        lines = f.readlines()
        data = list()
        all_word = list()
        for line in lines:
            split = line.replace("\n", "").split(",")
            split[1] = split[1].split(" ")
            for word in split[1]:
                # if stop_word.count(word) != 0:
                #     split[1].remove(word)
                if len(word) < 3:
                    split[1].remove(word)
            all_word += split[1]
            data.append(split)
        all_word = list(set(all_word))
        size = len(all_word)
        for line in data:
            new_line = [0] * size
            for word in line[1]:
                new_line[all_word.index(word)] = 1
            line[1] = new_line
        divide = math.floor(len(lines) * p)
        train_data = random.sample(data, divide)
        for i in train_data:
            data.remove(i)
        test_data = data
        print(len(all_word))
        print(len(train_data))
        print(len(test_data))
    return train_data, test_data


def divide_data(data, n):
    divide = math.floor(len(data) / n)
    divided = list()
    for k in range(n - 1):
        divided.append(data[:divide])
        for line in divided[-1]:
            data.remove(line)
    divided.append(data[:])
    print("Divide finish!")
    return divided


def run_test(i):
    train_data, test_data = read_train("./train.csv", 0.8)
    print("Read finish!")
    knn = KNN(train_data)
    knn.train()
    print(knn.test(i, test_data) / len(test_data), " k = " + str(i))


if __name__ == "__main__":
    pool = multiprocessing.Pool(4)
    train_data, test_data = read_train("./train.csv", 0.8)
    print("Read finish!")
    test_data = divide_data(test_data, 8)
    knn = KNN(train_data)
    knn.train()
    for i in range(8):
        pool.apply_async(knn.test, (1, test_data[i],))
    pool.close()
    pool.join()
    print("Multiprocessing finish!")
    # run_test(10)
