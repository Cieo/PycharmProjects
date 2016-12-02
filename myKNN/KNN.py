import numpy as np
import math
import random


class KNN:
    def __init__(self, all_word, train_data):
        self.train_data = train_data
        self.all_word = all_word

    def train(self):
        size = len(self.all_word)
        for line in self.train_data:
            new_line = [0] * size
            for word in line[1]:
                new_line[self.all_word.index(word)] += 1
            line[1] = new_line
        print("Train finish!")

    def test(self, k, test_data):
        size = len(self.all_word)
        correct = 0
        for line in test_data:
            new_line = [0] * size
            for word in line[1]:
                new_line[self.all_word.index(word)] += 1
            line[1] = new_line
            result = list(map(lambda x: get_dist(x, line), self.train_data))
            result.sort(key=lambda x: x[0])
            result = result[:k]
            emotions = {"anger": 0, "disgust": 0, "fear": 0, "guilt": 0, "joy": 0, "sadness": 0, "shame": 0}
            for res in result:
                emotions[res[1]] += 1
            predict = sorted(emotions.items(), key=lambda x: x[1])[-1][0]
            if predict == line[0]:
                correct += 1
                print("Right!", correct / size)
            else:
                print("Wrong!")
        print("Test finish!")


def get_dist(train_line, test_line):
    x1 = np.array(train_line[1])
    x2 = np.array(test_line[1])
    return sum((x1 - x2) ** 2), train_line[0]


def read_train(path, p):
    with open(path) as f:
        lines = f.readlines()
        data = list()
        all_word = list()
        for line in lines:
            split = line.replace("\n", "").split(",")
            split[1] = split[1].split(" ")
            all_word += split[1]
            data.append(split)
        divide = math.floor(len(lines) * p)
        train_data = random.sample(data, divide)
        for i in train_data:
            data.remove(i)
        test_data = data
    return list(set(all_word)), train_data, test_data


if __name__ == "__main__":
    all_word, train_data, test_data = read_train("./train.csv", 0.8)
    print("Read finish!")
    knn = KNN(all_word, train_data)
    knn.train()
    knn.test(10, test_data)
