import numpy as np
import math
import random


class Forest:
    def __init__(self, pos_data, neg_data):
        self.length, self.width = np.shape(pos_data)
        self.trees = list()
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.tree_num = 0

    def bulid_forest(self, tree_num, attr_num, samp_num):
        self.tree_num = tree_num
        for i in range(tree_num):
            self.trees.append(
                Tree(random.sample(range(self.width - 1), attr_num), random.sample(self.pos_data, samp_num),
                     random.sample(self.neg_data, samp_num)))

    def predict(self, test_data, k):
        result = list()
        correct = 0
        progress = 0
        for test_line in test_data:
            result = list(map(lambda x: x.predict(test_line, k), self.trees))
            one = len(list(filter(lambda x: x == 1, result)))
            if one > self.tree_num / 2 and test_line[-1] == 1:
                correct += 1
                print("right!", one, test_line[-1])
            elif one < self.tree_num / 2 and test_line[-1] == 0:
                correct += 1
                print("right!", one, test_line[-1])
            else:
                print("wrong!",one, test_line[-1])
            progress += 1
            print(str(progress/len(test_data) * 100)+"%")
        print(correct / len(test_data))


def get_dist(attr, train_line, test_line):
    merge = list(zip(train_line, test_line))
    sum = 0
    for i in attr:
        tr, te = merge[i]
        sum += (tr - te) ** 2
    return [sum, train_line[-1]]


class Tree:
    def __init__(self, attr, pos_data, neg_data):
        self.train_data = pos_data + neg_data
        self.attr = attr

    def predict(self, test_line, k):
        result = list()
        for train_line in self.train_data:
            result.append(get_dist(self.attr, train_line, test_line))
        result.sort(key=lambda i: i[0])
        # print(result, test_line)
        one = len(list(filter(lambda x: x[-1] == 1, result[:100])))
        if one > k / 2:
            return 1
        else:
            return 0


def get_pos(x):
    if x[-1] == 1:
        return x


def get_neg(x):
    if x[-1] == 0:
        return x


def read_train(p, path):
    with open(path) as f:
        lines = f.readlines()
        lines.remove(lines[0])
        divide = math.floor(len(lines) * p)
        train_data = list()
        test_data = list()
        for i in range(divide):
            splits = list()
            splits.extend(lines[i].split(','))
            train_data.append(list(float(a) for a in splits))
        for i in range(divide, len(lines)):
            splits = list()
            splits.extend(lines[i].split(','))
            test_data.append(list(float(a) for a in splits))
        max_col = list()
        for i in range(np.shape(train_data)[1]):
            max_col.append(max(list(map(lambda x:x[i],train_data+test_data))))
            if max_col[-1] == 0:
                max_col[-1] = 1
        for line in train_data:
            for i in range(len(line)):
                line[i] /= max_col[i]
        for line in test_data:
            for i in range(len(line)):
                line[i] /= max_col[i]
        pos_train = list(filter(get_pos, train_data))
        neg_train = list(filter(get_neg, train_data))
    return pos_train, neg_train, test_data


if __name__ == "__main__":
    pos_train, neg_train, test_data = read_train(0.8, "./train.csv")
    forest = Forest(pos_train, neg_train)
    forest.bulid_forest(101, 10, 1000)
    print("Forest built up")
    forest.predict(test_data, 71)
