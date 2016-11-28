import numpy as np
import math
import random


class Forest:
    def __init__(self, train_data):
        self.length, self.width = np.shape(train_data)
        self.trees = list()
        self.train_data = train_data
        self.tree_num = 0

    def bulid_forest(self, tree_num, attr_num, samp_num):
        self.tree_num = tree_num
        for i in range(tree_num):
            self.trees.append(
                Tree(random.sample(range(self.width - 1), attr_num), random.sample(self.train_data, samp_num)))

    def predict(self, test_data):
        pass


class Tree:
    def __init__(self, attr, train_data):
        self.train_data = train_data
        self.root = bulid_tree(attr, train_data)

    def predict(self, test_line):
        pass


def getID3(attr, train_data, entropy):
    classes = set(map(lambda x: x[attr], train_data))
    size = len(train_data)
    entropy_attr = 0
    for clas in classes:
        class_size = len(list(filter(lambda x: x[attr] == clas, train_data)))
        class_right = len(list(filter(lambda x: x[attr] == clas and x[-1] == 1, train_data)))
        class_error = class_size - class_right
        if class_right != 0 and class_error != 0:
            entropy_attr += class_size / size * (
                -class_right / class_size * np.log2(class_right / class_size) - class_error / class_size * np.log2(
                    class_error / class_size))
        elif class_error == 0:
            entropy_attr += class_size / size * (-class_right / class_size * np.log2(class_right / class_size))
        elif class_right == 0:
            entropy_attr += class_size / size * (-class_error / class_size * np.log2(class_error / class_size))
    return attr, entropy - entropy_attr


def bulid_tree(root, attr, train_data):
    size = len(train_data)
    right = len(list(filter(lambda x: x[-1] == 1, train_data)))
    error = size - right
    if size == right:
        root["result"] = 1
        return
    elif size == error:
        root["result"] = 0
        return
    elif len(attr) == 0:
        if right > error:
            root["result"] = 1
        else:
            root["result"] = 0
    entropy = 0
    if right != 0 and error != 0:
        entropy = -right / size * np.log2(right / size) - error / size * np.log2(error / size)
    elif right == 0:
        entropy = -error / size * np.log2(error / size)
    elif error == 0:
        entropy = -right / size * np.log2(right / size)

    best_attr = (sorted(list(map(lambda x: getID3(x, train_data, entropy), attr)), key=lambda x: x[-1])[-1])[0]
    classes = set(map(lambda x: x[best_attr], train_data))
    attr_copy = list(attr).remove(best_attr)
    for clas in classes:
        root[clas] = dict()
        bulid_tree(root[clas], attr_copy, list(filter(lambda x: x[best_attr] == clas, train_data)))


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
            splits[0] = str(round(int(splits[0])/10))
            splits.remove(splits[2])
            splits.remove(splits[3])
            splits.remove(splits[7])
            train_data.append(splits)
        for i in range(divide, len(lines)):
            splits = list()
            splits.extend(lines[i].split(','))
            splits[0] = str(round(int(splits[0])/10))
            splits.remove(splits[2])
            splits.remove(splits[3])
            splits.remove(splits[7])
            test_data.append(splits)
    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = read_train(0.8, "./train.csv")
    forest = Forest(train_data)
    forest.bulid_forest(101, 10, 1000)
    print("Forest built up")
    forest.predict(test_data, 71)
