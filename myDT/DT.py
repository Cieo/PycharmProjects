import numpy as np
import math
import random
import multiprocessing


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
        correct = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for test_line in test_data:
            result = list(map(lambda x: x.predict(test_line), self.trees))
            right = len(list(filter(lambda x: x == 1, result)))
            error = len(list(filter(lambda x: x == 0, result)))
            if right > self.tree_num * 0.4 and test_line[-1] == 1:
                correct += 1
                TP += 1
            elif error > self.tree_num * 0.6 and test_line[-1] == 0:
                correct += 1
                TN += 1
            elif right > self.tree_num * 0.4 and test_line[-1] == 0:
                FP += 1
            elif error > self.tree_num * 0.6 and test_line[-1] == 1:
                FN += 1
        # print(correct / len(test_data))
        print(2 * TP / (2 * TP + FP + FN))


class Tree:
    def __init__(self, attr, train_data):
        self.train_data = train_data
        self.root = dict()
        self.attr = attr
        bulid_tree(self.root, attr, train_data)

    def predict(self, test_line):
        root = self.root
        classes = set(test_line[:-1])
        while "result" not in root:
            key = list(set(root.keys()) & classes)
            if len(key) == 0:
                return -1
            root = root[key[0]]
        return root["result"]


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
        return

    entropy = 0
    if right != 0 and error != 0:
        entropy = -right / size * np.log2(right / size) - error / size * np.log2(error / size)
    elif right == 0:
        entropy = -error / size * np.log2(error / size)
    elif error == 0:
        entropy = -right / size * np.log2(right / size)

    best_attr = (sorted(list(map(lambda x: getID3(x, train_data, entropy), attr)), key=lambda x: x[-1])[-1])[0]
    classes = set(map(lambda x: x[best_attr], train_data))
    attr_copy = list(attr)
    attr_copy.remove(best_attr)
    for clas in classes:
        root[clas] = dict()
        bulid_tree(root[clas], attr_copy, list(filter(lambda x: x[best_attr] == clas, train_data)))


def read_train(p, path):
    with open(path) as f:
        lines = f.readlines()
        lines.remove(lines[0])
        divide = math.floor(len(lines) * p)
        data = list()
        for i in range(len(lines)):
            splits = list()
            splits.extend(lines[i].replace('\n', '').split(','))
            splits[0] = str(round(int(splits[0]) / 5)) + "age"
            splits[10] = str(round(int(splits[10]) / 1000)) + "gain"
            splits[11] = str(round(int(splits[11]) / 1000)) + "loss"
            splits[12] = str(round(int(splits[12]) / 10)) + "hour"
            splits[-1] = int(splits[-1])
            splits.remove(splits[2])
            splits.remove(splits[2])
            splits.remove(splits[5])
            data.append(splits)
        train_data = random.sample(data, divide)
        for line in train_data:
            data.remove(line)
        test_data = data

    return train_data, test_data


def run_forest(i):
    print("test", i)
    train_data, test_data = read_train(0.9, "./train.csv")
    print("File read finish!", i)
    forest = Forest(train_data)
    forest.bulid_forest(67, 9, 10000)
    print("Forest built up!", i)
    forest.predict(test_data)
    print("Predict finish!", i)


if __name__ == "__main__":
    pool = multiprocessing.Pool()
    for i in range(5):
        pool.apply_async(run_forest, (i,))
    pool.close()
    pool.join()
    print()
    print("Finish!")
