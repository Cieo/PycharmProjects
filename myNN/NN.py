import random

import numpy as np
import math


def read_train(p,path):
    with open(path) as f:
        lines = f.readlines()
        lines.remove(lines[0])
        divide = math.floor(len(lines) * p)
        x_train = list()
        y_train = list()
        x_test = list()
        y_test = list()
        for i in range(divide):
            splits = [1]
            splits.extend(lines[i].split(','))
            x_train.append(list(float(a) for a in splits[:-1]))
            y_train.append([float(splits[-1])])
        for i in range(divide, len(lines)):
            splits = [1]
            splits.extend(lines[i].split(','))
            x_test.append(list(float(a) for a in splits[:-1]))
            y_test.append([float(splits[-1])])
    return x_train, y_train, x_test, y_test


def nonlin(x, deriv=False):
    if deriv:
        return 1-x**2
    return np.tanh(x)


def train_nn(x, y):
    x = x
    y = y

    np.random.seed(1)
    wrange = 0.2
    # randomly initialize our weights with mean 0
    syn0 = wrange * np.random.random((10, 5)) - wrange / 2
    syn1 = wrange * np.random.random((5, 1)) - wrange / 2

    for j in range(2000):
        choice = [random.randint(0, len(x)-1) for i in range(10)]
        xs = np.array([x[i] for i in choice])
        ys = np.array([y[i] for i in choice])
        for k in range(100):
            l0 = xs
            # Feed forward through layers 0, 1, and 2
            l1 = nonlin(np.dot(l0, syn0))
            l2 = nonlin(np.dot(l1, syn1))

            # how much did we miss the target value?
            l2_error = ys - l2

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            l2_delta = l2_error * nonlin(l2, deriv=True)
            # how much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(syn1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            l1_delta = l1_error * nonlin(l1, deriv=True)

            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)
            print(l2_error)
    return syn0, syn1


def predict(syn0, syn1, x, y):
    x = np.array(x)
    y = np.array(y)

    np.random.seed(1)

    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    one = 0
    zero = 0
    print(l2)
    for i, j in zip(l2, y):
        if i > 0.5 and j == 1:
            one += 1
        elif i <= 0.5 and j == 0:
            zero += 1
    print(zero, one, len(y))
    print((zero + one) / len(y))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_train(1,"./data/train.csv")
    x_train1, y_train1, x_test1, y_test1 = read_train(1,"./data/test.csv")
    syn0, syn1 = train_nn(x_train, y_train)
    predict(syn0, syn1, x_train1, y_train1)
