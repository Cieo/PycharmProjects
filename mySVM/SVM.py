import numpy as np
import random


class Sample:
    def __init__(self, x, y):
        self.x = x
        if y == 0:
            self.y = -1
        else:
            self.y = 1
        self.y_p = 0

    def __correct__(self):
        return np.sign(self.y) == np.sign(self.y_p)


def getE(ii, size, train_set, a):
    sum = 0
    for k in range(size):
        kk = train_set[k]
        sum += a[k] * kk.y * (kk.x.dot(ii.x))
    return sum - ii.y


def getN(ii, jj):
    return 2 * ii.x.dot(jj.x) - ii.x.dot(ii.x) - jj.x.dot(jj.x)


def getW(a, i):
    return a * i.y * i.x


def trainSVM(size, train_set):
    a = [0.01] * size
    b = 0
    e = 10 ** -5
    num_changed_alphas = 0
    tol = 0.1
    max_passes = 100
    passes = 0
    C = 1
    while (passes < max_passes):
        print(num_changed_alphas)
        num_changed_alphas = 0
        for i in range(size):
            E_i = getE(train_set[i], size, train_set, a)
            if (train_set[i].y * E_i < -tol and a[i] < C) or (train_set[i].y * E_i > tol and a[i] > 0):
                j = random.choice(range(size))
                while j == i:
                    j = random.choice(range(size))
                E_j = getE(train_set[j], size, train_set, a)
                a_i_old = a[i]
                a_j_old = a[j]
                L = 0
                H = 0
                if train_set[i].y == train_set[j].y:
                    L = np.maximum(0, a[j] - a[i])
                    H = np.minimum(C, C + a[j] - a[i])
                elif train_set[i].y != train_set[j].y:
                    L = np.maximum(0, a[j] + a[i] - C)
                    H = np.minimum(C, a[j] + a[i])
                if L == H:
                    continue
                n = getN(train_set[i], train_set[j])
                a_j_new_unc = np.nan_to_num(a[j] - train_set[j].y * (E_i - E_j) / n)
                a_j_new = 0
                if a_j_new_unc > H:
                    a_j_new = H
                elif L <= a_j_new_unc and a_j_new_unc <= H:
                    a_j_new = a_j_new_unc
                elif a_j_new_unc < L:
                    a_j_new = L
                if abs(a_j_new - a_j_old) < e:
                    continue
                a_i_new = a_i_old + train_set[i].y * train_set[j].y * (a_j_old - a_j_new)
                b1_new = b - E_i - train_set[i].y * (a_i_new - a_i_old) * (train_set[i].x.dot(train_set[i].x)) - \
                         train_set[j].y * (a_j_new - a_j_old) * (train_set[i].x.dot(train_set[j].x))
                b2_new = b - E_j - train_set[i].y * (a_i_new - a_i_old) * (train_set[i].x.dot(train_set[j].x)) - \
                         train_set[j].y * (a_j_new - a_j_old) * (train_set[j].x.dot(train_set[j].x))
                if 0 < a_i_new and a_i_new < C:
                    b = b1_new
                elif 0 < a_j_new and a_j_new < C:
                    b = b2_new
                else:
                    b = (b1_new + b2_new) / 2
                a[i] = a_i_new
                a[j] = a_j_new
                num_changed_alphas += 1

        if num_changed_alphas <= 1:
            passes += 1
        else:
            passes = 0
    w = 0
    for i in range(size):
        w += getW(a[i], train_set[i])
    return w, b


def predict(w, b, train_set):
    print("w = ", w)
    print("b = ", b)
    right = 0
    for i in train_set:
        i.y_p = i.x.dot(w) + b
        if i.__correct__():
            right += 1
        print("y_p = ", i.y_p)
    print("accuracy = ", right / len(train_set))


def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        lines.remove(lines[0])
        samples = list()
        for line in lines:
            splits = line.split(',')
            samples.append(Sample(np.array(list(float(x) for x in splits[:-1])), int(splits[-1])))
    return samples


if __name__ == "__main__":
    samples = list()
    samples.append(Sample(np.array([3, -1]), -1))
    samples.append(Sample(np.array([-1, 1]), 1))
    samples.append(Sample(np.array([1, 1]), 1))
    samples.append(Sample(np.array([-1, -1]), -1))
    # print(samples)
    train = read_file("./data/train.csv")
    test = read_file("./data/test.csv")
    w, b = trainSVM(len(train), train)
    predict(w, b, test)
