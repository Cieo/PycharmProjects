import math


def accumulation(a, b):
    sum = 0

    if len(a) != len(b):
        print("Error : 'a' and 'b' not in same dimension")
        return 1

    for (ai, bi) in zip(a, b):
        sum += ai * bi

    return sum


def getP(x, w):
    s = accumulation(x, w)
    P = 1 / (1 + math.e ** -s)
    return P


class Datacase:
    def __init__(self, x=list(), y=0):
        self.x = [1]
        self.x.extend(x)
        self.y = y
        self.y_predict = -1

    def __predict__(self, w):
        self.y_predict = 1 if getP(self.x, w) > 0.5 else 0

    def __correct__(self):
        return self.y_predict == self.y


def updateW(dataset, w, n):
    newW = [0] * len(w)
    for datacase in dataset:
        for i in range(0, len(w)):
            newW[i] += (getP(datacase.x, w) - datacase.y) * datacase.x[i]
    for i in range(0, len(w)):
        newW[i] *= -n
        newW[i] += w[i]
    return newW


def trainWLimit(dataset, limit, n):
    """
    :param dataset : trainset used to train w
    :param limit : time limit of iteration
    :return: trained w and the accuracy
    """

    w = [0] * len(dataset[0].x)
    while limit > 0:
        limit -= 1
        w = updateW(dataset, w, n)
    predictAll(dataset, w)
    return (w, ratio(dataset))


def predictAll(dataset, w):
    for dataCase in dataset:
        dataCase.__predict__(w)


def ratio(dataset):
    # TP FN FP TN
    base = [0] * 4
    for dataCase in dataset:
        if dataCase.y == 1 and dataCase.y_predict == 1:
            base[0] += 1
        elif dataCase.y == 1 and dataCase.y_predict == 0:
            base[1] += 1
        elif dataCase.y == 0 and dataCase.y_predict == 1:
            base[2] += 1
        elif dataCase.y == 0 and dataCase.y_predict == 0:
            base[3] += 1
    # Accuracy Recall Precision F1
    result = [0] * 4
    result[0] = (base[0] + base[-1]) / (sum(base))
    result[1] = (base[0]) / (base[0] + base[1]) if (base[0] + base[1]) != 0 else 0
    result[2] = (base[0]) / (base[0] + base[2]) if (base[0] + base[2]) != 0 else 0
    result[3] = (2 * result[2] * result[1]) / (result[2] * result[1]) if (base[2] * base[1]) != 0 else 0
    return result


def readFile(dataPath):
    data = open(dataPath, "r")
    dataLines = data.readlines()
    dataset = list()
    for dataLine in dataLines:
        dataset.append(Datacase([int(x) for x in dataLine.split(",")[0:-2]], [int(x) for x in dataLine.split(",")][-1]))
    return dataset


if __name__ == "__main__":
    trainset = readFile("./data/train.csv")
    testset = readFile("./data/test.csv")
    for i in range(1,100):
        (w, result) = trainWLimit(trainset, i*10, 1/len(trainset))
        print(str(i*10)+"-----------------------")
        print(w)
        print(result)
        print("result-------------------------")
        predictAll(testset, w)
        print(ratio(testset))
