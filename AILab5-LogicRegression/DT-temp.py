import math


class Datacase:
    def __init__(self, x=list(), y=0):
        self.x = x
        self.y = y
        self.y_predict = -1

    def __predict__(self, root):
        if root.result != -1:
            self.y_predict = root.result
            return
        for node in root.nodes:
            if node.value == self.x[node.label]:
                self.__predict__(node)


    def __correct__(self):
        return self.y_predict == self.y


class Node:
    def __init__(self, label, value, dataset, depth):
        self.label = label
        self.value = value
        self.nodes = list()
        self.treeData = dataset
        self.result = -1
        self.depth = depth


def appendTree(root):
    correct = 0
    for datacase in root.treeData:
        correct += datacase.y
    if correct == 0:
        root.result = 0
        return
    elif correct == len(root.treeData):
        root.result = 1
        return
    elif root.depth == len(root.treeData[0].x):
        root.result = 1 if correct > (len(root.treeData) / 2) else 0
        return
    (bestLabel, dividedSet) = getBestFeature(root.treeData)
    for value in dividedSet.keys():
        newNode = Node(bestLabel, value, dividedSet[value], root.depth + 1)
        root.nodes.append(newNode)
        appendTree(root.nodes[-1])


def getBestFeature(dataset):
    bestLabel = 0
    bestGain = 0
    for label in range(0, len(dataset[0].x)):
        gain = getID3(dataset, label)
        if gain > bestGain:
            bestGain = gain
            bestLabel = label
    diviedSet = dict()
    for datacase in dataset:
        if datacase.x[bestLabel] not in diviedSet:
            diviedSet[datacase.x[bestLabel]] = list()
        diviedSet[datacase.x[bestLabel]].append(datacase)
    return (bestLabel, diviedSet)


def getID3(dataset, label):
    labelCount = dict()
    size = len(dataset)
    for datacase in dataset:
        if datacase.x[label] not in labelCount:
            labelCount[datacase.x[label]] = list()
        labelCount[datacase.x[label]].append(datacase)
    HD = getEntropy(dataset)
    HDA = 0
    for label in labelCount.values():
        HDA += len(label) / size * getEntropy(label)
    return HD - HDA


def getEntropy(dataset):
    size = len(dataset)
    datasetCorrect = 0
    datasetError = 0
    for datacase in dataset:
        if datacase.y == 1:
            datasetCorrect += 1
        else:
            datasetError += 1
    if datasetError == 0:
        return (-datasetCorrect / size) * math.log2(datasetCorrect / size)
    elif datasetCorrect == 0:
        return (-datasetError / size) * math.log2(datasetError / size)

    return (-datasetCorrect / size) * math.log2(datasetCorrect / size) + (-datasetError / size) * math.log2(
        datasetError / size)


def predictAll(dataset, root):
    for dataCase in dataset:
        dataCase.__predict__(root)


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

    example = list()
    example.append(Datacase([1, 3, 0, 1], 0))
    example.append(Datacase([1, 3, 0, 2], 0))
    example.append(Datacase([2, 3, 0, 1], 1))
    example.append(Datacase([3, 2, 0, 1], 1))
    example.append(Datacase([3, 1, 1, 1], 1))
    example.append(Datacase([3, 1, 1, 2], 0))
    example.append(Datacase([2, 1, 1, 2], 1))
    example.append(Datacase([1, 2, 0, 1], 0))
    example.append(Datacase([1, 1, 1, 1], 1))
    example.append(Datacase([3, 2, 1, 1], 1))
    example.append(Datacase([1, 2, 1, 2], 1))
    example.append(Datacase([2, 2, 0, 2], 1))
    example.append(Datacase([2, 3, 1, 1], 1))
    example.append(Datacase([3, 2, 0, 1], 0))
    root = Node(-1, -1, trainset, 0)
    appendTree(root)
    predictAll(testset,root)
    print(ratio(testset))
