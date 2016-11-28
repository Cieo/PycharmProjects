import math

class Datacase:
    def __init__(self, x=list(), y=0):
        self.x = x
        self.y = y
        self.y_predict = -1

    def __predict__(self, root):
        # if we came to the leaf, set the result of this data_case
        if root.result != -1:
            self.y_predict = root.result
            return
        # else keep going down the tree
        for node in root.nodes:
            if node.value == self.x[node.label]:
                self.__predict__(node)
                return
        # if we found that there is no branch for this data_case, choose
        # the branch that is closest to it
        bestError = 999
        bestNode = 999
        for node in root.nodes:
            if bestError > abs(node.value - self.x[node.label]):
                bestError = abs(node.value - self.x[node.label])
                bestNode = node
        self.__predict__(bestNode)

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

# count the number of leaves of a tree
def countTree(root):
    if root.result != -1:
        return 1
    count = 0
    for node in root.nodes:
        count += countTree(node)
    return count


def appendTree(root):
    correct = 0
    # judge if we have come to the leaf
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
    # get the best label and the divided set then append the tree
    (bestLabel, dividedSet) = getBestFeature(root.treeData)
    # append new node to current root
    for value in dividedSet.keys():
        newNode = Node(bestLabel, value, dividedSet[value], root.depth + 1)
        root.nodes.append(newNode)
        appendTree(root.nodes[-1])

# found the label that get biggest gain
def getBestFeature(dataset):
    bestLabel = 0
    bestGain = 0
    # compute the gain of each label and choose the best label
    for label in range(0, len(dataset[0].x)):
        gain = getID3(dataset, label)
        if gain > bestGain:
            bestGain = gain
            bestLabel = label
    diviedSet = dict()
    # divide the dataset
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

# function to get the 1 result of a data_case
def right(datacase):
    return datacase.y

# function to get the 0 result of a data_case
def wrong(datacase):
    return 1 if datacase.y != 1 else 0


def getGini(dataset, label):
    labelCount = dict()
    size = len(dataset)
    for datacase in dataset:
        if datacase.x[label] not in labelCount:
            labelCount[datacase.x[label]] = list()
        labelCount[datacase.x[label]].append(datacase)
    gini = 0
    for label in labelCount.values():
        gini += (len(label) / size) * (
            1 - (sum(map(right, label)) / len(label)) ** 2 - (sum(map(wrong, label)) / len(label)) ** 2)
    return gini


def getC45(dataset, label):
    labelCount = dict()
    size = len(dataset)
    for datacase in dataset:
        if datacase.x[label] not in labelCount:
            labelCount[datacase.x[label]] = list()
        labelCount[datacase.x[label]].append(datacase)
    HD = getEntropy(dataset)
    HDA = 0
    splitinfo = 0
    for label in labelCount.values():
        HDA += len(label) / size * getEntropy(label)
        splitinfo += (-len(label) / size) * math.log2(len(label) / size)
    gain = HD - HDA
    return gain / splitinfo if splitinfo != 0 else 0


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

# analyse the result
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
    result[3] = (2 * result[2] * result[1]) / (result[2] + result[1]) if (base[2] + base[1]) != 0 else 0
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

    root = Node(-1, -1, trainset, 0)
    appendTree(root)
    print(countTree(root))
    predictAll(testset, root)
    print(ratio(testset))
