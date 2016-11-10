from random import choice


def accumulation(a, b):
    sum = 0

    if len(a) != len(b):
        print("Error : 'a' and 'b' not in same dimension")
        return 1

    for (ai, bi) in zip(a, b):
        sum += ai * bi

    return sum


class Datacase:
    def __init__(self, x=list(), y=0):
        self.x = [1]
        self.x.extend(x)
        self.y = y
        self.y_predict = 0

    def __predict__(self, w):
        self.y_predict = 1 if accumulation(self.x, w) > 0 else -1

    def __correct__(self):
        return self.y_predict == self.y

    def __update__(self, w):
        newW = list()
        for (wi, xi) in zip(w, self.x):
            wi += xi * self.y
            newW.append(wi)
        return newW


def trainWLimit(dataset, limit):

    """
    :param dataset : trainset used to train w
    :param limit : time limit of iteration
    :return: trained w and the accuracy
    """


    w = [0] * len(dataset[0].x)
    i = 0
    while i < len(dataset) and limit > 0:
        limit -= 1
        dataset[i].__predict__(w)
        if dataset[i].__correct__():
            # train w with next sample
            i += 1
        else:
            # train w with this sample
            # and go back to the first sample to train again
            w = dataset[i].__update__(w)
            i = 0

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
        elif dataCase.y == 1 and dataCase.y_predict == -1:
            base[1] += 1
        elif dataCase.y == -1 and dataCase.y_predict == 1:
            base[2] += 1
        elif dataCase.y == -1 and dataCase.y_predict == -1:
            base[3] += 1
    # Accuracy Recall Precision F1
    result = [0] * 4
    result[0] = (base[0] + base[-1]) / (sum(base))
    result[1] = (base[0]) / (base[0] + base[1]) if (base[0] + base[1]) != 0 else 0
    result[2] = (base[0]) / (base[0] + base[2]) if (base[0] + base[2]) != 0 else 0
    result[3] = (2 * result[2] * result[1]) / (result[2] * result[1]) if (base[2] * base[1]) != 0 else 0
    return result


def trainWPocket(dataset, limit):

    """
    :param dataset : trainset used to train w
    :param limit : time limit of iteration
    :return: trained w and the accuracy
    """

    # init the variables for pocket
    bestW = [0] * len(dataset[0].x)
    bestResult = [0] * 4
    w = bestW
    predictAll(dataset, bestW)
    result = bestResult

    while limit > 0:
        # copy the current dataset
        # because we will change the dataset later
        datasetCopy = dataset
        for dataCase in datasetCopy:
            if ~dataCase.__correct__():
                # update w and check the accuracy of the new w
                limit -= 1
                w = dataCase.__update__(w)
                predictAll(dataset, w)
                result = ratio(dataset)
                # update pocket if new w have higher accuracy
                if result[0] > bestResult[0]:
                    bestW = w
                    bestResult = result
                    print(bestResult[0])
                    break
                # roll back if not
                else:
                    w = bestW
        # get out of loop if we can't find a better w
        if result[0] < bestResult[0]:
            print("陷入局部最优")
            print(bestW)
            break

    return (bestW, bestResult)


def readFile(dataPath, labelPath):
    data = open(dataPath, "r")
    label = open(labelPath, "r")
    dataLines = data.readlines()
    labelLines = label.readlines()
    dataset = list()
    for (dataLine, labelLine) in zip(dataLines, labelLines):
        dataset.append(Datacase([int(x) for x in dataLine.split()], [int(x) for x in labelLine.split()][0]))
    return dataset


if __name__ == "__main__":
    trainset = readFile("./train_data.txt", "./train_labels.txt")
    testset = readFile("./test_data.txt", "./test_labels.txt")
    (w1, result1) = trainWPocket(trainset, 15000)
    (w2, result2) = trainWLimit(trainset, 15000)
    print("trainset: Pocket & Limited")
    print(result1, result2)

    predictAll(testset, w1)
    result3 = ratio(testset)
    predictAll(testset, w2)
    result4 = ratio(testset)
    print("testset: Pocket & Limited")
    print(result3, result4,"\n")
    print("Pocket: accuracy:",result3[0]," precision:",result3[2], " recall:",result3[1]," F-1:",result3[3])
    print("init: accuracy:",result4[0]," precision:",result4[2], " recall:",result4[1]," F-1:",result4[3])
