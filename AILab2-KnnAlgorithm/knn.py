import math


# Class to record the emotion for both Classification and Regression
class Emotion:
    def __init__(self):
        # primaryEmotion is the emotion for Classification
        self.primaryEmotion = ""

        # emotions is the probability for every emotion in Regression
        self.emotions = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sad": 0, "surprise": 0}


class Word:
    def __init__(self, v, e="", es=["0", "0", "0", "0", "0", "0"], id1="text0"):

        """
        Function to initial the Word Class
        :param v: vector of this text in one hot
        :param e: the emotion of this text get from the files
        :param es: the emotions probability get from the files
        :param id1: the id of text get from the files
        """
        self.id = id1
        self.emotion = Emotion()
        self.emotion.primaryEmotion = e
        self.emotion.emotions = {"anger": float(es[0]), "disgust": float(es[1]), "fear": float(es[2]),
                                 "joy": float(es[3]), "sad": float(es[4]),
                                 "surprise": float(es[5])}

        self.vector = v

        # op is the list to drop the word that is too short
        # to optimize result
        op = list()
        for element in self.vector:
            if len(element) < 3:
                op.append(element)

        for element in op:
            self.vector.remove(element)

        # knnResult is the emotion class that get from the knn algorithm
        self.knnResult = Emotion()
        self.distance = 0

    def getKNN(self, trainSet, method, k, problem):

        """
        Function to get the knn-result of this text base on the train set
        :param trainSet: the train set get from the files , a list of Word class
        :param method: the method of computing distance , either 'Manhattan' or 'Euclidean'
        :param k: the number k of the K-NN algorithm
        :param problem: the problem we are going to solve , change algorithm base on the problem you input ,
                        either 'Classification' of 'Regression'
        :return: no return
        """

        # change the method of computing distance and then get the distance
        if method == "Manhattan":
            for word in trainSet:
                word.distance = len(word.vector) + len(self.vector) - 2 * len(word.vector & self.vector)
        elif method == "Euclidean":
            for word in trainSet:
                word.distance = math.sqrt(len(word.vector) + len(self.vector) - 2 * len(word.vector & self.vector))
        else:
            for word in trainSet:
                word.distance = len(word.vector & self.vector) / (len(self.vector) * len(word.vector))

        # change the problem to select different knn algorithm
        if problem == "Classification":
            # dictionary to count the emotion in the former K emotions
            result = {"sad": 0, "happy": 0, "fear": 0, "surprise": 0, "disgust": 0, "anger": 0}

            # select the former K emotions and add right to every emotion base on their quantity
            for word in (sorted(trainSet, key=lambda Word: Word.distance))[:k]:
                result[word.emotion.primaryEmotion] += (1 / word.distance)

            # get the result of Knn , judge if there is more than one emotion that has same quantity
            sortedResult = (sorted(result.items(), key=lambda d: d[1]))
            if sortedResult[-1][1] != sortedResult[-2][1]:
                self.knnResult.primaryEmotion = (sorted(result.items(), key=lambda d: d[1]))[-1][0]
            else:
                self.knnResult.primaryEmotion = "unknown"

        # algorithm for Regression , select the former K emotion and make a weighting
        else:
            for word in (sorted(trainSet, key=lambda Word: Word.distance))[:k]:
                for key in word.emotion.emotions:
                    self.knnResult.emotions[key] += (word.emotion.emotions[key] / word.distance)
            total = sum(self.knnResult.emotions.values())
            for key in self.knnResult.emotions:
                self.knnResult.emotions[key] /= total


def divideWord(filePath, problem):
    """
    Function to read text from the input files
    :param filePath: the path of the text files
    :param problem: the problem we are going to solve, either 'Classification' or 'Regression',
                    change the format of reading text base on the problem.
    :return: the list of Word class get from the files
    """
    countWordNumber = list()
    wordsVector = list()
    with open(filePath) as f:
        lines = f.readlines()[1:]
        if problem == "Classification":
            for line in lines:
                tempWords = line.split()[2:]
                countWordNumber.extend(tempWords[1:])
                wordsVector.append(Word(set(tempWords[1:]), tempWords[0]))
        else:
            for line in lines:
                tempWords = line.replace('\n', '').replace('？', '0').split(',')
                wordsVector.append(Word(set(tempWords[1].split()), es=tempWords[-6:], id1=tempWords[0]))
        print(len(set(countWordNumber)))
    return wordsVector


def getAllKnn(testWords, trainWords, method, k, problem):
    """
    Function to get all the knn result of the test set , base on the getKnn methon of the Word class
    :param testWords: the test set you are going to test
    :param trainWords: the train set you used to build the model
    :param method: the way of judging distance
    :param k: the k of K-NN algorithm
    :param problem: the problem we are going to solve, either 'Classification or 'Regression'
    :return: no return, print the result of Knn
    """

    if problem == "Classification":
        # variables to record the result of knn
        total = len(testWords)
        correct = 0
        error = 0
        unknown = 0
        # count and print the result
        for testWord in testWords:
            testWord.getKNN(trainWords, method, k, problem)
            print(testWord.knnResult.primaryEmotion)
            if testWord.knnResult.primaryEmotion == "happy":
                print("3")
            elif testWord.knnResult.primaryEmotion == "sad":
                print("2")
            elif testWord.knnResult.primaryEmotion == "anger":
                print("1")

            if testWord.emotion.primaryEmotion == testWord.knnResult.primaryEmotion:
                correct += 1
            elif testWord.knnResult.primaryEmotion == "unknown":
                unknown += 1
            else:
                error += 1
        print("K = %s , correct = %s , error = %s , unknown = %s , total = %s , rate = %s%%" % (
            k, correct, error, unknown, total, (correct / total) * 100))

    else:
        # write the result into the files
        f = open("./RResult" + str(k) + ".txt", "w")
        for testWord in testWords:
            testWord.getKNN(trainWords, method, k, problem)
            f.write(testWord.id + " ")
            f.write(str(testWord.knnResult.emotions["anger"]))
            f.write(" " + str(testWord.knnResult.emotions["disgust"]))
            f.write(" " + str(testWord.knnResult.emotions["fear"]))
            f.write(" " + str(testWord.knnResult.emotions["joy"]))
            f.write(" " + str(testWord.knnResult.emotions["sad"]))
            f.write(" " + str(testWord.knnResult.emotions["surprise"]) + "\n")
        print(k, "done")
        f.close()


if __name__ == "__main__":

    # read the files
    wordsVectorTrainC = divideWord("./Classification/train.txt", "Classification")
    wordsVectorTestC = divideWord("./Classification/test.txt", "Classification")
    # wordsVectorTrainR = divideWord("./Regression/Dataset_train.csv", "Regression")
    # wordsVectorTestR = divideWord("./Regression/Dataset_test.csv", "Regression")
    # get the result for different k
    # for i in range(1, 64):
    getAllKnn(wordsVectorTestC, wordsVectorTrainC, "Euclidean", 1, "Classification")
        # getAllKnn(wordsVectorTestR, wordsVectorTrainR, "Euclidean", i, "Regression")
        #     best RR k in Euclidenan = 16
