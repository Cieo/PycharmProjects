import math


class Emotion:
    def __init__(self):
        """
        primaryEmotion : emotion of classification
        emotions : emotions of regression
        """
        self.primaryEmotion = ""
        self.emotions = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sad": 0, "surprise": 0}


class Structure:
    def __init__(self):
        """
        totalArticles : number of articles
        emotionArticles : number of articles with certain emotion
        emotionLength : number of words with certain emotion
        emotionLengthNonRepeat : number of non-repeat words with certain words
        wordList : dict that record information of every word (like the number of appearence in certain emotion)
        """
        self.totalArticles = 0
        self.emotionArticles = {"anger": list(), "disgust": list(), "fear": list(), "joy": list(), "sad": list(),
                                "surprise": list()}
        self.emotionLength = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sad": 0, "surprise": 0}
        self.emotionLengthNonRepeat = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sad": 0, "surprise": 0}
        self.wordList = dict()


class Word:
    def __init__(self, text, emotionArticles):
        """
        :param text : text of the word
        :param emotionArticles : articles list divided by emotion label

        emotionTimes : the number of appearence in certain emotion , inital with 0.0001 (applying the laplace smoothing)
        """
        self.text = text
        self.emotionTimes = {"anger": 1, "disgust": 1, "fear": 1, "joy": 1, "sad": 1,
                             "surprise": 1}
        for key in emotionArticles.keys():
            for article in emotionArticles[key]:
                self.emotionTimes[key] += article.count(text)


class Article:
    def __init__(self, v, e="", es=["0", "0", "0", "0", "0", "0"], id1="0"):
        """
        :param v : list of words in article
        :param e : emotion for classification
        :param es : emotions for regression
        :param id1 : text id

        emotion : emotion information get from article
        nbResult : emotion information get from Naive Bayes
        """
        self.id = id1
        self.emotion = Emotion()
        self.emotion.primaryEmotion = e
        self.emotion.emotions = {"anger": float(es[0]), "disgust": float(es[1]), "fear": float(es[2]),
                                 "joy": float(es[3]), "sad": float(es[4]),
                                 "surprise": float(es[5])}
        self.vector = v
        self.nbResult = Emotion()

    def getBayesClassification(self, structure, newWordLength):
        """
        :param structure : data from train set

        articleP : dict to record the probability of every emotion
        keys : word list of all words in train set
        """
        articleP = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sad": 0, "surprise": 0}
        keys = structure.wordList.keys()
        for key in articleP.keys():
            # initial probability of a emotion
            articleP[key] = len(structure.emotionArticles[key]) / structure.totalArticles
            print("initial " + key + " = " + str(articleP[key]))
            for word in self.vector:
                # probability of emotion with the appearence of word
                if keys.__contains__(word):
                    articleP[key] *= structure.wordList[word].emotionTimes[key] / (
                        structure.emotionLength[key] + structure.emotionLengthNonRepeat[key] + newWordLength[key])
                    # if structure.wordList[word].emotionTimes[key] != 0.0001:
                        # print(key + " = " + key + " * " + str(structure.wordList[word].emotionTimes[key] / (
                        #     structure.emotionLength[key] + structure.emotionLengthNonRepeat[key])) + " = " + str(articleP[key]))
                    # else:
                    #     print(key + " = " + key + " * " + str(structure.wordList[word].emotionTimes[key] / (
                    #         structure.emotionLength[key] + structure.emotionLengthNonRepeat[key])) + " = " + str(
                    #         articleP[key]) + "       smooth")
                else:
                    articleP[key] *= 1 / (structure.emotionLength[key] + structure.emotionLengthNonRepeat[key] + newWordLength[key])
                    # print(key + " = " + key +" * "+ str(0.0001 / (structure.emotionLength[key] + structure.emotionLengthNonRepeat[key])) +" = "+str(articleP[key])+"       smooth")
        print("-------divider-------")
        sortedArticleP = sorted(articleP.items(), key=lambda d: d[1])
        self.nbResult.primaryEmotion = sortedArticleP[-1][0]

    def getBayesRegression(self, trainset):
        """
        :param trainset : data from train set

        total : sum of probability of all emotions (used when normalizing)
        gain : temporary variable recording probability get from one article
        """
        total = 0
        for article in trainset:
            for key in article.emotion.emotions.keys():
                # compute probability gain from one article
                gain = article.emotion.emotions[key]
                for word in set(self.vector):
                    gain *= (article.vector.count(word) + 1) / (len(article.vector) + len(set(self.vector)))
                self.nbResult.emotions[key] += gain
        # normalize the probability
        for key in self.nbResult.emotions.keys():
            total += self.nbResult.emotions[key]
        if total != 0:
            for key in self.nbResult.emotions.keys():
                self.nbResult.emotions[key] /= total


def divideWord(filePath, problem):
    countWordNumber = list()
    wordsVector = list()
    with open(filePath) as f:
        lines = f.readlines()[1:]
        if problem == "Classification":
            for line in lines:
                tempWords = line.split()[2:]
                countWordNumber.extend(tempWords[1:])
                wordsVector.append(Article(tempWords[1:], tempWords[0]))
        else:
            for line in lines:
                tempWords = line.replace('\n', '').replace('？', '0').split(',')
                wordsVector.append(Article(tempWords[1].split(), es=tempWords[-6:], id1=tempWords[0]))
    return wordsVector


def getTrainSetStructure(trainSet):
    structure = Structure()
    wordList = set()
    structure.totalArticles = len(trainSet)
    for article in trainSet:
        structure.emotionArticles[article.emotion.primaryEmotion].append(article.vector)
        structure.emotionLength[article.emotion.primaryEmotion] += len(article.vector)
        wordList |= set(article.vector)
    for word in wordList:
        structure.wordList[word] = Word(word, structure.emotionArticles)
    return structure


def optimaze(trainSet, testSet):
    """
    function to remove the word in train set that didn't show up in test set

    :param trainSet : data from train set
    :param testSet : data from test set
    """
    wordList = list()
    for artical in testSet:
        wordList.extend(artical.vector)
    for artical in trainSet:
        for word in artical.vector:
            if wordList.count(word) == 0:
                artical.vector.remove(word)

def getNewWordsLength(trainSet,testSet):
    newWordsLength = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sad": 0,
                                "surprise": 0}
    trainSetLength = {"anger": set(), "disgust": set(), "fear": set(), "joy": set(), "sad": set(),
                                "surprise": set()}
    testSetLength  = {"anger": set(), "disgust": set(), "fear": set(), "joy": set(), "sad": set(),
                                "surprise": set()}
    for artical in trainSet:
        trainSetLength[artical.emotion.primaryEmotion] |= set(artical.vector)
    for artical in testSet:
        testSetLength[artical.emotion.primaryEmotion] |= set(artical.vector)
    for key in newWordsLength.keys():
        newWordsLength[key] = len(trainSetLength[key] | testSetLength[key]) - len(trainSetLength[key])
    return newWordsLength


if __name__ == "__main__":
    # read the files
    wordsVectorTrainC = divideWord("./Classification/train.txt", "Classification")
    wordsVectorTestC = divideWord("./Classification/test.txt", "Classification")
    optimaze(wordsVectorTrainC, wordsVectorTestC)
    trainStructureC = getTrainSetStructure(wordsVectorTrainC)
    wordsVectorTrainR = divideWord("./Regression/Dataset_train.csv", "Regression")
    wordsVectorTestR = divideWord("./Regression/Dataset_test.csv", "Regression")

    total = 1000
    right = 0
    for artical in wordsVectorTestC:
        artical.getBayesClassification(trainStructureC)
        if artical.emotion.primaryEmotion == artical.nbResult.primaryEmotion:
            right += 1

    for artical in wordsVectorTestR:
        artical.getBayesRegression(wordsVectorTrainR)

    with open("./RResult" + ".csv", "w") as f:
        for artical in wordsVectorTestR:
            f.write(str(artical.nbResult.emotions["anger"]))
            f.write("," + str(artical.nbResult.emotions["disgust"]))
            f.write("," + str(artical.nbResult.emotions["fear"]))
            f.write("," + str(artical.nbResult.emotions["joy"]))
            f.write("," + str(artical.nbResult.emotions["sad"]))
            f.write("," + str(artical.nbResult.emotions["surprise"]) + "\n")
    print((right / total)*100)
