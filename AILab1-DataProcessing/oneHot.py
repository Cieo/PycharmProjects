import math

# Open the file to write the answer
oneHot = open("C:/Users/Cieo233/Desktop/onehot","w+")
TF = open("C:/Users/Cieo233/Desktop/TF","w+")
TFIDF = open("C:/Users/Cieo233/Desktop/TFIDF","w+")
sMatrix = open("C:/Users/Cieo233/Desktop/smatrix","w+")
APlusBf = open("C:/Users/Cieo233/Desktop/APlusB","w+")


with open("C:/Users/Cieo233/Desktop/semeval","r") as f:

    # words: list to store all the word in the text (including the repeated words)
    words = list()
    # IDF: dict to store the number of files that contain certain words (non-repeated)
    IDF = dict()
    # lines: list of all the text in the semeval file
    lines = f.readlines()

    # use a loop to split and get all the word in the text
    for line in lines:
        words.extend(line.split("\t").__getitem__(2).replace("\n","").split(" "))

    # use a set to distinct the words and then recover the order
    orderedWords = list(set(words))
    orderedWords.sort(key=words.index)

    # initial the IDF for every word and write the title of one-hot
    for word in orderedWords:
        IDF[word] = 0
        oneHot.write(word + " ")
    oneHot.write("\n")

    # lineNum: md of sparse matrix
    # wordNum: nd of sparse matrix
    # totalWordNum: td of sparse matrix
    lineNum = len(lines)
    wordNum = len(orderedWords)
    totalWordNum = len(words)

    # write md & nd & td into the file
    sMatrix.write(str(lineNum) + "\n" + str(wordNum) + "\n" + str(totalWordNum) + "\n")

    # TFList: list to store the TF of all file
    TFList = list()
    # TFListLine: temp list to store the TF of every file
    TFListLine = list()

    # A: sparse matrix of first 623 files
    A = dict()
    # B: sparse matrix of later 623 files
    B = list()

    for line in lines:

        # get the i of sparse matrix
        i = lines.index(line)

        # get the denominator of TF
        line = line.split("\t").__getitem__(2).replace("\n","").split(" ")
        Enkj = len(line)

        ### SLOVE FOR TF AND IDF ###

        # get the molecular of TF and count the number of files for every word in the same time
        for word in orderedWords:
            # for TF
            nij = line.count(word)

            # for IDF
            if nij != 0:
                IDF[word] += 1

            # for TF
            tfij = nij/Enkj
            TFListLine.append(tfij)
            TF.write(str(tfij)+" ")

        TF.write("\n")

        # restore the TF of single file into the total list for the TF-IDF later
        TFList.append(list(TFListLine))
        TFListLine.clear()

        ### SOLVE FOR ONE-HOT AND SPARSE MATRIXX ###

        # initial the result of one-hot matrix of every file
        tempLine = wordNum * "0"

        # get the one-hot matrix of every file
        for word in line:

            # replace 0 with 1 in the one-hot matrix
            place = orderedWords.index(word)
            tempLine = tempLine[:place] + "1" + tempLine[place+1:]

            # get the j for sparse matrix
            j = place
            # output the sparse matrix for single word
            sMatrix.write(str(i)+"\t"+str(j)+"\t"+"1"+"\n")

            # cut the sparse matrix for A & B
            if i <= 622:
                A[str(i)+","+str(j)] = 1
            else:
                B.append(list([i-623,j,1]))

        # out put the one-hot matrix for single file
        oneHot.write(tempLine+"\n")

    ### SOLVE FOR TF-IDF ###

    # compute the TF-IDF
    for TFListLine in TFList:
        for word in orderedWords:
            # i: the index of current word in the non-repeated word list, to get the TF of word
            i = orderedWords.index(word)
            TFIDF.write(str(math.log(lineNum/(IDF[word]+1))*TFListLine[i])+" ")
        TFIDF.write("\n")

    ### SOLVE FOR A + B

    # compute the sparse matrix addition A + B
    # merge A & B into dict A
    for Bitem in B:
        if A.__contains__(str(Bitem[0])+","+str(Bitem[1])):
            A[str(Bitem[0])+","+str(Bitem[1])] += Bitem[2]
        else:
            A[str(Bitem[0])+","+str(Bitem[1])] = Bitem[2]

    # write the title of A + B
    APlusBf.write("622\n"+str(wordNum)+"\n"+str(len(A))+"\n")

    # APlusB: list to store the merge result of A + B in dict A
    APlusB = list()

    # import the result from dict A
    for key in A.keys():
        APlusB.append([int(str(key).split(",")[0]),int(str(key).split(",")[1]),A[key]])

    # sort the list for increasing order
    APlusB.sort(key=lambda x:(x[0],x[1]))

    # wire the result of A + B into file
    for APlusBItem in APlusB:
        APlusBf.write(str(APlusBItem[0])+" "+str(APlusBItem[1])+" "+str(APlusBItem[2])+"\n")

    # close the files
    APlusBf.close()
    TF.close()
    TFIDF.close()
    oneHot.close()
    sMatrix.close()
