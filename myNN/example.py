import numpy as np
import math

alphas = [0.01]


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

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

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_train(0.8, "./train.csv")

    X = np.array(x_train)
    y = np.array(y_train)

    for alpha in alphas:

        # randomly initialize our weights with mean 0
        synapse_0 = 2 * np.random.random((59, 12)) - 1
        synapse_1 = 2 * np.random.random((12, 1)) - 1

        for j in range(10000):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0, synapse_0))
            layer_2 = sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = layer_2 - y

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))

        correct = 0
        for i in range(len(y_train)):
            if abs(layer_2[i] - y[i]) < 0.5:
                correct += 1
        print("train accuracy = ",correct/len(y_train))
        print(correct,len(y_train))

        layer_0 = x_test
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        correct = 0
        for i in range(len(y_test)):
            if abs(layer_2[i] - y_test[i]) < 0.5:
                correct += 1
        print("test accuracy = ", correct / len(y_test))
        print(correct, len(y_test))
        #
        # x_train, y_train, x_test, y_test = read_train(1, "./data/test.csv")
        # layer_0 = x_train
        # layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        # layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        #
        # correct = 0
        # for i in range(len(y_train)):
        #     if abs(layer_2[i] - y[i]) < 0.5:
        #         correct += 1
        # print("test accuracy = ", correct / len(y_train))
        # print(correct, len(y_train))