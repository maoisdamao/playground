import math
import random
import numpy as np


def sigmoid(x):
    return 1/(1+math.exp(-x))


# the derivative function of sigmoid
def dsigmoid(y):
    return y*(1-y)


def initRandomMatrix(I, J, a, b):
    matrix = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            matrix[i][j] = random.uniform(a, b)
    return matrix


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 是为了偏置节点
        self.nh = nh
        self.no = no

        # 激活值（输出值）
        self.ai = [1.0] * self.ni  # input
        self.ah = [1.0] * self.nh  # hidden layer
        self.ao = [1.0] * self.no  # output layer

        # weights matrix randomize init
        self.wi = initRandomMatrix(self.ni, self.nh,  -0.2, 0.2)
        self.wo = initRandomMatrix(self.nh, self.no,  -2.0, 2.0)

    def forwardPropagate(self, inputs):
        # forward propagation
        if len(inputs) != self.ni - 1:
            print('incorrect number of inputs')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += (self.ai[i] * self.wi[i][j])
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += (self.ah[j] * self.wo[j][k])
            self.ao[k] = sigmoid(sum)

        return self.ao

    def backPropagate(self, targets, N):
        # compute the output deltas
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])
        # update output weight matrix
        for j in range(self.nh):
            for k in range(self.no):
                # dError/dweight[j][k]
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change

        # compute hidden layer deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])

        # update input weight matrix
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change

        # 计算误差平方和
        # 1/2 是为了好看，**2 是平方
        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def print_weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])
        print('')

    def test(self, patterns):
        for p in patterns:
            inputs = p[0]
            print('Inputs:', p[0], '\tTarget', p[1],
                  '\tPrediction:', self.forwardPropagate(inputs))

    def train(self, patterns, max_iterations=2000, N=0.5):
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.forwardPropagate(inputs)
                error = self.backPropagate(targets, N)

        print("iterations:", max_iterations, "learning rate:", N)
        print('square error', error)
        self.test(patterns)


if __name__ == "__main__":
    pattern = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [1]]
    ]
    testNN = NN(2, 3, 1)
    testNN.train(pattern)
