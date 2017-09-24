from __future__ import print_function

import math

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return 1. / (1. + math.exp(-x))

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return y * (1. - y)

# make a matrix (we could use NumPy to speed this up)
def make_matrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = make_matrix(self.ni, self.nh)  # [ni, nh]
        self.wo = make_matrix(self.nh, self.no)

        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = 0.0

        # bias term
        # (w, b) (\in (nh, ni+1)) \frac{x}{1}
        for j in range(self.nh):
            self.wi[self.ni - 1][j] = 1.0

        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = 0.0

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def back_propagate(self, targets, lr):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += lr * change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += lr * change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def train(self, inputs, targets, lr=0.5):
        # N: learning rate
        errors = []
        for input, target in zip(inputs, targets):
            self.update(input)
            error = self.back_propagate(target, lr)
            errors.append(error)

        print(self.wi)
        return errors


_input = input()

_N, _I, _H, _O = _input.split()
_N = int(_N)
_I = int(_I)
_H = int(_H)
_O = int(_O)

_inputs = []
_targets = []
for _inputs_i in range(_N):
    _inputs_temp = list(map(float, input().strip().split(' ')))
    _inputs.append(_inputs_temp)
    _targets_temp = list(map(float, input().strip().split(' ')))
    _targets.append(_targets_temp)

nn = NN(_I, _H, _O)
res = nn.train(_inputs, _targets)

for res_cur in res:
    print(str(res_cur))

"""
4 2 2 1
0 0
1
0 1
1
1 0
1
1 1
0

0.125
0.121
0.114
0.145
"""
