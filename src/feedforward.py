import copy
import math
from math import exp
from random import random


class FeedForward:
    def __init__(self, *count):
        self.activation = sigmoid
        self.derivative = sigmoid_derivative
        self.speed = 0.1
        self.w = []
        for i in range(len(count)-1):
            self.w.append(get_random_matrix(count[i], count[i+1]))

    def think(self, x: list) -> list[float]:
        for w in self.w:
            y = [0.0]*len(w[0])
            for j in range(len(y)):
                for i in range(len(x)):
                    y[j] += x[i] * w[i][j]
                y[j] = self.activation(y[j])
            x = y
        return x

    def learn(self, dataset: list[dict], iteration_count=-1, acceptable_error=0, delta_error=-100000):

        error = self.dataset_error(dataset)
        min_err = error
        best_w = copy.deepcopy(self.w)
        self.learn_step(dataset)

        tmp = self.dataset_error(dataset)
        d_error = error - tmp

        count = 1

        while count != iteration_count and error > acceptable_error and d_error > delta_error:
            if count % 1000 == 0:
                print(str(count * 100 / iteration_count) + "%; error=" + str(error) + ";")

            self.learn_step(dataset)
            count += 1
            self.speed *= 0.99
            tmp = self.dataset_error(dataset)
            d_error = error - tmp
            error = tmp

            if error < min_err:
                min_err = error
                best_w = copy.deepcopy(self.w)


            assert d_error > -10000

        self.w = best_w
        # print(self.w)
        return count

    def learn_step(self, dataset: list[dict]):
        dw = []
        for k in range(len(self.w)):
            dw.append([])
            for i in range(len(self.w[k])):
                dw[k].append([])
                for j in range(len(self.w[k][i])):
                    dw[k][i].append(0)

        for data in dataset:

            neurons_out = [data["in"]]
            neurons_in = []
            x = data["in"]
            for ww in self.w:
                y = []
                for i in range(len(ww[0])):
                    y.append(0.0)
                for j in range(len(y)):
                    for i in range(len(x)):
                        y[j] += x[i] * ww[i][j]
                neurons_in.append(y.copy())
                for j in range(len(y)):
                    y[j] = self.activation(y[j])
                neurons_out.append(y.copy())
                x = y

            err = [[]]
            for i in range(len(data["out"])):
                err[-1].append(data["out"][i] - neurons_out[-1][i])
            for ww in reversed(self.w):
                err.append([])
                for i in range(len(ww)):
                    err[-1].append(0)
                    for j in range(len(ww[i])):
                        # g = err[-2]
                        err[-1][i] += err[-2][j] * ww[i][j]
            err.reverse()
            for k in range(len(self.w)):
                for i in range(len(self.w[k])):
                    for j in range(len(self.w[k][i])):
                        deriv = self.derivative(neurons_in[k][j])
                        dw[k][i][j] += self.speed * err[k+1][j] * deriv * neurons_out[k][i]
                        # self.w[k][i][j] += self.speed * err[k+1][j] * deriv * neurons_out[k][i]

        for k in range(len(self.w)):
            for i in range(len(self.w[k])):
                for j in range(len(self.w[k][i])):
                    self.w[k][i][j] += dw[k][i][j] / len(dataset)

    def dataset_error(self, dataset: list[dict]) -> float:
        actual = []
        expected = []
        for data in dataset:
            actual.extend(self.think(data["in"]))
            expected.extend(data["out"])
        error = square_error(actual, expected)
        return error


def ReLU(x: float) -> float:
    if x <= 0:
        return x/100
    return x


def ReLU_derivative(x: float) -> float:
    if x <= 0:
        return 0.01
    return 1


def sigmoid(x: float) -> float:
    return 1/(1+exp(-x))*10


def sigmoid_derivative(x: float) -> float:
    return exp(-x)/((1+exp(-x))**2)


def get_random_matrix(x: int, y: int, min_val=-1, max_val=1):
    arr = []
    for xx in range(x):
        arr.append([])
        for yy in range(y):
            arr[-1].append(random() * (max_val - min_val) + min_val)
    return arr


def square_error(actual: list[float], expected: list[float]) -> float:
    err = 0
    for i in range(len(actual)):
        err += (actual[i] - expected[i])**2
    return err


def root_mean_square_error(actual: list[float], expected: list[float]) -> float:
    err = square_error(actual, expected)
    err /= len(actual)
    err = math.sqrt(err)
    return err

