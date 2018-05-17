import numpy as np


class Relu:
    @staticmethod
    def activation(z):
        z[z<0] = 0
        return z


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1/(1 + np.exp(-z))