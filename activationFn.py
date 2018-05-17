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

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))
