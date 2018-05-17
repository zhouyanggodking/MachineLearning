import numpy as np


class MSE:
    def __init__(self, activation_fn):
        """

        :param activation_fn: class object of activation function
        """
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """

        :param y_true: (array) one hot encoded truth vector
        :param y_pred: (array) prediction vector
        :return:
        """
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)
