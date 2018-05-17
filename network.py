import numpy as np


class Network:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: Dimensions of the neural net (input, hidden layer, output)
        :param activations: activations for layers

        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None

        # weights
        self.w = {}
        # biased term
        self.b = {}
        # activations
        self.activations = {}

        for i in range(len(dimensions) - 1):
            print(i)
            # Xavier initialization to prevent neuron activations from being too large or too small
            # see http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
            self.w[i+1] = np.random.rand(dimensions[i], dimensions[i+1]) / np.sqrt(dimensions[i])
            self.b[i+1] = np.zeros(dimensions[i+1])
            self.activations[i+2] = activations[i]

    def _feed_forward(self, x):
        """
        Execute a forward feed through the network
        :param x: (array) Batch of input data vectors
        :return: Node outputs and activations per layer.
                The numbering of the output is equivalent to the layer numbers
        """
        # z = x * w + b
        z = {}

        # activations: f(z)
        a = {1: x}  # first layer has no activations as input, the x is the input

        for i in range(1, self.n_layers):   # be careful with the subscript
            # current layer i
            # activation layer i+1
            z[i+1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i+1] = self.activations[i+1].activation(z[i+1])
        return z, a

    def predict(self, x):
        """

        :param x: (array), containing parameters
        :return: (array) A 2D of shape (n_cases, n_classes)
        """

        _, a = self._feed_forward(x)
        return a[self.n_layers]

    def _update_w_b(self, index, dw, delta):
        """
        update weights and biases
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error
        :return:
        """

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, 0)

    def _back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layer of the net
        a = {
            1: x,
            2: f(x*w1 + b1)
            3: f(a2*w2 + b2)
        }
        :param z: (dict) x*w + b
        :param a:  (dict) f(z)
        :param y_true: (array) One hot encoded truth vector
        :return:
        """

        # first calc for the output layer, like i = 3
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        # determine partial derivative and delta for the rest of the layer
        # each iteration requires the delta from the previous layer, propagating backward
        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i]) * self.activations[i].prime(z[i])
            dw = np.dot(a[i-1].T, delta)
            update_params[i-1] = (dw, delta)
            # update weights and biases
            for key, value in update_params.items():
                self._update_w_b(key, value[0], value[1])

