from network import *
import numpy as py
from activationFn import *

nn = Network((2, 3, 1), (Relu, Sigmoid))

print(nn.w)
print(nn.b)
