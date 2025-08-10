from QuackGrad.TensorClass import Tensor
import numpy as np

"""
Simple NN which is going to train on MNIST

Architecure:
-   784 (Input)
-   128 (Hidden, ReLU)
-   64 (Hidden, ReLU)
-   10 (Output, SoftMax)

loss function: cross entropy
"""

inputData = np.random.rand(784)
labels = np.zeros(10)
labels[3] = 1

layers = [784, 128, 64, 10]

weights = [
    Tensor.rand((784, 128)),
    Tensor.rand((128, 64)),
    Tensor.rand((64, 10)),
]

biases = [
    Tensor.rand((128,)),
    Tensor.rand((64,)),
    Tensor.rand((10,)),
]

# forward pass
x = Tensor(inputData)
z1 = x @ weights[0] + biases[0]
h1 = z1.ReLU()

z2 = h1 @ weights[1] + biases[1]
h2 = z2.ReLU()

out = h2 @ weights[2] + biases[2]
probabilities = out.softmax()

# backward propagation
outputGradient = probabilities.data - labels

probabilities.backwardPropagation(outputGradient)
