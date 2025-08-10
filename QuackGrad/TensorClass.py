"""
Auto grad, tracks operations and dynamically calculates their gradients.
It does this by creating a directed acyclic graph (DAG) (Here is PyTorch talking about it: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) 
In which, each tensor is a node and has children nodes which were used to calculate it.
Given this, in backpropagation it starts by starting at the greatest grand parent (the node with no parents)
and uses the chain rule to calculate the gradients for the nodes underneath it.

Auto grad can be used for tensor and vectors but also scalars (e.g., y = x^2 where x is a single peice of data like: 5)

Example:
    lets say you have the equation:
    y = x^2 + 3z

    the auto grad will create a DAG like this:
                  y
                /   \
                x^2   3z

    then when backprop is called for y:
                  y'
                /   \
            (x^2)'  (3z)'
                
    where (x^2)' and (3z)' are partial derivatives

    so, lets say the dy/dy = 1:
                 y
                /   \
            x^2   3z

    so the partial gradient of x^2: 2x * upstreamGradient
    and the partial gradient of 3z: 3 * upstreamGradient

    The nice thing is this is done dynamically and so auto grad can be used to backpropagate in ML libraries
    without having to code the backpropagation functions out right.

Note:
    Unlike pytorch this wont do backpropagation for complex ML layers like pooling, or convolutional layers (in CNN)
    Also wont be overly optimised
"""
#To get data object overrides: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types 


import numpy as np

class Tensor:
    def __init__(self, data, childTensors=[], operator=""):
        self.data = np.array(data)
        self.gradient = np.zeros_like(self.data)
        self.childTensors = childTensors
        self.operator = operator
    
    def __str__(self): # returns the tensors data like: a = 5, print(a)
        return f"{self.data}"

    def _ensureTensor(self, other):
        if(isinstance(other, Tensor) == False):
            other = Tensor(other)
        return other

    def __mul__(self, other):
        other = self._ensureTensor(other)
        return Tensor(self.data * other.data, [self, other], "*")

    def __rmul__(self, other):
        other = self._ensureTensor(other)
        return other * self

    def __truediv__(self, other):
        other = self._ensureTensor(other)
        return Tensor(self.data / other.data, [self, other], "/")

    def __rtruediv__(self, other):
        other = self._ensureTensor(other)
        return other / self

    def __add__(self, other):
        other = self._ensureTensor(other)
        return Tensor(self.data + other.data, [self, other], "+")

    def __radd__(self, other):
        other = self._ensureTensor(other)
        return other + self

    def __sub__(self, other):
        other = self._ensureTensor(other)
        return Tensor(self.data - other.data, [self, other], "-")

    def __rsub__(self, other):
        other = self._ensureTensor(other)
        return other - self

    def __floordiv__(self, other):
        other = self._ensureTensor(other)
        return Tensor(self.data // other.data, [self, other], "//")

    def __rfloordiv__(self, other):
        other = self._ensureTensor(other)
        return other // self

    def __pow__(self, other): # power: x ** 2
        other = self._ensureTensor(other)
        return Tensor(self.data ** other.data, [self, other], "**")

    def __rpow__(self, other):
        other = self._ensureTensor(other)
        return other ** self

    def __mod__(self, other): # x % 2
        other = self._ensureTensor(other)
        return Tensor(self.data % other.data, [self, other], "%")

    def __rmod__(self, other):
        other = self._ensureTensor(other)
        return other % self
    
    def __matmul__(self, other): # a @ b
        other = self._ensureTensor(other)
        return Tensor(self.data @ other.data, [self, other], "@")

    def __rmatmul__(self, other):
        other = self._ensureTensor(other)
        return other @ self

