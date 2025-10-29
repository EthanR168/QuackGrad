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
    def __init__(self, data, requiresGrad=True, childTensors=None, operator=""):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.childTensors = childTensors
        self.operator = operator
        self._backProp = lambda self, grad: None
        self.requiresGrad = requiresGrad
    
    @staticmethod
    def rand(shape, requiresGrad=True):
        data = np.random.rand(*shape).astype(np.float32)
        return Tensor(data, requiresGrad=requiresGrad)
    
    @staticmethod
    def randNormal(bounds, shape, requiresGrad=True):
        data = np.random.normal(0, bounds, size=(shape))
        return Tensor(data, requiresGrad=requiresGrad)
    
    @staticmethod
    def zeros(shape, requiresGrad=True):
        data = np.zeros(shape).astype(np.float32)

        return Tensor(data, requiresGrad=requiresGrad)

    @property
    def T(self):
        from .Ops.matrix import transpose
        return transpose(self)    

    def zeroGrad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float32)

    def __str__(self): # returns the tensors data like: a = 5, print(a)
        return f"{self.data}"

    def _ensureTensor(self, other):
        if(isinstance(other, Tensor) == False):
            other = Tensor(other, requiresGrad=False)
        return other

    def __mul__(self, other):
        from .Ops.basic import mul
        return mul(self, other)

    def __rmul__(self, other):
        other = self._ensureTensor(other)
        return other * self

    def __truediv__(self, other):
        from .Ops.basic import truediv
        return truediv(self, other)

    def __rtruediv__(self, other):
        other = self._ensureTensor(other)
        return other / self

    def __add__(self, other):
        from .Ops.basic import add
        return add(self, other)

    def __radd__(self, other):
        other = self._ensureTensor(other)
        return other + self

    def __sub__(self, other):
        from .Ops.basic import sub
        return sub(self, other)

    def __rsub__(self, other):
        other = self._ensureTensor(other)
        return other - self

    def __floordiv__(self, other):
        from .Ops.basic import floordiv
        return floordiv(self, other)

    def __rfloordiv__(self, other):
        other = self._ensureTensor(other)
        return other // self

    def __pow__(self, other): # power: x ** 2
        from .Ops.basic import pow
        return pow(self, other)

    def __rpow__(self, other):
        other = self._ensureTensor(other)
        return other ** self

    def __mod__(self, other): # x % 2
        from .Ops.basic import mod
        return mod(self, other)

    def __rmod__(self, other):
        other = self._ensureTensor(other)
        return other % self
    
    def __matmul__(self, other): # a @ b
        from .Ops.matrix import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        other = self._ensureTensor(other)
        return other @ self

    def ReLU(self):
        from .Ops.basic import ReLU
        return ReLU(self)
    
    def softmax(self, axis=-1):
        from .Ops.statistics import softmax
        return softmax(self, axis)
    
    def exp(self):
        from .Ops.basic import exp
        return exp(self)
    
    def max(self, axis=None, keepdims=False):
        from .Ops.statistics import max
        return max(self, axis, keepdims)
    
    def sum(self, axis=None, keepdims=False):
        from .Ops.statistics import sum
        return sum(self, axis, keepdims)

    def log(self):
        from .Ops.basic import log
        return log(self)
    
    def mean(self, axis=None, keepdims=False):
        from .Ops.statistics import mean
        return mean(self, axis, keepdims)

    def reshape(self, *shape):
        from .Ops.matrix import reshape
        return reshape(self, *shape)

    def __neg__(self):
        from .Ops.basic import neg
        return neg(self)

    def backwardPropagation(self, grad=None):
        if(grad is None):
            grad = np.ones_like(self.data)
        self.grad += np.array(grad)
        tree = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                if(v.childTensors is not None):
                    for child in v.childTensors:
                        build(child)
                tree.append(v)
        build(self)
        
        for node in reversed(tree):
            if(node.requiresGrad == True):
                node._backProp(node, node.grad)
