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
        self.grad = np.zeros_like(self.data)
        self.childTensors = childTensors
        self.operator = operator
        self._backProp = lambda self, grad: None
    
    @staticmethod
    def rand(shape):
        data = np.random.rand(*shape)
        return Tensor(data)

    def __str__(self): # returns the tensors data like: a = 5, print(a)
        return f"{self.data}"

    def _ensureTensor(self, other):
        if(isinstance(other, Tensor) == False):
            other = Tensor(other)
        return other

    def __mul__(self, other):
        other = self._ensureTensor(other)
        out = Tensor(self.data * other.data, [self, other], "*")
        def backProp_mul(self, grad):
            self.childTensors[0].grad += grad * self.childTensors[1].data
            self.childTensors[1].grad += grad * self.childTensors[0].data
        out._backProp = backProp_mul
        return out

    def __rmul__(self, other):
        other = self._ensureTensor(other)
        return other * self

    def __truediv__(self, other):
        other = self._ensureTensor(other)
        out = Tensor(self.data / other.data, [self, other], "/")
        def backProp_truediv(self, grad):
            x, y = self.childTensors
            x.grad += grad * (1 / y.data)
            y.grad += grad * (-x.data / (y.data ** 2))
        out._backProp = backProp_truediv
        return out

    def __rtruediv__(self, other):
        other = self._ensureTensor(other)
        return other / self

    def __add__(self, other):
        other = self._ensureTensor(other)
        out = Tensor(self.data + other.data, [self, other], "+")
        def backProp_add(self, grad):
            self.childTensors[0].grad += grad 
            self.childTensors[1].grad += grad
        out._backProp = backProp_add
        return out

    def __radd__(self, other):
        other = self._ensureTensor(other)
        return other + self

    def __sub__(self, other):
        other = self._ensureTensor(other)
        out = Tensor(self.data - other.data, [self, other], "-")
        def backProp_sub(self, grad):
            self.childTensors[0].grad += grad 
            self.childTensors[1].grad -= grad
        out._backProp = backProp_sub
        return out

    def __rsub__(self, other):
        other = self._ensureTensor(other)
        return other - self

    def __floordiv__(self, other):
        other = self._ensureTensor(other)
        out = Tensor(self.data // other.data, [self, other], "//")
        def backProp_floordiv(self, grad):
            raise NotImplementedError()
        out._backProp = backProp_floordiv
        return out

    def __rfloordiv__(self, other):
        other = self._ensureTensor(other)
        return other // self

    def __pow__(self, other): # power: x ** 2
        other = self._ensureTensor(other)
        out = Tensor(self.data ** other.data, [self, other], "**")
        def backProp_pow(self, grad):
            x, y = self.childTensors
            x.grad += grad * (y.data * (x.data ** (y.data - 1)))
            y.grad += grad * (self.data * np.log(x.data))
        out._backProp = backProp_pow
        return out

    def __rpow__(self, other):
        other = self._ensureTensor(other)
        return other ** self

    def __mod__(self, other): # x % 2
        other = self._ensureTensor(other)
        out = Tensor(self.data % other.data, [self, other], "%")
        def backProp_mod(self, grad):
            raise NotImplementedError()
        out._backProp = backProp_mod
        return out

    def __rmod__(self, other):
        other = self._ensureTensor(other)
        return other % self
    
    def __matmul__(self, other): # a @ b
        other = self._ensureTensor(other)

        def backProp_matmul(self, grad):
            x, y = self.childTensors
            x.grad += grad @ y.data.T
            y.grad += np.outer(x.data, grad)

        if(self.data.ndim == 0 or other.data.ndim == 0): 
            out = Tensor(self.data * other.data, [self, other], "@")
        else:
            out = Tensor(self.data @ other.data, [self, other], "@")
        out._backProp = backProp_matmul
        return out

    def __rmatmul__(self, other):
        other = self._ensureTensor(other)
        return other @ self

    def ReLU(self):
        data = np.maximum(0, self.data)
        out = Tensor(data, [self], "relu")
        def backProp_relu(self, _):
            grad = out.grad
            reluGrad = grad * (self.data > 0)
            self.grad += reluGrad
        out._backProp = backProp_relu
        return out
    
    def softmax(self):
        shifted = self.data - self.data.max(axis=0, keepdims=True)
        exps = np.exp(shifted)
        sumExps = exps.sum(axis=0, keepdims=True)
        softmax_out = exps / sumExps
        out = Tensor(softmax_out, [self], "softmax")
        
        def backProp_softmax(self, grad):
            y = softmax_out 
            dot = np.sum(grad * y, axis=0, keepdims=True) 
            grad_input = y * (grad - dot)
            self.childTensors[0].grad += grad_input
        
        out._backProp = backProp_softmax
        return out
    
    def exp(self):
        data = np.exp(self.data)
        out = Tensor(data, [self], "exp")

        def backProp_exp(self, grad):
            self.childTensors[0].grad += grad * self.data
        out._backProp = backProp_exp
        return out
    
    def max(self, axis=None, keepdims=False):
        data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(data, [self], "max")

        def backProp_max(self, grad):
            grad_input = np.zeros_like(self.childTensors[0].data)
            if axis is None:
                mask = (self.childTensors[0].data == self.data)
                grad_input[mask] = grad
            else:
                expanded_out = np.expand_dims(self.data, axis=axis)
                mask = (self.childTensors[0].data == expanded_out)
                grad_input += grad * mask
            self.childTensors[0].grad += grad_input
        out._backProp = backProp_max
        return out
    
    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, [self], "sum")

        def backProp_sum(self, grad):
            grad_input = grad
            if axis is not None and not keepdims:
                grad_input = np.expand_dims(grad, axis=axis)
            grad_input = np.ones_like(self.childTensors[0].data) * grad_input
            self.childTensors[0].grad += grad_input
        out._backProp = backProp_sum
        return out

    def backwardPropagation(self, grad=None):
        if(grad is None):
            grad = np.ones_like(self.data)
        self.grad += np.array(grad)
        tree = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.childTensors:
                    build(child)
                tree.append(v)
        build(self)
        
        for node in reversed(tree):
            node._backProp(node, node.grad)