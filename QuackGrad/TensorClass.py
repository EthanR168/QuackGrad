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
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
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
    
    def zeroGrad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float32)

    def __str__(self): # returns the tensors data like: a = 5, print(a)
        return f"{self.data}"

    def _ensureTensor(self, other):
        if(isinstance(other, Tensor) == False):
            other = Tensor(other, requiresGrad=False)
        return other

    def __mul__(self, other):
        other = self._ensureTensor(other)
        req = self.requiresGrad or other.requiresGrad
        out = Tensor(self.data * other.data, req, childTensors=[self, other], operator="*")
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
        req = self.requiresGrad or other.requiresGrad
        out = Tensor(self.data / other.data, req, childTensors=[self, other], operator="/")
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
        req = self.requiresGrad or other.requiresGrad
        out = Tensor(self.data + other.data, req, childTensors=[self, other], operator="+")
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
        req = self.requiresGrad or other.requiresGrad
        out = Tensor(self.data - other.data, req, childTensors=[self, other], operator="-")
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
        req = self.requiresGrad or other.requiresGrad
        out = Tensor(self.data // other.data, req, childTensors=[self, other], operator="//")
        def backProp_floordiv(self, grad):
            raise NotImplementedError()
        out._backProp = backProp_floordiv
        return out

    def __rfloordiv__(self, other):
        other = self._ensureTensor(other)
        return other // self

    def __pow__(self, other): # power: x ** 2
        other = self._ensureTensor(other)
        req = self.requiresGrad or other.requiresGrad
        out = Tensor(self.data ** other.data, req, childTensors=[self, other], operator="**")
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
        req = self.requiresGrad or other.requiresGrad
        out = Tensor(self.data % other.data, req, childTensors=[self, other], operator="%")
        def backProp_mod(self, grad):
            raise NotImplementedError()
        out._backProp = backProp_mod
        return out

    def __rmod__(self, other):
        other = self._ensureTensor(other)
        return other % self
    
    def __matmul__(self, other): # a @ b
        other = self._ensureTensor(other)
        req = self.requiresGrad or other.requiresGrad

        def backProp_matmul(self, grad):
            x, y = self.childTensors
            x.grad += grad @ y.data.T
            if(x.data.ndim == 1):
                y.grad += np.outer(x.data, grad)
            else:
                y.grad += x.data.T @ grad

        if(self.data.ndim == 0 or other.data.ndim == 0): 
            out = Tensor(self.data * other.data, req, childTensors=[self, other], operator="@")
        else:
            out = Tensor(self.data @ other.data, req, childTensors=[self, other], operator="@")
        out._backProp = backProp_matmul
        return out

    def __rmatmul__(self, other):
        other = self._ensureTensor(other)
        return other @ self

    def ReLU(self):
        data = np.maximum(0, self.data)
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="relu")

        def backProp_relu(self, grad):
            reluGrad = grad * (self.childTensors[0].data > 0)
            self.childTensors[0].grad += reluGrad

        out._backProp = backProp_relu
        return out
    
    def softmax(self, axis=-1):
        exps = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        sumExps = exps.sum(axis=axis, keepdims=True)
        softmax_out = exps / sumExps
        out = Tensor(softmax_out, self.requiresGrad, childTensors=[self], operator="softmax")
        
        def backProp_softmax(self, grad):
            y = softmax_out 
            dot = np.sum(grad * y, axis=axis, keepdims=True) 
            grad_input = y * (grad - dot)
            self.childTensors[0].grad += grad_input
        
        out._backProp = backProp_softmax
        return out
    
    def exp(self):
        data = np.exp(self.data)
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="exp")

        def backProp_exp(self, grad):
            self.childTensors[0].grad += grad * self.data
        out._backProp = backProp_exp
        return out
    
    def max(self, axis=None, keepdims=False):
        data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="max")

        def backProp_max(self, grad):
            grad_input = np.zeros_like(self.childTensors[0].data)
            if axis is None:
                mask = (self.childTensors[0].data == self.data)
                grad_input[mask] = grad
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                mask = (self.childTensors[0].data == np.expand_dims(self.data, axis=axis))
                grad_input += grad * mask
            self.childTensors[0].grad += grad_input
        out._backProp = backProp_max
        return out
    
    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="sum")

        def backProp_sum(self, grad):
            grad_input = grad
            if axis is not None and not keepdims:
                grad_input = np.expand_dims(grad, axis=axis)
            grad_input = np.ones_like(self.childTensors[0].data) * grad_input
            self.childTensors[0].grad += grad_input
        out._backProp = backProp_sum
        return out

    def log(self):
        data = np.log(self.data)
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="log")

        def backProp_log(self, grad):
            self.childTensors[0].grad += grad / self.childTensors[0].data

        out._backProp = backProp_log
        return out
    
    def mean(self, axis=None, keepdims=False):
        data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="mean")

        def backProp_mean(self, grad):
            gradInput = grad
            x = self.childTensors[0].data
            if axis is None:
                N = x.size
                gradInput = np.ones_like(x) * (grad / N)
            else:
                if not keepdims:
                    gradInput = np.expand_dims(grad, axis=axis)
                N = x.shape[axis]
                gradInput = np.ones_like(x) * (grad / N)
            self.childTensors[0].grad += gradInput

        out._backProp = backProp_mean
        return out

    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="reshape")

        def backProp_reshape(self, grad):
            self.childTensors[0].grad += grad.reshape(self.childTensors[0].data.shape)

        out._backProp = backProp_reshape
        return out

    @property
    def T(self):
        data = self.data.T
        out = Tensor(data, self.requiresGrad, childTensors=[self], operator="transpose")

        def backProp_T(self, grad):
            self.childTensors[0].grad += grad.T

        out._backProp = backProp_T
        return out

    def __neg__(self):
        out = Tensor(-self.data, self.requiresGrad, childTensors=[self], operator="neg")

        def backProp_neg(self, grad):
            self.childTensors[0].grad -= grad

        out._backProp = backProp_neg
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
                if(v.childTensors is not None):
                    for child in v.childTensors:
                        build(child)
                tree.append(v)
        build(self)
        
        for node in reversed(tree):
            if(node.requiresGrad == True):
                node._backProp(node, node.grad)
