import numpy as np
from ..TensorClass import Tensor

def matmul(self, other): # a @ b
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

def reshape(self, *shape):
    data = self.data.reshape(*shape)
    out = Tensor(data, self.requiresGrad, childTensors=[self], operator="reshape")

    def backProp_reshape(self, grad):
        self.childTensors[0].grad += grad.reshape(self.childTensors[0].data.shape)

    out._backProp = backProp_reshape
    return out

def transpose(self):
    data = self.data.T
    out = Tensor(data, self.requiresGrad, childTensors=[self], operator="transpose")

    def backProp_T(self, grad):
        self.childTensors[0].grad += grad.T

    out._backProp = backProp_T
    return out