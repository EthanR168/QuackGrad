import numpy as np
from ..TensorClass import Tensor

def add(self, other):
    other = self._ensureTensor(other)
    req = self.requiresGrad or other.requiresGrad
    out = Tensor(self.data + other.data, req, childTensors=[self, other], operator="+")
    def backProp_add(self, grad):
        self.childTensors[0].grad += grad 
        self.childTensors[1].grad += grad
    out._backProp = backProp_add
    return out

def sub(self, other):
    other = self._ensureTensor(other)
    req = self.requiresGrad or other.requiresGrad
    out = Tensor(self.data - other.data, req, childTensors=[self, other], operator="-")
    def backProp_sub(self, grad):
        self.childTensors[0].grad += grad 
        self.childTensors[1].grad -= grad
    out._backProp = backProp_sub
    return out

def mul(self, other):
    other = self._ensureTensor(other)
    req = self.requiresGrad or other.requiresGrad
    out = Tensor(self.data * other.data, req, childTensors=[self, other], operator="*")
    def backProp_mul(self, grad):
        self.childTensors[0].grad += grad * self.childTensors[1].data
        self.childTensors[1].grad += grad * self.childTensors[0].data
    out._backProp = backProp_mul
    return out

def truediv(self, other):
    other = self._ensureTensor(other)
    req = self.requiresGrad or other.requiresGrad
    out = Tensor(self.data / other.data, req, childTensors=[self, other], operator="/")
    def backProp_truediv(self, grad):
        x, y = self.childTensors
        x.grad += grad * (1 / y.data)
        y.grad += grad * (-x.data / (y.data ** 2))
    out._backProp = backProp_truediv
    return out

def floordiv(self, other):
    other = self._ensureTensor(other)
    req = self.requiresGrad or other.requiresGrad
    out = Tensor(self.data // other.data, req, childTensors=[self, other], operator="//")
    def backProp_floordiv(self, grad):
        raise NotImplementedError()
    out._backProp = backProp_floordiv
    return out

def pow(self, other): # power: x ** 2
    other = self._ensureTensor(other)
    req = self.requiresGrad or other.requiresGrad
    out = Tensor(self.data ** other.data, req, childTensors=[self, other], operator="**")
    def backProp_pow(self, grad):
        x, y = self.childTensors

        if(x.requiresGrad == True):
            x.grad += grad * (y.data * (x.data ** (y.data - 1)))

        if(y.requiresGrad == True):
            y.grad += grad * (self.data * np.log(x.data))
    out._backProp = backProp_pow
    return out

def mod(self, other): # x % 2
    other = self._ensureTensor(other)
    req = self.requiresGrad or other.requiresGrad
    out = Tensor(self.data % other.data, req, childTensors=[self, other], operator="%")
    def backProp_mod(self, grad):
        raise NotImplementedError()
    out._backProp = backProp_mod
    return out

def ReLU(self):
    data = np.maximum(0, self.data)
    out = Tensor(data, self.requiresGrad, childTensors=[self], operator="relu")

    def backProp_relu(self, grad):
        reluGrad = grad * (self.childTensors[0].data > 0)
        self.childTensors[0].grad += reluGrad

    out._backProp = backProp_relu
    return out

def exp(self):
    data = np.exp(self.data)
    out = Tensor(data, self.requiresGrad, childTensors=[self], operator="exp")

    def backProp_exp(self, grad):
        self.childTensors[0].grad += grad * self.data
    out._backProp = backProp_exp
    return out

def log(self):
    data = np.log(self.data)
    out = Tensor(data, self.requiresGrad, childTensors=[self], operator="log")

    def backProp_log(self, grad):
        self.childTensors[0].grad += grad / self.childTensors[0].data

    out._backProp = backProp_log
    return out

def neg(self):
    out = Tensor(-self.data, self.requiresGrad, childTensors=[self], operator="neg")

    def backProp_neg(self, grad):
        self.childTensors[0].grad -= grad

    out._backProp = backProp_neg
    return out

