import numpy as np
from ..TensorClass import Tensor

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

def max(self, axis=None, keepdims=False):
    data = self.data.max(axis=axis, keepdims=keepdims)
    out = Tensor(data, self.requiresGrad, childTensors=[self], operator="max")
    def backProp_max(self, grad):
        x = self.childTensors[0].data

        if axis is None:
            x_max = np.max(x)
            mask = (x == x_max).astype(np.float32)                
            count = np.sum(mask, dtype=np.float32)                
            grad_input = mask * (grad / count)                    
        else:
            ax = axis if isinstance(axis, tuple) else (axis,)
            ax = tuple(a if a >= 0 else x.ndim + a for a in ax)

            x_max = np.max(x, axis=ax, keepdims=True)             
            mask = (x == x_max).astype(np.float32)                 
            count = np.sum(mask, axis=ax, keepdims=True)          

            g = grad
            if not keepdims:
                for a in sorted(ax):
                    g = np.expand_dims(g, axis=a)

            grad_input = mask * (g / count)    
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

