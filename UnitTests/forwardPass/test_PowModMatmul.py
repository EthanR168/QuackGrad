from QuackGrad.TensorClass import Tensor
import numpy as np

class Test_Pow:
    def test_ScalarPow(self):
        a = Tensor(2)
        b = Tensor(3)
        c = a ** b
        assert c.data == 8

    def test_1DPow(self):
        a = Tensor([1, 2])
        b = Tensor(3)
        c = a ** b
        assert np.all(c.data == [1, 8])

    def test_2DPow(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor(3)
        c = a ** b
        expected = Tensor([
            [1, 8],
            [27, 64],
        ])
        assert np.all(c.data == expected.data)

    def test_3DPow(self):
        a = Tensor([
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 0],
            ],
        ])
        b = Tensor(2)
        c = a ** b
        expected = Tensor([
            [
                [1, 4],
                [9, 16],
            ],
            [
                [25, 36],
                [49, 0],
            ],
        ])
        assert np.all(c.data == expected.data)

class Test_Mod:
    def test_ScalarMod(self):
        a = Tensor(2)
        b = Tensor(3)
        c = a % b
        assert c.data == 2

    def test_1DPow(self):
        a = Tensor([1, 2])
        b = Tensor(3)
        c = a % b
        assert np.all(c.data == [1, 2])

    def test_2DPow(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor(3)
        c = a % b
        expected = Tensor([
            [1, 2],
            [0, 1],
        ])
        assert np.all(c.data == expected.data)

    def test_3DPow(self):
        a = Tensor([
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 0],
            ],
        ])
        b = Tensor(3)
        c = a % b
        expected = Tensor([
            [
                [1, 2],
                [0, 1],
            ],
            [
                [2, 0],
                [1, 0],
            ],
        ])
        assert np.all(c.data == expected.data)
        
class Test_Matmul:
    def test_ScalarMatmul(self):
        a = Tensor(2)
        b = Tensor(3)
        c = a @ b
        assert c.data == 6

    def test_1DMatmul(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = a @ b
        assert np.all(c.data == 11)

    def test_2DMatmul(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor([
            [5, 6],
            [7, 8],
        ])
        c = a @ b
        expected = Tensor([
            [1*5 + 2*7, 1*6 + 2*8],
            [3*5 + 4*7, 3*6 + 4*8],
        ])
        assert np.all(c.data == expected.data)

    def test_3DMatmul(self):
        a = Tensor([
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ])
        b = Tensor([
            [
                [9, 8],
                [7, 6],
            ],
            [
                [5, 4],
                [3, 2],
            ],
        ])
        c = a @ b
        expected = Tensor([
            [
                [1*9 + 2*7, 1*8 + 2*6],
                [3*9 + 4*7, 3*8 + 4*6],
            ],
            [
                [5*5 + 6*3, 5*4 + 6*2],
                [7*5 + 8*3, 7*4 + 8*2],
            ],
        ])
        assert np.all(c.data == expected.data)