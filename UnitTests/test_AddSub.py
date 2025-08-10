from QuackGrad.TensorClass import Tensor
import numpy as np

class Test_Add:
    def test_ScalarAdd(self):
        a = Tensor(2)
        b = Tensor(3)
        c = a + b
        assert c.data == 5

    def test_1DAdd(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = a + b
        assert np.all(c.data == [4, 6])

    def test_2DAdd(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor([
            [5, 6],
            [7, 8],
        ])
        c = a + b
        expected = Tensor([
            [6, 8],
            [10, 12],
        ])
        assert np.all(c.data == expected.data)

    def test_3DAdd(self):
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
        c = a + b
        expected = Tensor([
            [
                [10, 10],
                [10, 10],
            ],
            [
                [10, 10],
                [10, 10],
            ],
        ])
        assert np.all(c.data == expected.data)

class Test_Sub:
    def test_ScalarSub(self):
        a = Tensor(2)
        b = Tensor(3)
        c = a - b
        assert np.all(c.data == -1)

    def test_1DSub(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = a - b
        assert np.all(c.data == [-2, -2])

    def test_2DSub(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor([
            [5, 6],
            [7, 8],
        ])
        c = a - b
        expected = Tensor([
            [-4, -4],
            [-4, -4],
        ])
        assert np.all(c.data == expected.data)

    def test_3DSub(self):
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
                [9, 10],
                [11, 12],
            ],
            [
                [13, 14],
                [15, 16],
            ],
        ])
        c = a - b
        expected = Tensor([
            [
                [-8, -8],
                [-8, -8],
            ],
            [
                [-8, -8],
                [-8, -8],
            ],
        ])
        assert np.all(c.data == expected.data)

class Test_TensorOpWithScalar:
    def test_3DAdd_WithScalar(self):
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
        b = 5
        c = a + b
        expected = Tensor([
            [
                [6, 7],
                [8, 9],
            ],
            [
                [10, 11],
                [12, 13],
            ],
        ])
        assert np.all(c.data == expected.data)

    def test_3DSub_WithScalar(self):
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
        b = 5
        c = a - b
        expected = Tensor([
            [
                [-4, -3],
                [-2, -1],
            ],
            [
                [0, 1],
                [2, 3],
            ],
        ])
        assert np.all(c.data == expected.data)

        