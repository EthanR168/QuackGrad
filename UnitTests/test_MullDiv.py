from QuackGrad.TensorClass import Tensor
import numpy as np

class Test_Mull:
    def test_ScalarMull(self):
        a = Tensor(2)
        b = Tensor(3)
        c = a * b
        assert c.data == 6

    def test_1DMull(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = a * b
        assert np.all(c.data == [3, 8])

    def test_2DMull(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor([
            [5, 6],
            [7, 8],
        ])
        c = a * b
        expected = Tensor([
            [5, 12],
            [21, 32],
        ])
        assert np.all(c.data == expected.data)

    def test_3DMull(self):
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
        c = a * b
        expected = Tensor([
            [
                [9, 16],
                [21, 24],
            ],
            [
                [25, 24],
                [21, 16],
            ],
        ])
        assert np.all(c.data == expected.data)

class Test_Div:
    def test_ScalarDiv(self):
        a = Tensor(2)
        b = Tensor(3)
        c = b / a
        assert c.data == 1.5

    def test_1DDiv(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = b / a
        assert np.all(c.data == [3, 2])

    def test_2DDiv(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor([
            [4, 5],
            [6, 8],
        ])
        c = b / a
        expected = Tensor([
            [4, 2.5],
            [2, 2],
        ])
        assert np.all(c.data == expected.data)

    def test_3DDiv(self):
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
                [6, 6],
            ],
            [
                [5, 12],
                [14, 2],
            ],
        ])
        c = b / a
        expected = Tensor([
            [
                [9, 4],
                [2, 1.5],
            ],
            [
                [1, 2],
                [2, 0.25],
            ],
        ])
        assert np.all(c.data == expected.data)

class Test_FloorDiv:
    def test_ScalarFloorDiv(self):
        a = Tensor(2)
        b = Tensor(3)
        c = b // a
        assert c.data == 1

    def test_1DFloorDiv(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = b // a
        assert np.all(c.data == [3, 2])

    def test_2DFloorDiv(self):
        a = Tensor([
            [1, 2],
            [3, 4],
        ])
        b = Tensor([
            [4, 5],
            [6, 8],
        ])
        c = b // a
        expected = Tensor([
            [4, 2],
            [2, 2],
        ])
        assert np.all(c.data == expected.data)

    def test_3DFloorDiv(self):
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
                [6, 6],
            ],
            [
                [5, 12],
                [14, 2],
            ],
        ])
        c = b // a
        expected = Tensor([
            [
                [9, 4],
                [2, 1],
            ],
            [
                [1, 2],
                [2, 0],
            ],
        ])
        assert np.all(c.data == expected.data)

class Test_TensorOpWithScalar:
    def test_3DMul_WithScalar(self):
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
        c = a * b
        expected = Tensor([
            [
                [5, 10],
                [15, 20],
            ],
            [
                [25, 30],
                [35, 40],
            ],
        ])
        assert np.all(c.data == expected.data)

    def test_3DDiv_WithScalar(self):
        a = Tensor([
            [
                [1, 2],
                [4, 6],
            ],
            [
                [8, 10],
                [12, 14],
            ],
        ])
        b = 4
        c = a / b
        expected = Tensor([
            [
                [0.25, 0.5],
                [1, 1.5],
            ],
            [
                [2, 2.5],
                [3, 3.5],
            ],
        ])
        assert np.all(c.data == expected.data)

    def test_3DFloorDiv_WithScalar(self):
        a = Tensor([
            [
                [1, 2],
                [4, 6],
            ],
            [
                [8, 10],
                [12, 14],
            ],
        ])
        b = 4
        c = a // b
        expected = Tensor([
            [
                [0, 0],
                [1, 1],
            ],
            [
                [2, 2],
                [3, 3],
            ],
        ])
        assert np.all(c.data == expected.data)