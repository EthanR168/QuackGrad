from QuackGrad.TensorClass import Tensor
import numpy as np
import time

"""
Simple NN which is going to train on MNIST

Architecure:
-   784 (Input)
-   128 (Hidden, ReLU)
-   64 (Hidden, ReLU)
-   10 (Output, SoftMax)

loss function: cross entropy
optimisation function: SGD
"""

images = np.load('Examples/MNISTData/train_images.npy')  
labels = np.load('Examples/MNISTData/train_labels.npy')

layers = [784, 128, 64, 10]

weights = [
    Tensor.randNormal(np.sqrt(2/784), (784, 128)),
    Tensor.randNormal(np.sqrt(2/128), (128, 64)),
    Tensor.randNormal(np.sqrt(2/64), (64, 10)),
]

biases = [
    Tensor.zeros((128,)),
    Tensor.zeros((64,)),
    Tensor.zeros((10,)),
]

learningRate = 0.01

for epoch in range(10):
    start = time.time()

    numCorrect = 0
    allLoses = 0
    allProbabilities = []
    for i in range(len(images)):
        # forward pass
        x = Tensor(images[i])
        z1 = x @ weights[0] + biases[0]
        h1 = z1.ReLU()

        z2 = h1 @ weights[1] + biases[1]
        h2 = z2.ReLU()

        out = h2 @ weights[2] + biases[2]
        probabilities = out.softmax()
        allProbabilities.append(probabilities.data)

        # backward propagation
        outputGradient = probabilities.data - labels[i]
        allLoses += -np.sum(labels[i] * np.log(probabilities.data + 1e-12))

        probabilities.backwardPropagation(outputGradient)

        for x in range(len(weights)):
            weights[x].data -= learningRate * weights[x].grad
            weights[x].zeroGrad()
        for y in range(len(biases)):
            biases[y].data -= learningRate * biases[y].grad
            biases[y].zeroGrad()

    print(f"Epoch {epoch + 1} finished")
    print(f"took: {round(time.time() - start, 4)}s")
    print(f"average time per image: {round((time.time() - start) / len(images), 4)}s")

    for i in range(len(labels)):
        if(np.argmax(allProbabilities[i]) == np.argmax(labels[i])):
            numCorrect += 1

    print(f"average Loss: {allLoses / len(images)}")
    print(f"total Accuracy: {round(100 * (numCorrect / len(images)), 4)}%")
    print("")