import numpy as np
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST data
trainset = datasets.MNIST(root='Examples/MNISTData', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='Examples/MNISTData', train=False, download=True, transform=transform)

# Preprocess and save the data as arrays
train_images = trainset.data.view(-1, 28*28).numpy().astype('float16') / 255.0  # Normalize pixel values to [0,1]
train_labels = np.eye(10, dtype='float16')[trainset.targets.numpy()]          # One-hot encode labels

test_images = testset.data.view(-1, 28*28).numpy().astype('float16') / 255.0
test_labels = np.eye(10, dtype='float16')[testset.targets.numpy()]

# save to disk for reuse
np.save('Examples/MNISTData/train_images.npy', train_images)
np.save('Examples/MNISTData/train_labels.npy', train_labels)
np.save('Examples/MNISTData/test_images.npy', test_images)
np.save('Examples/MNISTData/test_labels.npy', test_labels)