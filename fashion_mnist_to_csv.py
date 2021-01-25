import os
import gzip
import numpy as np
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

if __name__ == "__main__":
    train_images, train_labels = load_mnist('', kind='train')
    test_images, test_labels = load_mnist('', kind='t10k')

    np.save('fashion_mnist_train_images', train_images)
    np.save('fashion_mnist_train_labels', train_labels)

    np.save('fashion_mnist_test_images', test_images)
    np.save('fashion_mnist_test_labels', test_labels)


