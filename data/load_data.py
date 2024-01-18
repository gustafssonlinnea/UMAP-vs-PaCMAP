from sklearn import datasets
import tensorflow.keras as keras
import numpy as np
import sys
sys.path.insert(0, './data')
import spiral3D
import circle2D


def get_dataset_by_name(name='Digits'):
    """Loads the dataset of choice.

    Args:
        name (str, optional): Name of the dataset. Defaults to 'Digits'.

    Returns:
        (1d array, 1d array): X (data, or features) and y (target, or labels)
    """
    if name == 'Digits':  # 64 dimensions
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target

    elif name == 'Iris':  # 4 dimensions
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

    elif name == 'CIFAR-10':  # 3 072 dimensions
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Concatenate training and testing sets
        X = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        
        # Flatten
        num_samples, height, width, channels = X.shape
        X = X.reshape(num_samples, height * width * channels)
        y = y.flatten()
        
    elif name == 'CIFAR-100':  # 3 072 dimensions
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

        # Concatenate training and testing sets
        X = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        
        # Flatten
        num_samples, height, width, channels = X.shape
        X = X.reshape(num_samples, height * width * channels)
        y = y.flatten()
        
    elif name == 'MNIST':  # 784 dimensions
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Concatenate training and testing sets
        X = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        
        # Flatten
        num_samples, height, width = X.shape
        X = X.reshape(num_samples, height * width)
        y = y.flatten()
        
    elif name == 'FMNIST':  # 784 dimensions
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        # Concatenate training and testing sets
        X = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        
        # Flatten
        num_samples, height, width = X.shape
        X = X.reshape(num_samples, height * width)
        y = y.flatten()
    
    elif name == 'Spiral3D':  # 3 dimensions
        s = spiral3D.Spiral3D()
        X, y = s.load_data()
        
    elif name == 'Circle2D':  # 3 dimensions
        c = circle2D.Circle2D()
        X, y = c.load_data()
        
    """        
    elif name == 'LFW':
        lfw = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        X, y = lfw.data, lfw.target
        X = X.reshape(X.shape[0], -1)

    elif name == 'COIL-20':
        # loading preprocessed coil_20 dataset
        # you can change it with any dataset that is in the ndarray format, with the shape (N, D)
        # where N is the number of samples and D is the dimension of each sample
        X = np.load('./data/coil_20.npy', allow_pickle=True)
        X = X.reshape(X.shape[0], -1)
        y = np.load('./data/coil_20_labels.npy', allow_pickle=True)"""
    
    return X, y
