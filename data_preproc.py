import numpy as np
import random
import pickle
from sklearn.utils import shuffle


training_file = "./data/train.p"
valid_file = "./data/valid.p"
test_file = "./data/test.p"


class Dataset:

    def __init__(self):
        self.train = self.open_file(training_file)
        self.valid = self.open_file(valid_file)
        self.test = self.open_file(test_file)
        self.X_train, self.y_train = self.train["features"], self.train["labels"]
        self.X_valid, self.y_valid = self.valid["features"], self.valid["labels"]
        self.X_test, self.y_test = self.test["features"], self.test["labels"]

    def open_file(self, file):
        with open(file, mode='rb') as f:
            return pickle.load(f)

    def print_data(self):
        n_train = self.y_train.shape[0]
        n_validation = self.y_valid.shape[0]
        n_test = self.y_test.shape[0]
        image_shape = self.X_train.shape[1:3]
        n_classes = np.unique(np.concatenate((self.y_train,
                                              self.y_valid,
                                              self.y_test))).size

        print("Number of training examples =", n_train)
        print("Number of validation examples =", n_validation)
        print("Number of testing examples =", n_test)
        print("Image data shape =", image_shape)
        print("Number of classes =", n_classes)

    def pre_process(self):
        """
        Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
        converting to grayscale, etc.
        Feel free to use as many code cells as needed.
        """
        # Shuffle training data
        X_train, y_train = shuffle(self.X_train, self.y_train)
        X_valid, y_valid = shuffle(self.X_valid, self.y_valid)
        X_test, y_test = shuffle(self.X_test, self.y_test)

        # Normalize data
        X_train_norm = (X_train - 128) / 128
        X_valid_norm = (X_valid - 128) / 128
        X_test_norm = (X_test - 128) / 128

        return {"X_train": X_train_norm,
                "X_valid": X_valid_norm,
                "X_test": X_test_norm}
