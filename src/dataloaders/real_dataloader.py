import numpy as np
import pandas as pd
import os.path as osp
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets

small_scale_datasets = ["iris", "wine", "glass", "breast_cancer", "hepatitis", "fish"]
large_scale_datasets = ["face", "mnist_784", "fmnist_784", "codon"]


class Real_DataLoader:
    """
    A data loader class designed to load and preprocess datasets for machine learning models.
    It supports a variety of datasets, handling them based on their characteristics.
    
    Attributes:
        name (str): The name of the dataset to be loaded.
        path (str): The path to the directory containing the datasets.
    
    Methods:
        load(): Loads and preprocesses the specified dataset.
    """
    def __init__(self, name, path="../datasets/real_datasets/"):
        """
        Initializes the Real_DataLoader object with dataset name and path.
        
        Parameters:
            name (str): The name of the dataset to load.
            path (str): The directory path where datasets are stored, defaulting to "../datasets/real_datasets/".
            
        Raises:
            ValueError: If the provided dataset name is not recognized.
        """
        self.name = name
        self.path = path
        if not (name in small_scale_datasets or name in large_scale_datasets):
            raise ValueError("Dataset not recognized 🥲")

    def load(self):
        """
        Loads and preprocesses the dataset specified during the object initialization.
        
        This method applies appropriate preprocessing for each dataset, such as dropping specific columns,
        handling missing values, and encoding the target variable. It adapts based on the dataset to load
        small-scale datasets from CSV files, fetch large-scale datasets like face, MNIST, and FashionMNIST,
        and specifically handle the "codon" dataset by dropping specified columns and label encoding the target.
        
        Returns:
            X (np.ndarray): The features of the dataset in a NumPy array format.
            Y_true (np.ndarray): The labels of the dataset, processed as required (e.g., label encoded).
            
        Raises:
            Exception: If there is an issue loading or processing the dataset.
        """
        try:
            if self.name in small_scale_datasets:
                df = pd.read_csv(osp.join(self.path, self.name + ".csv"), header=None)

            if self.name == "iris":
                X = df.iloc[1:, :-1].values.astype(float)
                Y_true = df.iloc[1:, -1].values.astype(float)
            elif self.name == "wine":
                X = df.iloc[1:, :-1].values.astype(float)
                Y_true = df.iloc[1:, -1].values.astype(int)
                Y_set = list(set(Y_true))
                Y_map = {Y_set[i]: i for i in range(len(Y_set))}
                Y_true = np.array([Y_map[y] for y in Y_true])
            elif self.name == "breast_cancer":
                X = df.iloc[:, 2:].values.astype(float)
                Y_true = df.iloc[:, 1]
                Y_set = list(set(Y_true))
                Y_map = {Y_set[i]: i for i in range(len(Y_set))}
                Y_true = np.array([Y_map[y] for y in Y_true])
            elif self.name == "hepatitis":
                df.replace("?", np.nan, inplace=True)
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.fillna(df.mean())
                X = df.iloc[1:, 1:].values.astype(float)
                Y_true = df.iloc[1:, 0].values.astype(int)
                Y_set = list(set(Y_true))
                Y_map = {Y_set[i]: i for i in range(len(Y_set))}
                Y_true = np.array([Y_map[y] for y in Y_true])
            elif self.name == "face":
                X, Y_true = fetch_olivetti_faces(return_X_y=True, shuffle=True)
            elif self.name == "mnist_784":
                mnist = datasets.MNIST(self.path, train=True, download=True)
                N = mnist.test_data.shape[0]
                X = mnist.test_data.numpy().reshape(N, 784) / 255
                Y_true = mnist.test_labels.numpy()
            elif self.name == "fmnist_784":
                fmnist = datasets.FashionMNIST(self.path, train=True, download=True)
                N = fmnist.test_data.shape[0]
                X = fmnist.test_data.numpy().reshape(N, 784) / 255
                Y_true = fmnist.test_labels.numpy()
            elif self.name == "fish":
                label_encoder = LabelEncoder()
                # The first column (at index 0) is the target that needs label encoding
                Y_true = label_encoder.fit_transform(df.iloc[1:, 0].values)
                # The rest of the columns (starting from index 1) are features
                X = df.iloc[1:, 1:].values.astype(float)
            elif self.name == "codon":
                    df = pd.read_csv(osp.join(self.path, self.name + ".csv"), header=0, low_memory=False)
                    df.drop(['UUU', 'UUC', 'SpeciesName'], axis=1, inplace=True)  # Drop specified columns
                    X = df.iloc[:, 1:].values.astype(float)  # Features, excluding the first column ('Kingdom')
                    label_encoder = LabelEncoder()
                    Y_true = label_encoder.fit_transform(df['Kingdom'].values)  # Label encode 'Kingdom'
            return X, Y_true
        except Exception as e:
                raise Exception(f"An error occurred while loading the dataset: {e}")