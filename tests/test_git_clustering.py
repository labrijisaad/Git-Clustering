import pytest
import numpy as np
import sys

sys.path.append("src/")
from git_cluster import GIT
from utils import alignPredictedWithTrueLabels, measures_calculator
from dataloaders import Toy_DataLoader, Real_DataLoader

# Define data path
toy_datasets_path = "datasets/toy_datasets"
real_datasets_path = "datasets/real_datasets"


def load_dataset(loader, name, path, sample_fraction=None):
    data, labels = loader(name=name, path=path).load()
    # If a sample fraction is provided and less than 1, sample the data
    if sample_fraction and sample_fraction < 1.0:
        np.random.seed(42)  # Ensure reproducibility
        sample_size = int(len(data) * sample_fraction)
        indices = np.random.choice(len(data), sample_size, replace=False)
        return data[indices], labels[indices]
    return data, labels


@pytest.fixture(
    params=[
        "circles",
        "impossible",
        "moons",
        "s-set",
        "smile",
        "complex8",
        "complex9",
        "chainlink",
    ]
)
def toy_dataset(request):
    return load_dataset(Toy_DataLoader, request.param, toy_datasets_path)


@pytest.fixture(params=["iris", "wine", "breast_cancer", "hepatitis", "fish"])
def small_scale_dataset(request):
    return load_dataset(Real_DataLoader, request.param, real_datasets_path)


@pytest.fixture(params=["face", "mnist_784", "fmnist_784", "codon"])
def large_scale_dataset(request):
    # Load only a 5% sample for large datasets
    return load_dataset(
        Real_DataLoader, request.param, real_datasets_path, sample_fraction=0.05
    )


# Test functions for toy datasets
def test_toy_datasets(toy_dataset):
    X, Y_true = toy_dataset
    git = GIT(k=12)  # k may vary based on the dataset characteristics
    Y_pred = git.fit_predict(X)
    Y_pred, Y_true = alignPredictedWithTrueLabels(Y_pred, Y_true)
    result = measures_calculator(X, Y_true, Y_pred)
    assert "f1" in result, "F1 score not calculated"


# Test functions for small scale real datasets
def test_small_scale_datasets(small_scale_dataset):
    X, Y_true = small_scale_dataset
    git = GIT(k=12)  # k may vary based on the dataset characteristics
    Y_pred = git.fit_predict(X)
    Y_pred, Y_true = alignPredictedWithTrueLabels(Y_pred, Y_true)
    result = measures_calculator(X, Y_true, Y_pred)
    assert "f1" in result, "F1 score not calculated"


# Test functions for large scale real datasets
def test_large_scale_datasets(large_scale_dataset):
    X, Y_true = large_scale_dataset
    git = GIT(k=12)  # k may vary based on the dataset characteristics
    Y_pred = git.fit_predict(X)
    Y_pred, Y_true = alignPredictedWithTrueLabels(Y_pred, Y_true)
    result = measures_calculator(X, Y_true, Y_pred)
    assert "f1" in result, "F1 score not calculated"
