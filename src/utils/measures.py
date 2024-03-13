import numpy as np
from sklearn.metrics import (
    f1_score,
    adjusted_rand_score,
    accuracy_score,
    normalized_mutual_info_score,
    silhouette_score
)
import pandas as pd
import sys

sys.path.append("..")


def cover_calculator(Y_pred):
    sample_num = Y_pred.shape[0]
    noise_num = Y_pred[Y_pred == -1].shape[0]
    return 1 - noise_num / sample_num


def f1_score_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred != -1]
    Y_true_ = Y_true[Y_pred != -1]
    return f1_score(Y_true_, Y_pred_, average="weighted")


def ARI_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred != -1]
    Y_true_ = Y_true[Y_pred != -1]
    return adjusted_rand_score(Y_true_, Y_pred_)


def ACC_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred != -1]
    Y_true_ = Y_true[Y_pred != -1]
    return accuracy_score(Y_true_, Y_pred_)


def match(current, true_set, noise_set):
    max_overlap = 0
    idx = None
    for j in true_set.keys():
        N = len(current & true_set[j] - noise_set)
        if N > max_overlap:
            max_overlap = N
            idx = j
    return idx


def alignPredictedWithTrueLabels(Y_pred, Y_true):
    """
    Aligns predicted labels with true labels to accurately evaluate clustering performance.

    This function adjusts the predicted labels to best match the true labels, facilitating
    a meaningful comparison between them. It handles noise in both predicted and true labels,
    ensuring that outliers do not distort the evaluation. The function specifically:
    - Filters 'noise' from true labels, if present.
    - Identifies and isolates noise within predicted labels.
    - Remaps predicted labels to match with true labels as closely as possible.
    - Reassigns the noise label to originally identified noisy predictions.

    Parameters:
    - Y_pred: array-like, Predicted labels as output by a clustering algorithm.
    - Y_true: array-like, True labels for the data points.

    Returns:
    - Y_pred: array-like, Predicted labels aligned with the true labels.
    - Y_true: array-like, True labels filtered and adjusted for direct comparison with Y_pred.
    """
    if type(Y_true[0]) == str:
        select_mask = Y_true != "noise"
        Y_true = Y_true[select_mask]
        Y_true = Y_true.astype(np.int)
        Y_pred = Y_pred[select_mask]
    noise_mask = Y_pred == -1
    noise_set = set(np.nonzero(noise_mask)[0])

    # the set of predicted classes, for example：pred_set[-1]={816,501}
    pred_set = {}
    for i, val in enumerate(list(set(Y_pred))):
        pred_set[i] = set(np.where(Y_pred == val)[0]) - noise_set

    true_set = {}
    for i, val in enumerate(sorted(list(set(Y_true)))):
        true_set[i] = set(np.where(Y_true == val)[0])

    Y_true = np.zeros_like(Y_pred)
    for i in true_set.keys():
        Y_true[list(true_set[i])] = i

    # sort the index of set according to its number of points
    sort_idx = np.argsort(-np.array(list(map(len, pred_set.values()))))

    # initial with -2, representing the false predicted points
    Y_pred = np.zeros_like(Y_pred) - 2
    for i in sort_idx:
        if len(true_set) == 0:
            break
        pred_idx = list(pred_set.keys())[i]
        # find the real label of pred_set[pred_idx] from true_set
        real_y = match(pred_set[pred_idx], true_set, noise_set)
        if real_y is None:
            Y_pred[list(pred_set[pred_idx])] = -2
        else:
            Y_pred[list(pred_set[pred_idx])] = real_y
            del true_set[real_y]
    Y_pred[noise_mask] = -1

    return Y_pred, Y_true


def measures_calculator(X, Y_true, Y_pred):
    """
    Calculates various clustering evaluation metrics based on true labels, predicted labels,
    and the original data points.
    
    Parameters:
    - X: array-like of shape (n_samples, n_features), Original data points used for clustering.
    - Y_true: array-like of shape (n_samples,), True labels for each sample.
    - Y_pred: array-like of shape (n_samples,), Predicted cluster labels for each sample.
    
    Returns:
    - DataFrame containing clustering metrics: F1 Score, Adjusted Rand Index (ARI),
      Accuracy (ACC), Normalized Mutual Information (NMI), Silhouette Score,
      coverage rate, and number of classes.
    """
    # Exclude noise points from evaluation
    is_not_noise = Y_pred != -1
    X_filtered = X[is_not_noise]
    Y_pred_filtered = Y_pred[is_not_noise]
    Y_true_filtered = Y_true[is_not_noise]
    
    # Calculate metrics
    num_classes = len(set(Y_pred_filtered)) - (1 if -1 in Y_pred_filtered else 0)
    coverage_rate = len(Y_pred_filtered) / len(Y_pred) if len(Y_pred) > 0 else 0
    
    f1 = f1_score(Y_true_filtered, Y_pred_filtered, average='weighted')
    ari = adjusted_rand_score(Y_true_filtered, Y_pred_filtered)
    acc = accuracy_score(Y_true_filtered, Y_pred_filtered)
    nmi = normalized_mutual_info_score(Y_true_filtered, Y_pred_filtered)
    
    # Calculate additional metrics if the predicted labels form at least one cluster excluding noise
    if num_classes > 0:
        silhouette = silhouette_score(X_filtered, Y_pred_filtered) if num_classes > 1 else 0
    else:
        silhouette= 0
    
    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        "f1": [f1], 
        "ARI": [ari], 
        "ACC": [acc], 
        "NMI": [nmi], 
        "Silhouette": [silhouette],
        "cover_rate": [coverage_rate], 
        "classes": [num_classes]
    })
    
    return metrics_df