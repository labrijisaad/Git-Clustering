import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .detect_local_mode import LCluster
from .topo_graph import TopoGraph

class GIT:
    def __init__(self, k=8, target_ratio=[1.0, 1.0], n_jobs=100):
        """
        Graph Intensity Topology (GIT) clustering algorithm designed to identify complex,
        non-convex clusters by leveraging the topology derived from the intensity function
        of the data points. It combines density-based clustering with topological data analysis
        to delineate cluster boundaries more accurately.

        Parameters
        ----------
        k : int, default=8
            The number of nearest neighbors to use for estimating local densities and constructing
            the topological graph of the data points.

        target_ratio : list of float, default=[1.0, 1.0]
            The target ratio of cluster sizes to be used during the pruning phase of the topological graph.
            It is utilized to achieve a distribution of cluster sizes that best matches this target ratio.

        n_jobs : int, default=100
            The number of parallel jobs to run for computations. Increasing this number can speed up
            the computation at the expense of memory consumption.

        Attributes
        ----------
        k : int
            The number of neighbors used in density estimation.

        target_ratio : list of float
            The desired ratio of cluster sizes for the pruning process.

        n_jobs : int
            The number of parallel jobs used during computation.
        """
        self.k = k
        self.target_ratio = target_ratio
        self.n_jobs = n_jobs

    def fit_predict(self, X):
        """
        Apply the GIT clustering algorithm to the input data, resulting in a set of cluster labels
        for each data point.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        Y : ndarray of shape (n_samples,)
            Cluster labels for each point in the dataset. The label is an integer that represents
            the assigned cluster, with different clusters corresponding to different integers.

        Notes
        -----
        The GIT clustering process involves the following steps:
        1. Detecting local clusters and descending manifolds in the input data.
        2. Constructing a topological graph where nodes represent local clusters.
        3. Pruning noisy edges from the graph to reveal the underlying global structure.
        4. Finalizing the clustering by merging connected local clusters based on the pruned topological graph.
        """
        LC = LCluster(k=self.k, n_jobs=self.n_jobs)
        V, Boundary, X_extend, Dis = LC.detect_descending_manifolds(X)

        TG = TopoGraph(target_ratio=self.target_ratio)
        _, _, clusters = TG.topograph_construction_pruning(V, X_extend, Boundary, Dis)

        for c, cluster_index in clusters.items():
            for cluster in cluster_index:
                X_extend[V[cluster], -1] = c

        Y = X_extend[:, -1].astype(int)
        return Y

    def plot_and_summarize_clusters(self, X, labels):
        """
        Plot the data points in a 2D space, with each point colored according to its cluster label,
        and calculate statistics for each cluster, including size and percentage. Return these
        statistics as a pandas DataFrame.
        """
        if X.shape[1] != 2:
            raise ValueError("X needs to be a 2D array for this plot function.")

        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # Plotting clusters
        plt.figure(figsize=(4, 3))
        ax = plt.subplot(111)
        for label, color in zip(unique_labels, colors):
            cluster_points = X[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10,
                       color=color, label=f'Cluster {label}')

        plt.title('Cluster Plot')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        # Calculating statistics
        total_samples = len(labels)
        stats = {'Cluster Label': [], 'Size': [], 'Percentage': []}

        for label in unique_labels:
            cluster_data = X[labels == label]
            size = cluster_data.shape[0]
            stats['Cluster Label'].append(label)
            stats['Size'].append(size)
            stats['Percentage'].append(round((size / total_samples) * 100, 2))

        # Creating the DataFrame from the dictionary
        stats_df = pd.DataFrame.from_dict(stats).set_index('Cluster Label')

        # Ensure DataFrame is sorted by cluster labels
        stats_df = stats_df.loc[unique_labels]

        return stats_df.sort_values(by=['Size'], ascending=False).T