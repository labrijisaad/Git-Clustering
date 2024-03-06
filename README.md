# Enhanced and Repackaged GIT Clustering 🌐🔍

📦 **Discover the Package on TestPyPI**: [git_cluster Package](https://test.pypi.org/project/git_cluster/)

🔍 **Dive Deeper in Our GitHub Repository**: [Git-Clustering GitHub Repo](https://github.com/labrijisaad/Git-Clustering)

## About 📖
This repository presents an enhanced version of the GIT (Graph of Intensity Topology) clustering algorithm, improved and repackaged for an ease of use. 🚀

## Features ✨
- **Broad Applicability:** Tested across a variety of datasets. 🌍
- **User-friendly Packaging:** Simplified integration into your projects. 📦

## Usage 🛠️
To get started, explore the [notebooks/Quick_Start_with_GIT.ipynb](https://github.com/labrijisaad/Git-Clustering/blob/main/notebooks/Quick_Start_with_GIT.ipynb) notebook for a step-by-step guide on applying this algorithm to your data.

## Testing in Google Colab 🧪
To validate the installation and functionality of the GIT Clustering package, follow these steps:

1. Install the package from TestPyPI:
    ```bash
    !pip install -i https://test.pypi.org/simple/ git_cluster
    ```
2. Download and prepare your dataset:
    ```bash
    !gdown --id 1yNwCStP3Sdf2lfvNe9h0WIZw2OQ3O2UP
    !unzip datasets.zip
    ```
3. Execute a sample clustering process:
    ```python
    from git_cluster import GIT
    from utils import matchY, measures_calculator, autoPlot
    from dataloaders import Toy_DataLoader as DataLoader

    X, Y_true = DataLoader(name='circles', path="/content/datasets/toy_datasets").load()
    Y_pred = GIT(k=12).fit_predict(X)
    autoPlot(X, Y_pred)
    ```

## Acknowledgments 🎉
- Original GIT Algorithm Authors

## Connect 🌐
<div align="center">
  <a href="https://www.linkedin.com/in/labrijisaad/">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" style="margin-bottom: 5px;"/>
  </a>
  <a href="https://github.com/labrijisaad">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" style="margin-bottom: 5px;"/>
  </a>
</div>