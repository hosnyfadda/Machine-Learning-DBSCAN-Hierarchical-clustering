# 🏆 DBSCAN & Hierarchical Clustering

## 📌 Table of Contents

- [Introduction](#introduction)
- [Algorithms Overview](#algorithms-overview)
  - [DBSCAN](#dbscan-density-based-spatial-clustering-of-applications-with-noise)
  - [Hierarchical Clustering](#hierarchical-clustering)
- [🛠 Installation](#installation)
- [🚀 Usage](#usage)
- [📊 Dataset](#dataset)
- [📷 Visualization](#visualization)
- [📜 Code](#code)
- [🤝 Contributions](#contributions)
- [📜 License](#license)

---

## 📖 Introduction

This repository demonstrates the implementation of **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** and **Hierarchical Clustering** algorithms for unsupervised machine learning tasks. These clustering techniques help in identifying patterns and grouping similar data points **without predefined labels**.

---

## 🔍 Algorithms Overview

### 1️⃣ DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

✅ **What is DBSCAN?**\
A density-based clustering algorithm that groups closely packed points while marking outliers as noise.

🔹 **Key Parameters:**

- **Epsilon (eps):** Defines the maximum radius of the neighborhood.
- **MinPts:** The minimum number of points required to form a dense region.

💡 **Advantages:**\
✔ Works well with clusters of arbitrary shape.\
✔ Robust to noise and outliers.

---

### 2️⃣ Hierarchical Clustering

✅ **What is Hierarchical Clustering?**\
A clustering method that builds a **hierarchy of clusters** using **two approaches**:

- **Agglomerative (Bottom-Up):** Starts with individual points and merges them.
- **Divisive (Top-Down):** Starts with a single cluster and splits it into smaller ones.

📌 **Linkage Criteria:**

- 🔹 **Single Linkage:** Minimum distance between clusters.
- 🔹 **Complete Linkage:** Maximum distance between clusters.
- 🔹 **Average Linkage:** Mean distance between clusters.

💡 **Advantages:**\
✔ Provides a visual dendrogram representation.\
✔ No need to specify the number of clusters in advance.

---

## 🛠 Installation

Ensure you have the following dependencies installed before running the script:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 🚀 Usage

To run the clustering script, execute:

```bash
python clustering.py
```

---

## 📊 Dataset

This repository supports **custom datasets** and built-in datasets from `scikit-learn`. You can modify the script to use different datasets.

---

## 📷 Visualization

- 📍 **DBSCAN:** Scatter plots showing **core, border, and noise points**.
- 📊 **Hierarchical Clustering:** Dendrograms and scatter plots for better cluster visualization.

---

## 📜 Code

Here is the Python implementation of DBSCAN and Hierarchical Clustering:

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_moons
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic dataset
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=2)
hierarchical_labels = hierarchical.fit_predict(X)

# Plot DBSCAN results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', edgecolors='k')
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot Hierarchical Clustering results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=hierarchical_labels, cmap='coolwarm', edgecolors='k')
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

# Generate and plot dendrogram for Hierarchical Clustering
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

---

## 🤝 Contributions

🎯 We welcome contributions! Feel free to **submit pull requests** or **open issues** for improvements.

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

