import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs

# Membuat dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Membuat dendrogram
plt.figure(figsize=(10, 5))
sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram untuk Hierarchical Clustering')
('Euclidean Distance')
plt.show()