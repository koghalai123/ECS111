import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Step 1: Load the data
data = np.loadtxt("clusterData_test_unlabeled.dat")
#data = np.loadtxt("clusterData_train.dat")


# Step 2: Define normalization methods
def z_score_normalization(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    return (data - data_mean) / data_std

def absolute_distance_normalization(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min) / (data_max - data_min)

# Step 3: Implement K-means clustering
def kmeans(data, k, max_iters=100):
    np.random.seed(42)  # For reproducibility
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

# Step 4: Implement DBSCAN
def dbscan(data, eps, min_samples):
    n = data.shape[0]
    labels = np.full(n, -1)  # Initialize all points as noise (-1)
    cluster_id = 0

    def region_query(point_idx):
        distances = np.linalg.norm(data - data[point_idx], axis=1)
        return np.where(distances <= eps)[0]

    def expand_cluster(point_idx, neighbors):
        nonlocal cluster_id
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:  # Previously labeled as noise
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:  # Unvisited
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1

    for point_idx in range(n):
        if labels[point_idx] != -1:  # Already processed
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors)

    return labels

# Step 5: Perform clustering with different normalization methods
k = 6  # Number of clusters

# No normalization
data_no_norm = data
labels_no_norm, centroids_no_norm = kmeans(data_no_norm, k)

# Z-score normalization
data_z_score = z_score_normalization(data)
labels_z_score, centroids_z_score = kmeans(data_z_score, k)

# Absolute distance normalization
data_abs_norm = absolute_distance_normalization(data)
labels_abs_norm, centroids_abs_norm = kmeans(data_abs_norm, k)

# Step 6: Perform DBSCAN
eps = 70  # Maximum distance for neighbors
min_samples = 5  # Minimum number of points to form a cluster
labels_dbscan = dbscan(data, eps, min_samples)

# Step 7: Plot all results as subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot initial data
axes[0, 0].scatter(data[:, 0], data[:, 1], s=10, color='blue', label='Initial Data')
axes[0, 0].set_title("Initial Data")
axes[0, 0].set_xlabel("X")
axes[0, 0].set_ylabel("Y")
axes[0, 0].legend()

# Plot clustering with no normalization
cmap = get_cmap('tab10')
colors = [cmap(i / k) for i in range(k)]
for i in range(k):
    cluster_points = data_no_norm[labels_no_norm == i]
    axes[0, 1].scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i], label=f'Cluster {i+1}')
axes[0, 1].scatter(centroids_no_norm[:, 0], centroids_no_norm[:, 1], s=100, color='black', marker='X', label='Centroids')
axes[0, 1].set_title("Clustering (No Normalization)")
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y")
axes[0, 1].legend()

# Plot clustering with z-score normalization
for i in range(k):
    cluster_points = data_z_score[labels_z_score == i]
    axes[1, 0].scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i], label=f'Cluster {i+1}')
axes[1, 0].scatter(centroids_z_score[:, 0], centroids_z_score[:, 1], s=100, color='black', marker='X', label='Centroids')
axes[1, 0].set_title("Clustering (Z-score Normalization)")
axes[1, 0].set_xlabel("X")
axes[1, 0].set_ylabel("Y")
axes[1, 0].legend()

# Plot clustering with absolute distance normalization
for i in range(k):
    cluster_points = data_abs_norm[labels_abs_norm == i]
    axes[1, 1].scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i], label=f'Cluster {i+1}')
axes[1, 1].scatter(centroids_abs_norm[:, 0], centroids_abs_norm[:, 1], s=100, color='black', marker='X', label='Centroids')
axes[1, 1].set_title("Clustering (Absolute Distance Normalization)")
axes[1, 1].set_xlabel("X")
axes[1, 1].set_ylabel("Y")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Step 8: Plot DBSCAN results
fig, ax = plt.subplots(figsize=(8, 6))
unique_labels = set(labels_dbscan)
for label in unique_labels:
    if label == -1:
        color = 'black'  # Noise points
        label_name = 'Noise'
    else:
        color = cmap(label / max(unique_labels))
        label_name = f'Cluster {label}'
    cluster_points = data[labels_dbscan == label]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=color, label=label_name)

ax.set_title("DBSCAN Clustering")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.show()