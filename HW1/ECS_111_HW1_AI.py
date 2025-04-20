import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans, DBSCAN

# Step 1: Load the data
data = np.loadtxt("clusterData_test_unlabeled.dat")
# data = np.loadtxt("clusterData_train.dat")

# Step 2: Define normalization methods
def z_score_normalization(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    return (data - data_mean) / data_std

def absolute_distance_normalization(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min) / (data_max - data_min)

# Step 3: Homemade K-means clustering
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

# Step 4: Homemade DBSCAN
def dbscan(data, eps, min_samples):
    n = data.shape[0]
    labels = np.full(n, 0)  # Initialize all points as noise (-1)
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
        if labels[point_idx] != 0:  # Already processed
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors)

    return labels

# Step 5: Perform clustering with homemade K-means
k = 6  # Number of clusters
labels_homemade_kmeans, centroids_homemade_kmeans = kmeans(data, k)

# Step 6: Perform clustering with premade K-means
kmeans_model = KMeans(n_clusters=k, random_state=42)
labels_premade_kmeans = kmeans_model.fit_predict(data)
centroids_premade_kmeans = kmeans_model.cluster_centers_

# Step 7: Perform clustering with homemade DBSCAN
eps = 15  # Maximum distance for neighbors
min_samples = 4  # Minimum number of points to form a cluster
labels_homemade_dbscan = dbscan(data, eps, min_samples)

# Step 8: Perform clustering with premade DBSCAN
dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
labels_premade_dbscan = dbscan_model.fit_predict(data)

# Step 9: Plot results for comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot homemade K-means
cmap = get_cmap('tab10')
colors = [cmap(i / k) for i in range(k)]
for i in range(k):
    cluster_points = data[labels_homemade_kmeans == i]
    axes[0, 0].scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i], label=f'Cluster {i+1}')
axes[0, 0].scatter(centroids_homemade_kmeans[:, 0], centroids_homemade_kmeans[:, 1], s=100, color='black', marker='X', label='Centroids')
axes[0, 0].set_title("Homemade K-means")
axes[0, 0].legend()

# Plot premade K-means
for i in range(k):
    cluster_points = data[labels_premade_kmeans == i]
    axes[0, 1].scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i], label=f'Cluster {i+1}')
axes[0, 1].scatter(centroids_premade_kmeans[:, 0], centroids_premade_kmeans[:, 1], s=100, color='black', marker='X', label='Centroids')
axes[0, 1].set_title("Premade K-means")
axes[0, 1].legend()

# Plot homemade DBSCAN
unique_labels = set(labels_homemade_dbscan)
for label in unique_labels:
    if label == -1:
        color = 'black'  # Noise points
        label_name = 'Noise'
    else:
        color = cmap(label / max(unique_labels))
        label_name = f'Cluster {label}'
    cluster_points = data[labels_homemade_dbscan == label]
    axes[1, 0].scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=color, label=label_name)
axes[1, 0].set_title("Homemade DBSCAN")
axes[1, 0].legend()

# Plot premade DBSCAN
unique_labels = set(labels_premade_dbscan)
for label in unique_labels:
    if label == -1:
        color = 'black'  # Noise points
        label_name = 'Noise'
    else:
        color = cmap(label / max(unique_labels))
        label_name = f'Cluster {label}'
    cluster_points = data[labels_premade_dbscan == label]
    axes[1, 1].scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=color, label=label_name)
axes[1, 1].set_title("Premade DBSCAN")
axes[1, 1].legend()

plt.tight_layout()
plt.show()