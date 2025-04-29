#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:07:55 2025

@author: koghalai
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal

# Load and normalize data
data = np.loadtxt("clusterData_test_unlabeled.dat")

def normalizeData(data, method='none'):
    if method == 'none':
        normalizedData = data
    elif method == 'range':
        maxVal = np.max(data, axis=0)
        minVal = np.min(data, axis=0)
        normalizedData = (data - minVal) / (maxVal - minVal)
    elif method == 'z-score':
        normalizedData = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return normalizedData

# Corrected K-means++ implementation
def kMeansPlusPlus(normalizedData, k=1, maxIter=50):
    rng = np.random.default_rng()
    # Initialize first centroid randomly (fixed missing bracket)
    centroids = [normalizedData[rng.integers(normalizedData.shape[0])]]
    
    # Select remaining centroids using D^2 weighting
    for _ in range(1, k):
        # Fixed list comprehension syntax
        distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in normalizedData])
        probs = distances / distances.sum()
        cumulative_probs = probs.cumsum()
        r = rng.random()
        new_centroid_idx = np.where(cumulative_probs >= r)[0][0]
        centroids.append(normalizedData[new_centroid_idx])
    
    centroids = np.array(centroids)
    
    # Regular K-means algorithm
    for _ in range(maxIter):
        # Assign points to nearest centroid
        labels, _ = pairwise_distances_argmin_min(normalizedData, centroids)
        
        # Update centroids
        new_centroids = np.array([normalizedData[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    # Get final labels
    labels, _ = pairwise_distances_argmin_min(normalizedData, centroids)
    return centroids, labels

# Hierarchical clustering implementation


def hierarchicalClustering(normalizedData, method='ward', k=3):
    """
    From-scratch implementation of agglomerative hierarchical clustering
    """
    n = len(normalizedData)
    if n < k:
        raise ValueError("Number of clusters cannot be greater than number of data points")
    
    # Initialize: each point is its own cluster
    clusters = [[i] for i in range(n)]
    distances = squareform(pdist(normalizedData))
    np.fill_diagonal(distances, np.inf)  # ignore self-distances
    
    # Store cluster distances in a dictionary for efficient lookup
    cluster_distances = {}
    for i in range(n):
        for j in range(i+1, n):
            cluster_distances[(i,j)] = distances[i,j]
    
    while len(clusters) > k:
        # Find the two closest clusters
        min_dist = np.inf
        merge_indices = (0, 1)
        
        # Check all pairs of clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                # Get all pairwise distances between clusters i and j
                dists = []
                for a in clusters[i]:
                    for b in clusters[j]:
                        if a < b:
                            dists.append(cluster_distances[(a,b)])
                        else:
                            dists.append(cluster_distances[(b,a)])
                
                # Compute linkage criterion
                if method == 'single':
                    current_dist = min(dists)
                elif method == 'complete':
                    current_dist = max(dists)
                elif method == 'average':
                    current_dist = sum(dists)/len(dists)
                elif method == 'ward':
                    # Ward's method: minimize variance after merging
                    merged = np.vstack([normalizedData[clusters[i]], normalizedData[clusters[j]]])
                    current_dist = np.sum((merged - np.mean(merged, axis=0))**2)
                
                if current_dist < min_dist:
                    min_dist = current_dist
                    merge_indices = (i, j)
        
        # Merge the two closest clusters
        i, j = merge_indices
        clusters[i].extend(clusters[j])
        del clusters[j]
        
        # Update distance dictionary for the new cluster
        for c in range(len(clusters)):
            if c != i:
                # Get all pairwise distances between new cluster i and cluster c
                dists = []
                for a in clusters[i]:
                    for b in clusters[c]:
                        if a < b:
                            dists.append(cluster_distances[(a,b)])
                        else:
                            dists.append(cluster_distances[(b,a)])
                
                # Update with new linkage distance
                if method == 'single':
                    new_dist = min(dists)
                elif method == 'complete':
                    new_dist = max(dists)
                elif method == 'average':
                    new_dist = sum(dists)/len(dists)
                elif method == 'ward':
                    merged = np.vstack([normalizedData[clusters[i]], normalizedData[clusters[c]]])
                    new_dist = np.sum((merged - np.mean(merged, axis=0))**2)
                
                cluster_distances[(min(i,c), max(i,c))] = new_dist
    
    # Create labels
    labels = np.zeros(n, dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels[point_idx] = cluster_idx
    
    return labels



def gmmClustering(normalizedData, k=3, maxIter=100, tol=1e-4):
    """
    From-scratch implementation of Gaussian Mixture Model using EM algorithm
    """
    n, d = normalizedData.shape
    
    # 1. Initialization
    # Randomly assign cluster responsibilities
    responsibilities = np.random.rand(n, k)
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
    
    # Initialize parameters
    weights = np.ones(k) / k  # Uniform weights
    means = np.array([normalizedData[np.random.choice(n)] for _ in range(k)])
    covariances = np.array([np.cov(normalizedData.T) for _ in range(k)])
    
    log_likelihood = -np.inf
    converged = False
    
    for iteration in range(maxIter):
        # 2. Expectation Step: Update responsibilities
        for j in range(k):
            responsibilities[:, j] = weights[j] * multivariate_normal.pdf(
                normalizedData, mean=means[j], cov=covariances[j])
        
        # Normalize responsibilities
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        
        # 3. Maximization Step: Update parameters
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n
        
        for j in range(k):
            # Update means
            means[j] = np.sum(responsibilities[:, j:j+1] * normalizedData, axis=0) / Nk[j]
            
            # Update covariances
            diff = normalizedData - means[j]
            covariances[j] = (responsibilities[:, j] * diff.T) @ diff / Nk[j]
            # Add small value to diagonal for numerical stability
            covariances[j] += 1e-6 * np.eye(d)
        
        # 4. Check convergence
        new_log_likelihood = 0
        for j in range(k):
            new_log_likelihood += weights[j] * multivariate_normal.pdf(
                normalizedData, mean=means[j], cov=covariances[j])
        new_log_likelihood = np.sum(np.log(new_log_likelihood))
        
        if np.abs(new_log_likelihood - log_likelihood) < tol:
            converged = True
            break
        log_likelihood = new_log_likelihood
    
    # Assign final labels
    labels = np.argmax(responsibilities, axis=1)
    return labels


# Plotting function
def plot_clusters(data, labels, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = matplotlib.colormaps.get_cmap('tab10')
    unique_labels = set(labels)
    
    for label in unique_labels:
        color = cmap(label / max(1, max(unique_labels)))
        cluster_points = data[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=color, label=f'Cluster {label}')
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    
    if ax is None:
        plt.show()

# Main execution
if __name__ == "__main__":
    # Normalize data
    normalizedData = normalizeData(data, method='none')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original data
    axes[0, 0].scatter(data[:, 0], data[:, 1], color='blue', s=10)
    axes[0, 0].set_title("Original Data")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    
    # K-means++ clustering
    k = 6
    centroids, kpp_labels = kMeansPlusPlus(normalizedData, k=k)
    plot_clusters(data, kpp_labels, "K-means++ Clustering", axes[0, 1])
    
    # Hierarchical clustering
    #hc_labels = hierarchicalClustering(normalizedData, k=k)
    #plot_clusters(data, hc_labels, "Hierarchical Clustering", axes[1, 0])
    
    # Gaussian Mixture Model clustering
    gmm_labels = gmmClustering(normalizedData, k=k)
    plot_clusters(data, gmm_labels, "Gaussian Mixture Model", axes[1, 1])
    
    plt.tight_layout()
    
    
    
    # Add this after your existing code, before the plt.show() in the main execution

    # Create comparison figure with premade algorithms
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original data again for reference
    axes2[0, 0].scatter(data[:, 0], data[:, 1], color='blue', s=10)
    axes2[0, 0].set_title("Original Data (Reference)")
    axes2[0, 0].set_xlabel("X")
    axes2[0, 0].set_ylabel("Y")
    
    # Premade K-means++ from sklearn
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    premade_kpp_labels = kmeans.fit_predict(normalizedData)
    plot_clusters(data, premade_kpp_labels, "Premade K-means++ (sklearn)", axes2[0, 1])
    
    # Premade Hierarchical clustering from scipy
    premade_hc_labels = fcluster(linkage(normalizedData, method='ward'), k, criterion='maxclust') - 1
    plot_clusters(data, premade_hc_labels, "Premade Hierarchical (scipy)", axes2[1, 0])
    
    # Premade GMM from sklearn
    premade_gmm = GaussianMixture(n_components=6, random_state=42)
    premade_gmm_labels = premade_gmm.fit_predict(normalizedData)
    plot_clusters(data, premade_gmm_labels, "Premade GMM (sklearn)", axes2[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    
    
    plt.show()