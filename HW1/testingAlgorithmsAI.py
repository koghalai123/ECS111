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
    Z = linkage(normalizedData, method=method)
    labels = fcluster(Z, k, criterion='maxclust')
    return labels

# Gaussian Mixture Model implementation
def gmmClustering(normalizedData, k=3, maxIter=100):
    gmm = GaussianMixture(n_components=k, max_iter=maxIter)
    labels = gmm.fit_predict(normalizedData)
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
    hc_labels = hierarchicalClustering(normalizedData, k=k)
    plot_clusters(data, hc_labels, "Hierarchical Clustering", axes[1, 0])
    
    # Gaussian Mixture Model clustering
    gmm_labels = gmmClustering(normalizedData, k=k)
    plot_clusters(data, gmm_labels, "Gaussian Mixture Model", axes[1, 1])
    
    plt.tight_layout()
    plt.show()