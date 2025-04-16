import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

data = np.loadtxt("clusterData_test_unlabeled.dat")

def normalizeData(data, method='none'):
    if method == 'none' :
        normalizedData=data
    elif method == 'range' :
        maxVal = np.max(data,axis=0)
        minVal = np.min(data,axis=0)
        normalizedData = (data-minVal)/(maxVal-minVal)
    elif method == 'z-score' :
        normalizedData=(data-np.mean(data,axis=0))/np.std(data,axis=0)
    return normalizedData 

def kMeans(normalizedData,k=1,maxIter=50):
    rng = np.random.default_rng()
    pointsNormalized = rng.random((k,2))
    maxVal = np.max(normalizedData,axis=0)
    minVal = np.min(normalizedData,axis=0)
    centroids = pointsNormalized*(maxVal-minVal)+minVal
    distanceMat = np.zeros((np.size(normalizedData,axis=0),k))
    for i in range(maxIter):
        for j in range(k):
            distanceMat[:,j] = np.sqrt(np.sum(np.square(normalizedData-centroids[j,:]),axis=1))
        clusterInds = np.argmin(distanceMat,axis=1)
        for j in range(k):
            centroids[j,:] = np.mean(normalizedData[np.where(clusterInds==j),:],axis=1)
    distanceMat[:,j] = np.sqrt(np.sum(np.square(normalizedData-centroids[j,:]),axis=1))
    clusterInds = np.argmin(distanceMat,axis=1)
    return centroids, clusterInds
            
normalizedData=normalizeData(data, method='none')

k=6
centroids, clusterInds = kMeans(normalizedData,k,maxIter=10)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot initial data
axes[0, 0].scatter(data[:, 0], data[:, 1], color='blue', label='Initial Data')
axes[0, 0].set_title("Initial Data")
axes[0, 0].set_xlabel("X")
axes[0, 0].set_ylabel("Y")
axes[0, 0].legend()
axes[0, 0].set_aspect('equal')

cmap = get_cmap('tab10')
colors = [cmap(i / k) for i in range(k)]
for i in range(k):
    plotData = data[clusterInds==i]
    axes[0, 1].scatter(plotData[:, 0], plotData[:, 1], color=colors[i], label=f'Cluster: {i+1}')

axes[0, 1].set_title("Clustered Data(Not normalized)")
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y")
axes[0, 1].legend()




normalizedData=normalizeData(data, method='range')
centroids, clusterInds = kMeans(normalizedData,k,maxIter=10)
cmap = get_cmap('tab10')
colors = [cmap(i / k) for i in range(k)]
for i in range(k):
    plotData = normalizedData[clusterInds==i]
    axes[1, 0].scatter(plotData[:, 0], plotData[:, 1], color=colors[i], label=f'Cluster: {i+1}')

axes[1, 0].set_title("Clustered Data(Normalized by range)")
axes[1, 0].set_xlabel("X")
axes[1, 0].set_ylabel("Y")
axes[1, 0].legend()
axes[1, 0].set_aspect('equal')





normalizedData=normalizeData(data, method='z-score')
centroids, clusterInds = kMeans(normalizedData,k,maxIter=10)
cmap = get_cmap('tab10')
colors = [cmap(i / k) for i in range(k)]
for i in range(k):
    plotData = normalizedData[clusterInds==i]
    axes[1, 1].scatter(plotData[:, 0], plotData[:, 1], color=colors[i], label=f'Cluster: {i+1}')

axes[1, 1].set_title("Clustered Data(Normalized by z-score)")
axes[1, 1].set_xlabel("X")
axes[1, 1].set_ylabel("Y")
axes[1, 1].legend()
axes[1, 1].set_aspect('equal')





