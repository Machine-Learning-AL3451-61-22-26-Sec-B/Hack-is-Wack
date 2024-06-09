from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
dataset = load_iris()
X = dataset.data
y = dataset.target

# Plot setup
plt.figure(figsize=(14, 7))
colormap = ['red', 'lime', 'black']

# Plotting real data
plt.subplot(1, 3, 1)
plt.scatter(X[:, 2], X[:, 3], c=colormap[y], s=40)
plt.title('Real')

# Clustering using K-Means
plt.subplot(1, 3, 2)
kmeans = KMeans(n_clusters=3)
pred_y_kmeans = kmeans.fit_predict(X)
centroids_kmeans = kmeans.cluster_centers_
plt.scatter(X[:, 2], X[:, 3], c=colormap[pred_y_kmeans], s=40)
plt.scatter(centroids_kmeans[:, 2], centroids_kmeans[:, 3], marker='*', s=200, c='orange', label='Centroids')
plt.title('KMeans')

# Clustering using Gaussian Mixture Model
plt.subplot(1, 3, 3)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3)
pred_y_gmm = gmm.fit_predict(X_scaled)
plt.scatter(X[:, 2], X[:, 3], c=colormap[pred_y_gmm], s=40)
plt.title('GMM Classification')

plt.show()
