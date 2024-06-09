from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.graph_objects as go

# Load the Iris dataset
dataset = load_iris()
X = dataset.data
y = dataset.target

# Plotting real data
real_fig = go.Figure(data=go.Scatter(x=X[:, 2], y=X[:, 3], mode='markers', marker=dict(color=y, colorscale='Viridis', size=10)))
real_fig.update_layout(title='Real Data')
real_fig.show()

# Clustering using K-Means
kmeans = KMeans(n_clusters=3)
pred_y_kmeans = kmeans.fit_predict(X)
centroids_kmeans = kmeans.cluster_centers_
kmeans_fig = go.Figure()
kmeans_fig.add_trace(go.Scatter(x=X[:, 2], y=X[:, 3], mode='markers', marker=dict(color=pred_y_kmeans, colorscale='Viridis', size=10), name='Clusters'))
kmeans_fig.add_trace(go.Scatter(x=centroids_kmeans[:, 2], y=centroids_kmeans[:, 3], mode='markers', marker=dict(symbol='star', color='orange', size=12), name='Centroids'))
kmeans_fig.update_layout(title='KMeans Clustering')
kmeans_fig.show()

# Clustering using Gaussian Mixture Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3)
pred_y_gmm = gmm.fit_predict(X_scaled)
gmm_fig = go.Figure(data=go.Scatter(x=X[:, 2], y=X[:, 3], mode='markers', marker=dict(color=pred_y_gmm, colorscale='Viridis', size=10)))
gmm_fig.update_layout(title='GMM Classification')
gmm_fig.show()
