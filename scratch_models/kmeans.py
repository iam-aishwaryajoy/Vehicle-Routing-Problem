import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

class Kmeans:

    def initialize_centroids(self, points, k):
        """Selects k random points as initial points"""
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:k]

    def closest_centroid(self, points, centroids):
        """Finds the closest centroid for all points"""
        
        centroids_reshape = centroids[:, np.newaxis]
        distances = np.sqrt(((points - centroids_reshape)**2).sum(axis=2)) #len(distance) = points_number * cluster
        return np.argmin(distances, axis=0) #Min distance to cluster

    def update_centroids(self, points, closest, centroids):
        """Updates the centroids to be the mean of points assigned to it"""
        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
        

    def compute(self, points, k, max_iters=100):
        centroids = self.initialize_centroids(points, k)
        for i in range(max_iters):
            closest = self.closest_centroid(points, centroids)
            new_centroids = self.update_centroids(points, closest, centroids)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return closest, centroids

    def plot_kmeans(self, cluster_labels, centroids, file_name):
        for i in range(centroids.shape[0]):  # Loop over each cluster
            # Plot points in the same cluster
            plt.scatter(points[cluster_labels == i, 0], points[cluster_labels == i, 1], label=f'Cluster {i}') #Point (x,y) in cluster 1
            
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, color='red', marker='X', label='Centroids')  #Centroids's x and y axis

        plt.title('K-Means Clustering')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.savefig(file_name)
        plt.close()

# Example usage:

df = pd.DataFrame(np.random.rand(100, 6), columns=['Steering Wheel', 'Distance', 'Torque', 'Speed', 'Yawrate', 'EPS Angle'])
points = df.to_numpy()
k = 3  # Number of clusters

pca = PCA(n_components=2)
points_reduced = pca.fit_transform(points)

#Scratch Model:
kmeans = Kmeans()
cluster_labels, centroids = kmeans.compute(points_reduced, k)
print("Cluster labels:", cluster_labels)
print("Centroids:", centroids)
kmeans.plot_kmeans(cluster_labels=cluster_labels, centroids=centroids, file_name='Scratch_kmeans.png')

#Compare real Kmeans:
kmeans_sklearn = KMeans(n_clusters=3, random_state=0).fit(points_reduced)
labels_sklearn = kmeans_sklearn.labels_
centroids_sklearn = kmeans_sklearn.cluster_centers_
distances = cdist(centroids_sklearn, centroids, 'euclidean')
print('Distance', distances)
ari_score = adjusted_rand_score(cluster_labels, labels_sklearn)
print(f"Adjusted Rand Index: {ari_score}")
print("Cluster labels:", labels_sklearn)
print("Centroids:", centroids_sklearn)
kmeans.plot_kmeans(cluster_labels=labels_sklearn, centroids=centroids_sklearn, file_name='Sklearn Kmeans.png')