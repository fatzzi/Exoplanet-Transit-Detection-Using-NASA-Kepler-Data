import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.inertia = 0 # measures cluster tightness

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # drop random initial centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # assign points to nearest centroid
            labels = self._assign_clusters(X)
            
            # move centroids to mean
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    # replace empty centroids
                    new_centroids[i] = X[np.random.choice(n_samples)]
            
            # check convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids

        # calculate final inertia
        self.inertia = 0
        final_labels = self._assign_clusters(X)
        for i in range(self.k):
            cluster_points = X[final_labels == i]
            if len(cluster_points) > 0:
                self.inertia += np.sum((cluster_points - self.centroids[i]) ** 2)

    def predict(self, X):
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        # get euclidean distance
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        
        # return closest centroid index
        return np.argmin(distances, axis=1)

# test block and elbow method
if __name__ == "__main__":
    print("Loading data...")
    # unsupervised so only X_train used
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy") # y_train just for checking at the end

    print("Running Elbow Method (this might take a few seconds)...")
    inertias = []
    K_range = range(1, 7) # test k=1 to k=6
    
    for k in K_range:
        kmeans = KMeans(k=k, max_iters=100)
        kmeans.fit(X_train)
        inertias.append(kmeans.inertia)
        print(f"K={k} calculated. Inertia: {kmeans.inertia:.2f}")

    # plot elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Lower is tighter)')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()

    # setting k=2 based on elbow
    best_k = 2 
    print(f"\nTraining final model with K={best_k}...")
    final_kmeans = KMeans(k=best_k)
    final_kmeans.fit(X_train)
    cluster_assignments = final_kmeans.predict(X_train)

    # compare clusters with actual labels
    print("\n--- Cluster Composition (Against NASA Labels) ---")
    for i in range(best_k):
        cluster_mask = (cluster_assignments == i)
        actual_labels_in_cluster = y_train[cluster_mask]
        
        # count planets vs false positives
        planets = np.sum(actual_labels_in_cluster == 0)
        false_positives = np.sum(actual_labels_in_cluster == 1)
        total = len(actual_labels_in_cluster)
        
        if total > 0:
            print(f"Cluster {i}: {total} total stars")
            print(f"  -> Confirmed Planets: {planets} ({planets/total*100:.1f}%)")
            print(f"  -> False Positives:   {false_positives} ({false_positives/total*100:.1f}%)")