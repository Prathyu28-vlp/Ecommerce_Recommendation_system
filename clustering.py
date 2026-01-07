"""
clustering.py
-------------------------------------
Performs user or item clustering using KMeans
for a recommendation system dataset.

Author: <Your Name>
Date: <Date>
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class ClusteringModel:
    """
    Clustering class for grouping users or items
    based on their features (e.g., ratings or embeddings).
    """

    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize clustering model.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.data_scaled = None
        self.labels = None

    def fit(self, data: pd.DataFrame, scale_data=True):
        """
        Fit KMeans to data.
        """
        if scale_data:
            self.data_scaled = self.scaler.fit_transform(data)
        else:
            self.data_scaled = data

        print(f"Training KMeans with {self.n_clusters} clusters...")
        self.labels = self.model.fit_predict(self.data_scaled)
        return self.labels

    def evaluate(self):
        """
        Evaluate clustering using silhouette score.
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before evaluation.")

        score = silhouette_score(self.data_scaled, self.labels)
        print(f"Silhouette Score: {score:.4f}")
        return score

    def visualize_clusters(self, data: pd.DataFrame, feature_x: str, feature_y: str, title="Cluster Visualization"):
        """
        Visualize clusters using 2D scatter plot.
        """
        if self.labels is None:
            raise ValueError("Please run fit() before visualizing clusters.")

        df_vis = data.copy()
        df_vis['Cluster'] = self.labels

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_vis, x=feature_x, y=feature_y, hue='Cluster', palette='tab10', s=60)
        plt.title(title)
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.show()

    def save_clusters(self, data: pd.DataFrame, output_file="clustered_data.csv"):
        """
        Save cluster assignments to CSV.
        """
        clustered_df = data.copy()
        clustered_df["Cluster"] = self.labels
        clustered_df.to_csv(output_file, index=False)
        print(f"Clustered data saved to '{output_file}'.")


# ----------------------------- USAGE EXAMPLE -----------------------------
if __name__ == "__main__":
    # Example with sample dataset
    df = pd.read_csv("rating_short.csv")  # Replace with your dataset
    print("Data loaded successfully.")

    # Example: Cluster users based on average rating behavior
    user_features = df.groupby("user_id")["rating"].agg(["mean", "count", "std"]).fillna(0)

    clustering = ClusteringModel(n_clusters=5)
    labels = clustering.fit(user_features)
    clustering.evaluate()
    clustering.visualize_clusters(user_features.reset_index(), "mean", "count", "User Clusters")
    clustering.save_clusters(user_features.reset_index(), "user_clusters.csv")
