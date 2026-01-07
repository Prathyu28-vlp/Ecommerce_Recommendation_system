import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    def __init__(self, user_item_matrix, user_mapping, item_mapping):
        self.user_item_matrix = user_item_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        
    def user_based_cf(self, user_id, n_recommendations=10, n_neighbors=50):
        """User-based collaborative filtering"""
        try:
            if user_id not in self.user_mapping:
                return []
                
            user_idx = self.user_mapping[user_id]
            
            # Calculate user similarities
            user_similarities = cosine_similarity(self.user_item_matrix)
            user_sim_scores = user_similarities[user_idx]
            
            # Find similar users
            similar_users = np.argsort(user_sim_scores)[::-1][1:n_neighbors+1]
            
            # Get recommendations
            user_ratings = self.user_item_matrix[user_idx]
            recommendations = np.zeros(len(user_ratings))
            
            for similar_user in similar_users:
                similarity = user_sim_scores[similar_user]
                similar_user_ratings = self.user_item_matrix[similar_user]
                
                # Only recommend items not rated by target user
                mask = (user_ratings == 0) & (similar_user_ratings > 0)
                recommendations += mask * similar_user_ratings * similarity
            
            # Get top recommendations
            unrated_items = np.where(user_ratings == 0)[0]
            item_scores = [(self.reverse_item_mapping[item], recommendations[item]) 
                          for item in unrated_items if recommendations[item] > 0]
            
            return sorted(item_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            print(f"Error in user-based CF: {str(e)}")
            return []
    
    def item_based_cf(self, user_id, n_recommendations=10, n_neighbors=50):
        """Item-based collaborative filtering"""
        try:
            if user_id not in self.user_mapping:
                return []
                
            user_idx = self.user_mapping[user_id]
            
            # Calculate item similarities
            item_similarities = cosine_similarity(self.user_item_matrix.T)
            
            # Get user's rated items
            user_ratings = self.user_item_matrix[user_idx]
            rated_items = np.where(user_ratings > 0)[0]
            
            recommendations = {}
            
            for item_idx in range(len(self.item_mapping)):
                if user_ratings[item_idx] == 0:  # Unrated item
                    score = 0
                    similarity_sum = 0
                    
                    # Find similar items that user has rated
                    item_sim_scores = item_similarities[item_idx]
                    similar_items = np.argsort(item_sim_scores)[::-1][:n_neighbors]
                    
                    for similar_item in similar_items:
                        if similar_item in rated_items:
                            similarity = item_sim_scores[similar_item]
                            rating = user_ratings[similar_item]
                            score += similarity * rating
                            similarity_sum += abs(similarity)
                    
                    if similarity_sum > 0:
                        recommendations[self.reverse_item_mapping[item_idx]] = score / similarity_sum
            
            return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            print(f"Error in item-based CF: {str(e)}")
            return []
    
    def matrix_factorization_svd(self, user_id, n_recommendations=10, n_components=50):
        """Matrix factorization using SVD"""
        try:
            if user_id not in self.user_mapping:
                return []
                
            user_idx = self.user_mapping[user_id]
            
            # Apply SVD
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd.fit_transform(self.user_item_matrix)
            item_factors = svd.components_.T
            
            # Predict ratings
            predicted_ratings = np.dot(user_factors[user_idx], item_factors.T)
            
            # Get unrated items
            user_ratings = self.user_item_matrix[user_idx]
            unrated_items = np.where(user_ratings == 0)[0]
            
            item_scores = [(self.reverse_item_mapping[item], predicted_ratings[item]) 
                          for item in unrated_items]
            
            return sorted(item_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            print(f"Error in SVD: {str(e)}")
            return []
    
    def matrix_factorization_nmf(self, user_id, n_recommendations=10, n_components=50):
        """Matrix factorization using Non-negative Matrix Factorization"""
        try:
            if user_id not in self.user_mapping:
                return []
                
            user_idx = self.user_mapping[user_id]
            
            # Apply NMF
            nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
            user_factors = nmf.fit_transform(self.user_item_matrix)
            item_factors = nmf.components_.T
            
            # Predict ratings
            predicted_ratings = np.dot(user_factors[user_idx], item_factors.T)
            
            # Get unrated items
            user_ratings = self.user_item_matrix[user_idx]
            unrated_items = np.where(user_ratings == 0)[0]
            
            item_scores = [(self.reverse_item_mapping[item], predicted_ratings[item]) 
                          for item in unrated_items]
            
            return sorted(item_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            print(f"Error in NMF: {str(e)}")
            return []