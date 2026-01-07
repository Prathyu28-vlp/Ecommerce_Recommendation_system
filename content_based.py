import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class ContentBasedFiltering:
    def __init__(self, user_item_matrix, user_mapping, item_mapping, df):
        self.user_item_matrix = user_item_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        self.df = df
        self.item_profiles = None
        self.user_profiles = None
        
    def create_item_profiles(self):
        """Create item profiles based on user ratings patterns"""
        try:
            # Create item features based on rating statistics
            item_features = []
            
            for item_idx in range(len(self.item_mapping)):
                item_id = self.reverse_item_mapping[item_idx]
                item_ratings = self.df[self.df['item_id'] == item_id]['rating']
                
                if len(item_ratings) > 0:
                    features = [
                        item_ratings.mean(),           # Average rating
                        item_ratings.std(),            # Rating variance
                        len(item_ratings),             # Number of ratings (popularity)
                        (item_ratings >= 4).sum(),     # Number of high ratings
                        (item_ratings <= 2).sum(),     # Number of low ratings
                    ]
                else:
                    features = [0, 0, 0, 0, 0]
                    
                item_features.append(features)
            
            self.item_profiles = np.array(item_features)
            
            # Normalize features
            scaler = StandardScaler()
            self.item_profiles = scaler.fit_transform(self.item_profiles)
            
            return True
            
        except Exception as e:
            print(f"Error creating item profiles: {str(e)}")
            return False
    
    def create_user_profiles(self):
        """Create user profiles based on their rating patterns"""
        try:
            user_features = []
            
            for user_idx in range(len(self.user_mapping)):
                user_id = self.reverse_user_mapping[user_idx]
                user_ratings = self.df[self.df['user_id'] == user_id]['rating']
                
                if len(user_ratings) > 0:
                    features = [
                        user_ratings.mean(),           # Average rating given
                        user_ratings.std(),            # Rating variance
                        len(user_ratings),             # Number of ratings given
                        (user_ratings >= 4).sum(),     # Number of high ratings given
                        (user_ratings <= 2).sum(),     # Number of low ratings given
                    ]
                else:
                    features = [0, 0, 0, 0, 0]
                    
                user_features.append(features)
            
            self.user_profiles = np.array(user_features)
            
            # Normalize features
            scaler = StandardScaler()
            self.user_profiles = scaler.fit_transform(self.user_profiles)
            
            return True
            
        except Exception as e:
            print(f"Error creating user profiles: {str(e)}")
            return False
    
    def get_content_based_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations based on content similarity"""
        try:
            if user_id not in self.user_mapping:
                return []
                
            if self.item_profiles is None:
                self.create_item_profiles()
            if self.user_profiles is None:
                self.create_user_profiles()
                
            user_idx = self.user_mapping[user_id]
            
            # Get user's rated items
            user_ratings = self.user_item_matrix[user_idx]
            rated_items = np.where(user_ratings > 0)[0]
            
            if len(rated_items) == 0:
                return []
            
            # Create user preference profile based on liked items
            liked_items = [item for item in rated_items if user_ratings[item] >= 4]
            
            if len(liked_items) == 0:
                liked_items = rated_items  # Use all rated items if no high ratings
            
            # Average profile of liked items
            user_preference_profile = np.mean(self.item_profiles[liked_items], axis=0)
            
            # Calculate similarity with all items
            similarities = cosine_similarity([user_preference_profile], self.item_profiles)[0]
            
            # Get unrated items
            unrated_items = np.where(user_ratings == 0)[0]
            
            item_scores = [(self.reverse_item_mapping[item], similarities[item]) 
                          for item in unrated_items]
            
            return sorted(item_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            print(f"Error in content-based filtering: {str(e)}")
            return []
    
    def get_item_similarity_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations based on item-to-item content similarity"""
        try:
            if user_id not in self.user_mapping:
                return []
                
            if self.item_profiles is None:
                self.create_item_profiles()
                
            user_idx = self.user_mapping[user_id]
            user_ratings = self.user_item_matrix[user_idx]
            
            # Calculate item similarities
            item_similarities = cosine_similarity(self.item_profiles)
            
            recommendations = {}
            
            # For each unrated item
            for item_idx in range(len(self.item_mapping)):
                if user_ratings[item_idx] == 0:  # Unrated item
                    score = 0
                    weight_sum = 0
                    
                    # Find similar items that user has rated
                    for rated_item_idx in np.where(user_ratings > 0)[0]:
                        similarity = item_similarities[item_idx][rated_item_idx]
                        rating = user_ratings[rated_item_idx]
                        
                        score += similarity * rating
                        weight_sum += abs(similarity)
                    
                    if weight_sum > 0:
                        recommendations[self.reverse_item_mapping[item_idx]] = score / weight_sum
            
            return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            print(f"Error in item similarity recommendations: {str(e)}")
            return []