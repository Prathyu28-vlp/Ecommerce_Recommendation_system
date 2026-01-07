import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import streamlit as st

class DataProcessor:
    def __init__(self, data_path='rating_short.csv'):
        self.data_path = data_path
        self.df = None
        self.user_item_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        
    def load_data(self):
        """Load and preprocess the rating data"""
        try:
            self.df = pd.read_csv(self.data_path)
            # Rename columns to standard format
            if 'userid' in self.df.columns:
                self.df = self.df.rename(columns={'userid': 'user_id', 'productid': 'item_id'})
            
            # Remove timestamp column if exists
            if 'date' in self.df.columns:
                self.df = self.df.drop('date', axis=1)
                
            # Create user and item mappings
            unique_users = self.df['user_id'].unique()
            unique_items = self.df['item_id'].unique()
            
            self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
            
            # Add mapped indices
            self.df['user_idx'] = self.df['user_id'].map(self.user_mapping)
            self.df['item_idx'] = self.df['item_id'].map(self.item_mapping)
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        if self.df is None:
            return None
            
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        # Create sparse matrix
        self.user_item_matrix = csr_matrix(
            (self.df['rating'], (self.df['user_idx'], self.df['item_idx'])),
            shape=(n_users, n_items)
        ).toarray()
        
        return self.user_item_matrix
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if self.df is None:
            return None, None
            
        train_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state, stratify=self.df['user_id']
        )
        return train_df, test_df
    
    def get_data_statistics(self):
        """Get basic statistics about the dataset"""
        if self.df is None:
            return {}
            
        stats = {
            'total_ratings': len(self.df),
            'unique_users': self.df['user_id'].nunique(),
            'unique_items': self.df['item_id'].nunique(),
            'rating_range': (self.df['rating'].min(), self.df['rating'].max()),
            'avg_rating': self.df['rating'].mean(),
            'sparsity': 1 - (len(self.df) / (self.df['user_id'].nunique() * self.df['item_id'].nunique()))
        }
        return stats
    
    def get_user_items(self, user_id):
        """Get items rated by a specific user"""
        if self.df is None:
            return []
        user_data = self.df[self.df['user_id'] == user_id]
        return user_data[['item_id', 'rating']].to_dict('records')
    
    def get_item_users(self, item_id):
        """Get users who rated a specific item"""
        if self.df is None:
            return []
        item_data = self.df[self.df['item_id'] == item_id]
        return item_data[['user_id', 'rating']].to_dict('records')