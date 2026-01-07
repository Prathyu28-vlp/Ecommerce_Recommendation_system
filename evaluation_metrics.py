import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    def __init__(self, df, user_mapping, item_mapping):
        self.df = df
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.train_df = None
        self.test_df = None
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )
        return self.train_df, self.test_df
    
    def calculate_rmse(self, predictions, actuals):
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(actuals, predictions))
    
    def calculate_mae(self, predictions, actuals):
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(actuals, predictions)
    
    def precision_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Precision@K"""
        if k == 0:
            return 0
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_recommended / min(k, len(recommended_k))
    
    def recall_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Recall@K"""
        if len(relevant_items) == 0:
            return 0
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate F1@K"""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def coverage(self, all_recommendations, total_items):
        """Calculate catalog coverage"""
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update([item for item, _ in recs])
        return len(unique_recommended) / total_items
    
    def diversity(self, recommendations):
        """Calculate intra-list diversity (simplified version)"""
        if len(recommendations) <= 1:
            return 0
        
        # Simple diversity based on item ID differences
        items = [item for item, _ in recommendations]
        diversity_score = 0
        count = 0
        
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                # Simple string-based diversity
                diversity_score += 1 if items[i] != items[j] else 0
                count += 1
        
        return diversity_score / count if count > 0 else 0
    
    def evaluate_model(self, model, model_name, test_users=None, k=10):
        """Evaluate a recommendation model"""
        if test_users is None:
            test_users = self.test_df['user_id'].unique()[:50]  # Sample for efficiency
        
        metrics = {
            'model_name': model_name,
            'precision_at_k': [],
            'recall_at_k': [],
            'f1_at_k': [],
            'diversity_scores': [],
            'coverage': 0
        }
        
        all_recommendations = []
        
        for user_id in test_users:
            try:
                # Get user's test items (relevant items)
                user_test_items = self.test_df[
                    (self.test_df['user_id'] == user_id) & 
                    (self.test_df['rating'] >= 4)
                ]['item_id'].tolist()
                
                if not user_test_items:
                    continue
                
                # Get recommendations from model
                if hasattr(model, 'item_based_cf'):
                    recommendations = model.item_based_cf(user_id, k*2)
                elif hasattr(model, 'get_content_based_recommendations'):
                    recommendations = model.get_content_based_recommendations(user_id, k*2)
                elif hasattr(model, 'weighted_hybrid'):
                    recommendations = model.weighted_hybrid(user_id, k*2)
                else:
                    continue
                
                if not recommendations:
                    continue
                    
                recommended_items = [item for item, _ in recommendations]
                all_recommendations.append(recommendations)
                
                # Calculate metrics
                precision = self.precision_at_k(recommended_items, user_test_items, k)
                recall = self.recall_at_k(recommended_items, user_test_items, k)
                f1 = self.f1_at_k(recommended_items, user_test_items, k)
                diversity = self.diversity(recommendations[:k])
                
                metrics['precision_at_k'].append(precision)
                metrics['recall_at_k'].append(recall)
                metrics['f1_at_k'].append(f1)
                metrics['diversity_scores'].append(diversity)
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {str(e)}")
                continue
        
        # Calculate average metrics
        if metrics['precision_at_k']:
            metrics['avg_precision'] = np.mean(metrics['precision_at_k'])
            metrics['avg_recall'] = np.mean(metrics['recall_at_k'])
            metrics['avg_f1'] = np.mean(metrics['f1_at_k'])
            metrics['avg_diversity'] = np.mean(metrics['diversity_scores'])
            metrics['coverage'] = self.coverage(all_recommendations, len(self.item_mapping))
        else:
            metrics['avg_precision'] = 0
            metrics['avg_recall'] = 0
            metrics['avg_f1'] = 0
            metrics['avg_diversity'] = 0
            metrics['coverage'] = 0
        
        return metrics
    
    def compare_models(self, models_dict, test_users=None, k=10):
        """Compare multiple models"""
        results = []
        
        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, model_name, test_users, k)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, results_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision comparison
        axes[0, 0].bar(results_df['model_name'], results_df['avg_precision'])
        axes[0, 0].set_title('Average Precision@10')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        axes[0, 1].bar(results_df['model_name'], results_df['avg_recall'])
        axes[0, 1].set_title('Average Recall@10')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 comparison
        axes[1, 0].bar(results_df['model_name'], results_df['avg_f1'])
        axes[1, 0].set_title('Average F1@10')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Coverage vs Diversity
        axes[1, 1].scatter(results_df['coverage'], results_df['avg_diversity'])
        for i, model in enumerate(results_df['model_name']):
            axes[1, 1].annotate(model, (results_df['coverage'].iloc[i], 
                                       results_df['avg_diversity'].iloc[i]))
        axes[1, 1].set_xlabel('Coverage')
        axes[1, 1].set_ylabel('Average Diversity')
        axes[1, 1].set_title('Coverage vs Diversity')
        
        plt.tight_layout()
        return fig