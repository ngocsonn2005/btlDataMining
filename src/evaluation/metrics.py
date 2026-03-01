# -*- coding: utf-8 -*-
"""
Metrics Module
Handles calculation of evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, mean_squared_error,
                            mean_absolute_error, r2_score)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json


class MetricsCalculator:
    """
    Class for calculating evaluation metrics
    """
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, Any]:
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Per-class metrics
        labels = sorted(set(y_true) | set(y_pred))
        metrics['per_class'] = {}
        
        for label in labels:
            precision = precision_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)
            
            metrics['per_class'][int(label)] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # ROC-AUC for binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    @staticmethod
    def clustering_metrics(X, labels) -> Dict[str, Any]:
        """
        Calculate clustering metrics
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Silhouette score
        if len(set(labels)) > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
        else:
            metrics['silhouette'] = 0
        
        # Davies-Bouldin index
        if len(set(labels)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        else:
            metrics['davies_bouldin'] = float('inf')
        
        # Calinski-Harabasz index
        if len(set(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['calinski_harabasz'] = 0
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique.astype(int), counts))
        metrics['n_clusters'] = len(unique)
        
        # Noise points (for HDBSCAN)
        if -1 in unique:
            metrics['noise_points'] = counts[unique == -1][0]
        
        return metrics
    
    @staticmethod
    def association_metrics(rules_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics for association rules
        
        Args:
            rules_df: Rules dataframe
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'total_rules': len(rules_df),
            'avg_support': rules_df['support'].mean(),
            'avg_confidence': rules_df['confidence'].mean(),
            'avg_lift': rules_df['lift'].mean(),
            'max_support': rules_df['support'].max(),
            'max_confidence': rules_df['confidence'].max(),
            'max_lift': rules_df['lift'].max(),
            'min_support': rules_df['support'].min(),
            'min_confidence': rules_df['confidence'].min(),
            'min_lift': rules_df['lift'].min()
        }
        
        # Rules by antecedent length
        if 'antecedents' in rules_df.columns:
            metrics['rules_by_length'] = {}
            for i in range(1, 4):
                count = rules_df[rules_df['antecedents'].apply(len) == i].shape[0]
                metrics['rules_by_length'][f'{i}_items'] = count
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        if np.any(mask):
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = float('inf')
        
        return metrics
    
    @staticmethod
    def format_metrics_table(metrics_dict: Dict[str, Any], prefix: str = '') -> pd.DataFrame:
        """
        Format metrics as a DataFrame table
        
        Args:
            metrics_dict: Dictionary of metrics
            prefix: Prefix for metric names
            
        Returns:
            DataFrame with metrics
        """
        rows = []
        
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                # Recursively format nested dictionaries
                nested_df = MetricsCalculator.format_metrics_table(value, f"{prefix}{key}_")
                rows.append(nested_df)
            elif isinstance(value, (list, np.ndarray)) and len(value) > 5:
                # Skip long lists
                rows.append(pd.DataFrame({
                    'Metric': [f"{prefix}{key}"],
                    'Value': [f"Array of length {len(value)}"]
                }))
            else:
                # Format value
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                rows.append(pd.DataFrame({
                    'Metric': [f"{prefix}{key}"],
                    'Value': [formatted_value]
                }))
        
        if rows:
            return pd.concat(rows, ignore_index=True)
        else:
            return pd.DataFrame(columns=['Metric', 'Value'])
    
    @staticmethod
    def save_metrics(metrics_dict: Dict[str, Any], output_path: str):
        """Save metrics to JSON file"""
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        
        converted = convert(metrics_dict)
        
        with open(output_path, 'w') as f:
            json.dump(converted, f, indent=2)
        
        print(f"✅ Saved metrics to {output_path}")