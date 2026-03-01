# -*- coding: utf-8 -*-
"""
Semi-Supervised Learning Module
Handles training with limited labeled data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import os
import joblib


class SemiSupervisedClassifier:
    """
    Class for semi-supervised learning with limited labels
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SemiSupervisedClassifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.results = {}
        self.learning_curves = {}
        
    def create_limited_labels(self,
                               y: np.ndarray,
                               labeled_percent: float,
                               random_state: int = 42) -> np.ndarray:
        """
        Create limited labeled data by masking most labels
        
        Args:
            y: Full labels
            labeled_percent: Percentage of labels to keep (0-100)
            random_state: Random state
            
        Returns:
            Array with -1 for unlabeled samples
        """
        y_limited = y.copy().astype(float)
        
        # Determine number of labeled samples per class
        unique_classes = np.unique(y)
        n_labeled_per_class = int(len(y) * labeled_percent / 100 / len(unique_classes))
        
        np.random.seed(random_state)
        
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            # Randomly select indices to keep labeled
            n_keep = min(n_labeled_per_class, len(cls_indices))
            keep_indices = np.random.choice(cls_indices, n_keep, replace=False)
            
            # Set others to -1 (unlabeled)
            mask = np.ones(len(cls_indices), dtype=bool)
            mask[np.isin(cls_indices, keep_indices)] = False
            unlabel_indices = cls_indices[mask]
            y_limited[unlabel_indices] = -1
        
        print(f"  • Labeled: {np.sum(y_limited != -1):,} samples ({labeled_percent:.1f}%)")
        print(f"  • Unlabeled: {np.sum(y_limited == -1):,} samples")
        
        return y_limited
    
    def train_self_training(self,
                            X: np.ndarray,
                            y_limited: np.ndarray,
                            base_estimator,
                            max_iter: int = 100,
                            threshold: float = 0.75,
                            **kwargs) -> Tuple[SelfTrainingClassifier, Dict[str, Any]]:
        """
        Train self-training classifier
        
        Args:
            X: Feature matrix
            y_limited: Labels with -1 for unlabeled
            base_estimator: Base classifier
            max_iter: Maximum iterations
            threshold: Confidence threshold
            **kwargs: Additional arguments
            
        Returns:
            Tuple (model, results)
        """
        print(f"\n🔄 Training Self-Training (threshold={threshold})...")
        
        model = SelfTrainingClassifier(
            base_estimator,
            max_iter=max_iter,
            threshold=threshold,
            **kwargs
        )
        
        model.fit(X, y_limited)
        
        # Get predictions for all data
        y_pred = model.predict(X)
        
        # Calculate metrics (only on originally labeled data)
        labeled_mask = y_limited != -1
        y_true_labeled = y_limited[labeled_mask]
        y_pred_labeled = y_pred[labeled_mask]
        
        results = {
            'model': model,
            'accuracy': accuracy_score(y_true_labeled, y_pred_labeled),
            'f1': f1_score(y_true_labeled, y_pred_labeled, average='macro'),
            'n_iterations': model.n_iter_,
            'transduction_': model.transduction_ if hasattr(model, 'transduction_') else None,
            'labeled_mask': labeled_mask
        }
        
        # Count pseudo-labels added
        n_pseudo = np.sum((y_limited == -1) & (y_pred != -1))
        results['pseudo_labels_added'] = n_pseudo
        
        print(f"  • Accuracy on labeled: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        print(f"  • Pseudo-labels added: {n_pseudo}")
        print(f"  • Iterations: {results['n_iterations']}")
        
        return model, results
    
    def train_label_propagation(self,
                                 X: np.ndarray,
                                 y_limited: np.ndarray,
                                 kernel: str = 'rbf',
                                 gamma: float = 20,
                                 n_neighbors: int = 7,
                                 **kwargs) -> Tuple[LabelSpreading, Dict[str, Any]]:
        """
        Train label propagation/spreading classifier
        
        Args:
            X: Feature matrix
            y_limited: Labels with -1 for unlabeled
            kernel: Kernel type ('knn' or 'rbf')
            gamma: Gamma parameter for rbf kernel
            n_neighbors: Number of neighbors for knn kernel
            **kwargs: Additional arguments
            
        Returns:
            Tuple (model, results)
        """
        print(f"\n🔄 Training Label Propagation (kernel={kernel})...")
        
        model = LabelSpreading(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            **kwargs
        )
        
        model.fit(X, y_limited)
        
        # Get predictions for all data
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Calculate metrics (only on originally labeled data)
        labeled_mask = y_limited != -1
        y_true_labeled = y_limited[labeled_mask]
        y_pred_labeled = y_pred[labeled_mask]
        
        results = {
            'model': model,
            'accuracy': accuracy_score(y_true_labeled, y_pred_labeled),
            'f1': f1_score(y_true_labeled, y_pred_labeled, average='macro'),
            'labeled_mask': labeled_mask,
            'classes_': model.classes_,
            'X_transduction': model.X_transduction_ if hasattr(model, 'X_transduction_') else None
        }
        
        # Confidence on unlabeled data
        unlabeled_mask = y_limited == -1
        if np.any(unlabeled_mask):
            results['unlabeled_confidence'] = np.max(y_pred_proba[unlabeled_mask], axis=1).mean()
        
        print(f"  • Accuracy on labeled: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        
        return model, results
    
    def run_experiment(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       base_estimator,
                       labeled_percents: List[float],
                       method: str = 'self_training',
                       random_state: int = 42) -> pd.DataFrame:
        """
        Run experiment with different percentages of labeled data
        
        Args:
            X: Feature matrix
            y: Full labels
            base_estimator: Base classifier
            labeled_percents: List of percentages to try
            method: Method ('self_training' or 'label_propagation')
            random_state: Random state
            
        Returns:
            DataFrame with results
        """
        print(f"\n{'='*60}")
        print(f"🔬 Running semi-supervised experiment ({method})")
        print(f"{'='*60}")
        
        results = []
        
        for percent in labeled_percents:
            print(f"\n📊 Labeled: {percent}%")
            print("-" * 40)
            
            # Create limited labels
            y_limited = self.create_limited_labels(y, percent, random_state)
            
            # Split into train/test (use all data for training, but only labeled for evaluation)
            X_train, X_test, y_limited_train, y_limited_test, y_train_full, y_test_full = train_test_split(
                X, y_limited, y, test_size=0.2, random_state=random_state, stratify=y
            )
            
            # Train model
            if method == 'self_training':
                model, model_results = self.train_self_training(
                    X_train, y_limited_train, base_estimator
                )
            else:  # label_propagation
                model, model_results = self.train_label_propagation(
                    X_train, y_limited_train
                )
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            
            # Filter out unlabeled in test (should not happen, but just in case)
            test_labeled_mask = y_limited_test != -1
            if np.any(test_labeled_mask):
                y_test_filtered = y_test_full[test_labeled_mask]
                y_pred_filtered = y_pred[test_labeled_mask]
                
                test_accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
                test_f1 = f1_score(y_test_filtered, y_pred_filtered, average='macro')
            else:
                test_accuracy = accuracy_score(y_test_full, y_pred)
                test_f1 = f1_score(y_test_full, y_pred, average='macro')
            
            # Train baseline supervised model on same limited data
            from sklearn.base import clone
            supervised_model = clone(base_estimator)
            
            # Use only labeled data for training
            train_labeled_mask = y_limited_train != -1
            if np.any(train_labeled_mask):
                X_train_labeled = X_train[train_labeled_mask]
                y_train_labeled = y_limited_train[train_labeled_mask].astype(int)
                supervised_model.fit(X_train_labeled, y_train_labeled)
                
                # Evaluate
                y_pred_supervised = supervised_model.predict(X_test)
                supervised_accuracy = accuracy_score(y_test_full, y_pred_supervised)
                supervised_f1 = f1_score(y_test_full, y_pred_supervised, average='macro')
            else:
                supervised_accuracy = 0
                supervised_f1 = 0
            
            results.append({
                'labeled_percent': percent,
                'method': method,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'supervised_accuracy': supervised_accuracy,
                'supervised_f1': supervised_f1,
                'improvement': test_f1 - supervised_f1,
                'pseudo_labels': model_results.get('pseudo_labels_added', 0),
                'iterations': model_results.get('n_iterations', 1)
            })
        
        results_df = pd.DataFrame(results)
        self.results[method] = results_df
        
        print(f"\n✅ Experiment complete")
        return results_df
    
    def plot_learning_curve(self, method: str = 'self_training'):
        """
        Plot learning curve comparing semi-supervised vs supervised
        
        Args:
            method: Method to plot
        """
        if method not in self.results:
            raise ValueError(f"No results for method {method}")
        
        df = self.results[method]
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(df['labeled_percent'], df['test_f1'], 'b-o', label=f'{method} (semi-supervised)', linewidth=2)
        plt.plot(df['labeled_percent'], df['supervised_f1'], 'r--s', label='Supervised (only labeled)', linewidth=2)
        
        # Fill improvement area
        plt.fill_between(df['labeled_percent'], 
                         df['supervised_f1'], 
                         df['test_f1'],
                         alpha=0.3, color='green', label='Improvement')
        
        plt.xlabel('Labeled Data (%)', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title(f'Learning Curve: {method} vs Supervised', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def analyze_pseudo_labels(self,
                              model,
                              X: np.ndarray,
                              y_limited: np.ndarray,
                              y_true: np.ndarray) -> pd.DataFrame:
        """
        Analyze quality of pseudo-labels
        
        Args:
            model: Trained semi-supervised model
            X: Feature matrix
            y_limited: Limited labels with -1
            y_true: True labels
            
        Returns:
            DataFrame with analysis
        """
        if not hasattr(model, 'transduction_'):
            raise ValueError("Model does not have transduction_ attribute")
        
        y_pred = model.transduction_
        
        # Identify pseudo-labeled samples
        pseudo_mask = (y_limited == -1) & (y_pred != -1)
        
        if not np.any(pseudo_mask):
            print("No pseudo-labels added")
            return pd.DataFrame()
        
        # Check accuracy of pseudo-labels
        pseudo_true = y_true[pseudo_mask]
        pseudo_pred = y_pred[pseudo_mask].astype(int)
        
        pseudo_accuracy = accuracy_score(pseudo_true, pseudo_pred)
        
        # Get prediction confidence (if available)
        if hasattr(model, 'label_distributions_'):
            confidence = np.max(model.label_distributions_[pseudo_mask], axis=1)
        else:
            confidence = np.ones(len(pseudo_true))
        
        # Create analysis dataframe
        analysis = pd.DataFrame({
            'true_label': pseudo_true,
            'predicted_label': pseudo_pred,
            'correct': pseudo_true == pseudo_pred,
            'confidence': confidence
        })
        
        print(f"\n🔍 Pseudo-label Analysis:")
        print(f"  • Total pseudo-labels: {len(analysis)}")
        print(f"  • Accuracy: {pseudo_accuracy:.4f}")
        print(f"  • Avg confidence: {confidence.mean():.4f}")
        
        # Accuracy by confidence threshold
        for threshold in [0.6, 0.7, 0.8, 0.9]:
            high_conf = analysis[analysis['confidence'] >= threshold]
            if len(high_conf) > 0:
                acc = high_conf['correct'].mean()
                print(f"  • Threshold {threshold:.1f}: {len(high_conf)} samples, acc={acc:.4f}")
        
        return analysis
    
    def save_results(self, output_dir: str):
        """Save results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for method, df in self.results.items():
            df.to_csv(os.path.join(output_dir, f'{method}_results.csv'), index=False)
        
        print(f"✅ Saved semi-supervised results to {output_dir}")