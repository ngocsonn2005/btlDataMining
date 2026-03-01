# -*- coding: utf-8 -*-
"""
Supervised Learning Module
Handles training and evaluation of supervised classifiers
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import joblib
import os
import time


class SupervisedClassifier:
    """
    Class for training and evaluating supervised classifiers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SupervisedClassifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def split_data(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set size
            random_state: Random state
            stratify: Whether to stratify split
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        stratify_y = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
        )
        
        print(f"📊 Data split:")
        print(f"  • Train: {X_train.shape[0]:,} samples")
        print(f"  • Test: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_naive_bayes(self,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          **kwargs) -> Dict[str, Any]:
        """
        Train Naive Bayes classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results
        """
        print("\n📊 Training Naive Bayes...")
        start_time = time.time()
        
        # Check if features are non-negative (for MultinomialNB)
        if np.min(X_train) >= 0:
            model = MultinomialNB(**kwargs)
        else:
            model = GaussianNB(**kwargs)
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        results = self._compute_metrics(y_test, y_pred, y_pred_proba)
        results['train_time'] = time.time() - start_time
        results['model'] = model
        
        self.models['naive_bayes'] = model
        self.results['naive_bayes'] = results
        
        print(f"  • Accuracy: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        print(f"  • Time: {results['train_time']:.2f}s")
        
        return results
    
    def train_logistic_regression(self,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   **kwargs) -> Dict[str, Any]:
        """
        Train Logistic Regression classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results
        """
        print("\n📊 Training Logistic Regression...")
        start_time = time.time()
        
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results = self._compute_metrics(y_test, y_pred, y_pred_proba)
        results['train_time'] = time.time() - start_time
        results['model'] = model
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        print(f"  • Accuracy: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        print(f"  • Time: {results['train_time']:.2f}s")
        
        return results
    
    def train_svm(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  **kwargs) -> Dict[str, Any]:
        """
        Train SVM classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results
        """
        print("\n📊 Training SVM...")
        start_time = time.time()
        
        default_params = {
            'C': 1.0,
            'kernel': 'linear',
            'random_state': 42,
            'class_weight': 'balanced',
            'probability': True
        }
        default_params.update(kwargs)
        
        model = SVC(**default_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results = self._compute_metrics(y_test, y_pred, y_pred_proba)
        results['train_time'] = time.time() - start_time
        results['model'] = model
        
        self.models['svm'] = model
        self.results['svm'] = results
        
        print(f"  • Accuracy: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        print(f"  • Time: {results['train_time']:.2f}s")
        
        return results
    
    def train_random_forest(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            **kwargs) -> Dict[str, Any]:
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results
        """
        print("\n📊 Training Random Forest...")
        start_time = time.time()
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results = self._compute_metrics(y_test, y_pred, y_pred_proba)
        results['train_time'] = time.time() - start_time
        results['model'] = model
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        print(f"  • Accuracy: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        print(f"  • Time: {results['train_time']:.2f}s")
        
        return results
    
    def train_xgboost(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results
        """
        print("\n📊 Training XGBoost...")
        start_time = time.time()
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        results = self._compute_metrics(y_test, y_pred, y_pred_proba)
        results['train_time'] = time.time() - start_time
        results['model'] = model
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        print(f"  • Accuracy: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        print(f"  • Time: {results['train_time']:.2f}s")
        
        return results
    
    def build_lstm_model(self,
                         input_dim: int,
                         embedding_dim: int = 100,
                         hidden_dim: int = 128,
                         dropout: float = 0.5,
                         learning_rate: float = 0.001) -> keras.Model:
        """
        Build LSTM model for text classification
        
        Args:
            input_dim: Input dimension
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Reshape((1, input_dim)),
            layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout),
            layers.LSTM(hidden_dim // 2, dropout=dropout),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        
        return model
    
    def train_lstm(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   **kwargs) -> Dict[str, Any]:
        """
        Train LSTM classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results
        """
        print("\n📊 Training LSTM...")
        start_time = time.time()
        
        default_params = {
            'embedding_dim': 100,
            'hidden_dim': 128,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 10,
            'validation_split': 0.1
        }
        default_params.update(kwargs)
        
        # Build model
        model = self.build_lstm_model(
            input_dim=X_train.shape[1],
            embedding_dim=default_params['embedding_dim'],
            hidden_dim=default_params['hidden_dim'],
            dropout=default_params['dropout'],
            learning_rate=default_params['learning_rate']
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=default_params['batch_size'],
            epochs=default_params['epochs'],
            validation_split=default_params['validation_split'],
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        results = self._compute_metrics(y_test, y_pred, y_pred_proba)
        results['train_time'] = time.time() - start_time
        results['history'] = history.history
        results['model'] = model
        
        self.models['lstm'] = model
        self.results['lstm'] = results
        
        print(f"  • Accuracy: {results['accuracy']:.4f}")
        print(f"  • F1-score: {results['f1']:.4f}")
        print(f"  • Time: {results['train_time']:.2f}s")
        
        return results
    
    def _compute_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Compute classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def cross_validate(self,
                       model_name: str,
                       X: np.ndarray,
                       y: np.ndarray,
                       cv: int = 5,
                       scoring: str = 'f1_macro') -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Create a new instance of the same model type
        if hasattr(model, 'get_params'):
            model_class = model.__class__
            params = model.get_params()
            cv_model = model_class(**params)
        else:
            cv_model = model
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(cv_model, X, y, cv=cv_splitter, scoring=scoring)
        
        results = {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        print(f"\n📊 Cross-validation results for {model_name}:")
        print(f"  • {scoring}: {results['mean']:.4f} ± {results['std']:.4f}")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        if not self.results:
            raise ValueError("No results available")
        
        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-score': results.get('f1', 0),
                'ROC-AUC': results.get('roc_auc', 0),
                'Train Time (s)': results.get('train_time', 0)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('F1-score', ascending=False)
        
        # Set best model
        self.best_model_name = df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best model: {self.best_model_name} (F1={df.iloc[0]['F1-score']:.4f})")
        
        return df
    
    def save_model(self, model_name: str, output_path: str):
        """Save model to disk"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if model_name == 'lstm':
            model.save(output_path)
        else:
            joblib.dump(model, output_path)
        
        print(f"✅ Saved {model_name} to {output_path}")
    
    def load_model(self, model_name: str, input_path: str):
        """Load model from disk"""
        if model_name == 'lstm':
            model = keras.models.load_model(input_path)
        else:
            model = joblib.load(input_path)
        
        self.models[model_name] = model
        print(f"✅ Loaded {model_name} from {input_path}")
        
        return model