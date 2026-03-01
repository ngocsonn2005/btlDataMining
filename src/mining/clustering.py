# -*- coding: utf-8 -*-
"""
Clustering Module
Handles text clustering and topic modeling
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA, TruncatedSVD
import hdbscan
import joblib
import os


class TextClusterer:
    """
    Class for clustering text data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TextClusterer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.labels = None
        self.feature_names = None
        
    def find_optimal_k(self,
                       X: np.ndarray,
                       k_min: int = 2,
                       k_max: int = 10,
                       method: str = 'silhouette',
                       random_state: int = 42) -> pd.DataFrame:
        """
        Find optimal number of clusters
        
        Args:
            X: Feature matrix
            k_min: Minimum number of clusters
            k_max: Maximum number of clusters
            method: Method to use ('silhouette', 'elbow', 'gap')
            random_state: Random state
            
        Returns:
            DataFrame with results for each k
        """
        print(f"🔍 Finding optimal k (range {k_min}-{k_max})...")
        
        results = []
        
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            labels = kmeans.fit_predict(X)
            
            # Silhouette score
            sil_score = silhouette_score(X, labels)
            
            # Davies-Bouldin index (lower is better)
            db_score = davies_bouldin_score(X, labels)
            
            # Calinski-Harabasz index (higher is better)
            ch_score = calinski_harabasz_score(X, labels)
            
            # Inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            results.append({
                'k': k,
                'silhouette': sil_score,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score,
                'inertia': inertia
            })
            
            print(f"  k={k:2d} | silhouette={sil_score:.4f} | DB={db_score:.4f} | CH={ch_score:.1f}")
        
        results_df = pd.DataFrame(results)
        
        # Suggest best k
        best_by_silhouette = results_df.loc[results_df['silhouette'].idxmax(), 'k']
        best_by_db = results_df.loc[results_df['davies_bouldin'].idxmin(), 'k']
        best_by_ch = results_df.loc[results_df['calinski_harabasz'].idxmax(), 'k']
        
        print(f"\n🎯 Suggested k:")
        print(f"  • By silhouette: {best_by_silhouette}")
        print(f"  • By Davies-Bouldin: {best_by_db}")
        print(f"  • By Calinski-Harabasz: {best_by_ch}")
        
        return results_df
    
    def fit_kmeans(self,
                   X: np.ndarray,
                   n_clusters: int,
                   random_state: int = 42,
                   **kwargs) -> np.ndarray:
        """
        Fit K-Means clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            random_state: Random state
            **kwargs: Additional arguments for KMeans
            
        Returns:
            Cluster labels
        """
        print(f"🔄 Fitting K-Means with k={n_clusters}...")
        
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto', **kwargs)
        self.labels = self.model.fit_predict(X)
        
        # Calculate metrics
        sil_score = silhouette_score(X, self.labels)
        db_score = davies_bouldin_score(X, self.labels)
        ch_score = calinski_harabasz_score(X, self.labels)
        
        print(f"  • Silhouette: {sil_score:.4f}")
        print(f"  • Davies-Bouldin: {db_score:.4f}")
        print(f"  • Calinski-Harabasz: {ch_score:.1f}")
        
        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"  • Cluster {cluster}: {count:,} samples ({count/len(X)*100:.1f}%)")
        
        return self.labels
    
    def fit_hdbscan(self,
                    X: np.ndarray,
                    min_cluster_size: int = 50,
                    min_samples: Optional[int] = None,
                    **kwargs) -> np.ndarray:
        """
        Fit HDBSCAN clustering
        
        Args:
            X: Feature matrix
            min_cluster_size: Minimum cluster size
            min_samples: Number of samples in a neighborhood
            **kwargs: Additional arguments for HDBSCAN
            
        Returns:
            Cluster labels (-1 indicates noise)
        """
        print(f"🔄 Fitting HDBSCAN (min_cluster_size={min_cluster_size})...")
        
        self.model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs)
        self.labels = self.model.fit_predict(X)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = np.sum(self.labels == -1)
        
        print(f"  • Found {n_clusters} clusters")
        print(f"  • Noise points: {n_noise:,} ({n_noise/len(X)*100:.1f}%)")
        
        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            if cluster == -1:
                print(f"  • Noise: {count:,} samples")
            else:
                print(f"  • Cluster {cluster}: {count:,} samples ({count/len(X)*100:.1f}%)")
        
        return self.labels
    
    def fit_agglomerative(self,
                          X: np.ndarray,
                          n_clusters: int,
                          linkage: str = 'ward',
                          **kwargs) -> np.ndarray:
        """
        Fit Agglomerative clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion
            **kwargs: Additional arguments for AgglomerativeClustering
            
        Returns:
            Cluster labels
        """
        print(f"🔄 Fitting Agglomerative clustering with k={n_clusters}...")
        
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
        self.labels = self.model.fit_predict(X)
        
        # Calculate metrics
        sil_score = silhouette_score(X, self.labels)
        db_score = davies_bouldin_score(X, self.labels)
        ch_score = calinski_harabasz_score(X, self.labels)
        
        print(f"  • Silhouette: {sil_score:.4f}")
        print(f"  • Davies-Bouldin: {db_score:.4f}")
        print(f"  • Calinski-Harabasz: {ch_score:.1f}")
        
        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"  • Cluster {cluster}: {count:,} samples ({count/len(X)*100:.1f}%)")
        
        return self.labels
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers (for K-Means only)"""
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        else:
            raise AttributeError("Model does not have cluster centers")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict clusters for new data"""
        if self.model is None:
            raise ValueError("No model fitted")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise AttributeError("Model does not support predict")
    
    def get_topic_words(self,
                        X: np.ndarray,
                        feature_names: List[str],
                        n_words: int = 10) -> Dict[int, List[str]]:
        """
        Get top words for each cluster (for K-Means)
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            n_words: Number of top words per cluster
            
        Returns:
            Dictionary mapping cluster ID to list of top words
        """
        if self.labels is None:
            raise ValueError("No labels available")
        
        if not hasattr(self.model, 'cluster_centers_'):
            raise AttributeError("Model does not have cluster centers")
        
        centers = self.model.cluster_centers_
        
        topic_words = {}
        for i in range(centers.shape[0]):
            # Get indices of top words
            top_indices = np.argsort(centers[i])[-n_words:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            topic_words[i] = top_words
        
        return topic_words
    
    def profile_clusters(self,
                         X: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         texts: Optional[List[str]] = None,
                         n_examples: int = 3) -> pd.DataFrame:
        """
        Create cluster profiling dataframe
        
        Args:
            X: Feature matrix
            feature_names: Feature names for topic extraction
            texts: Original texts for examples
            n_examples: Number of example texts per cluster
            
        Returns:
            DataFrame with cluster profiles
        """
        if self.labels is None:
            raise ValueError("No labels available")
        
        profiles = []
        
        for cluster_id in sorted(set(self.labels)):
            if cluster_id == -1:  # Skip noise for HDBSCAN
                continue
                
            mask = self.labels == cluster_id
            cluster_size = np.sum(mask)
            
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': cluster_size / len(self.labels) * 100
            }
            
            # Add top words if feature_names provided
            if feature_names is not None and hasattr(self.model, 'cluster_centers_'):
                centers = self.model.cluster_centers_
                top_indices = np.argsort(centers[cluster_id])[-10:][::-1]
                top_words = [feature_names[idx] for idx in top_indices]
                profile['top_words'] = ', '.join(top_words[:5])
            
            # Add example texts if provided
            if texts is not None:
                cluster_texts = np.array(texts)[mask]
                example_indices = np.random.choice(len(cluster_texts), min(n_examples, len(cluster_texts)), replace=False)
                examples = [cluster_texts[i][:100] + "..." if len(cluster_texts[i]) > 100 else cluster_texts[i] 
                           for i in example_indices]
                profile['examples'] = examples
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def project_2d(self, X: np.ndarray, method: str = 'pca', random_state: int = 42) -> np.ndarray:
        """
        Project data to 2D for visualization
        
        Args:
            X: Feature matrix
            method: Projection method ('pca' or 'svd')
            random_state: Random state
            
        Returns:
            2D projection
        """
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=random_state)
        else:
            reducer = TruncatedSVD(n_components=2, random_state=random_state)
        
        Z = reducer.fit_transform(X)
        
        explained_var = reducer.explained_variance_ratio_.sum() if method.lower() == 'pca' else None
        if explained_var:
            print(f"  • Explained variance: {explained_var:.2%}")
        
        return Z
    
    def save_model(self, output_path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.model, output_path)
        print(f"✅ Saved model to {output_path}")
    
    def load_model(self, input_path: str):
        """Load model from disk"""
        self.model = joblib.load(input_path)
        print(f"✅ Loaded model from {input_path}")