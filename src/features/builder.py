# -*- coding: utf-8 -*-
"""
Feature Builder Module
Handles feature engineering for text data
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gensim
from gensim.models import Word2Vec
import joblib
import os


class FeatureBuilder:
    """
    Class for building features from text data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FeatureBuilder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.scaler = None
        self.feature_names = []
        
    def build_tfidf_features(self, 
                             texts: List[str],
                             max_features: int = 5000,
                             ngram_range: Tuple[int, int] = (1, 2),
                             min_df: int = 5,
                             max_df: float = 0.8,
                             fit: bool = True) -> np.ndarray:
        """
        Build TF-IDF features from texts
        
        Args:
            texts: List of text strings
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            fit: Whether to fit the vectorizer
            
        Returns:
            TF-IDF feature matrix
        """
        print(f"📊 Building TF-IDF features (max_features={max_features})...")
        
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
            X = self.tfidf_vectorizer.fit_transform(texts)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        else:
            X = self.tfidf_vectorizer.transform(texts)
        
        print(f"  • Shape: {X.shape}")
        return X.toarray()
    
    def build_word2vec_features(self,
                                texts: List[str],
                                vector_size: int = 100,
                                window: int = 5,
                                min_count: int = 2,
                                workers: int = 4,
                                epochs: int = 10,
                                fit: bool = True) -> np.ndarray:
        """
        Build Word2Vec features (average of word vectors)
        
        Args:
            texts: List of text strings (already tokenized)
            vector_size: Size of word vectors
            window: Context window size
            min_count: Minimum word count
            workers: Number of worker threads
            epochs: Number of training epochs
            fit: Whether to fit the model
            
        Returns:
            Word2Vec feature matrix (average of word vectors per document)
        """
        print(f"📊 Building Word2Vec features (vector_size={vector_size})...")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        if fit or self.word2vec_model is None:
            # Train Word2Vec model
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
                epochs=epochs
            )
        
        # Compute document vectors (average of word vectors)
        X = []
        vocab = set(self.word2vec_model.wv.key_to_index.keys())
        
        for tokens in tokenized_texts:
            # Get vectors for words in vocabulary
            word_vectors = [self.word2vec_model.wv[word] for word in tokens if word in vocab]
            
            if word_vectors:
                # Average of word vectors
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words in vocabulary
                doc_vector = np.zeros(vector_size)
            
            X.append(doc_vector)
        
        X = np.array(X)
        print(f"  • Shape: {X.shape}")
        return X
    
    def build_rfm_like_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build RFM-like features from dataframe
        
        Args:
            df: Dataframe with review features
            
        Returns:
            Feature matrix
        """
        print("📊 Building RFM-like features...")
        
        features = []
        feature_names = []
        
        # Review length
        if 'review_length' in df.columns:
            features.append(df['review_length'].values.reshape(-1, 1))
            feature_names.append('review_length')
        
        # Helpful ratio
        if 'helpful_ratio' in df.columns:
            features.append(df['helpful_ratio'].values.reshape(-1, 1))
            feature_names.append('helpful_ratio')
        
        # Rating (if available)
        if 'rating' in df.columns:
            features.append(df['rating'].values.reshape(-1, 1))
            feature_names.append('rating')
        
        if features:
            X = np.hstack(features)
            print(f"  • Shape: {X.shape}")
            print(f"  • Features: {feature_names}")
            return X
        else:
            print("  • No RFM-like features found")
            return np.array([])
    
    def scale_features(self, 
                       X: np.ndarray,
                       method: str = 'standard',
                       fit: bool = True) -> np.ndarray:
        """
        Scale features
        
        Args:
            X: Feature matrix
            method: Scaling method ('standard' or 'minmax')
            fit: Whether to fit the scaler
            
        Returns:
            Scaled feature matrix
        """
        if fit or self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def combine_features(self, feature_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple feature matrices horizontally
        
        Args:
            feature_matrices: List of feature matrices
            
        Returns:
            Combined feature matrix
        """
        # Filter out empty matrices
        valid_matrices = [X for X in feature_matrices if X.size > 0]
        
        if not valid_matrices:
            return np.array([])
        
        X_combined = np.hstack(valid_matrices)
        print(f"📊 Combined features: {X_combined.shape}")
        
        return X_combined
    
    def save_models(self, output_dir: str):
        """Save feature models to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))
            print(f"✅ Saved TF-IDF vectorizer")
        
        if self.word2vec_model:
            self.word2vec_model.save(os.path.join(output_dir, 'word2vec.model'))
            print(f"✅ Saved Word2Vec model")
        
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
            print(f"✅ Saved scaler")
    
    def load_models(self, input_dir: str):
        """Load feature models from disk"""
        tfidf_path = os.path.join(input_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            print(f"✅ Loaded TF-IDF vectorizer")
        
        w2v_path = os.path.join(input_dir, 'word2vec.model')
        if os.path.exists(w2v_path):
            self.word2vec_model = Word2Vec.load(w2v_path)
            print(f"✅ Loaded Word2Vec model")
        
        scaler_path = os.path.join(input_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler")