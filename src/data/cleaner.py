# -*- coding: utf-8 -*-
"""
Data Cleaner Module
Handles text cleaning, preprocessing, and data transformation
"""

import re
import pandas as pd
import numpy as np
import os
from typing import List, Optional, Tuple, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


class DataCleaner:
    """
    Class for cleaning and preprocessing text data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.stop_words = set(stopwords.words('english')) if self.config.get('remove_stopwords', True) else set()
        self.stemmer = PorterStemmer() if self.config.get('do_stemming', True) else None
        self.lemmatizer = WordNetLemmatizer() if self.config.get('do_lemmatization', False) else None
        self.cleaned_data = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [t for t in tokens if t not in self.stop_words]
    
    def stem(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        if self.stemmer:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stem or lemmatize
        if self.stemmer:
            tokens = self.stem(tokens)
        elif self.lemmatizer:
            tokens = self.lemmatize(tokens)
        
        # Join back
        return ' '.join(tokens)
    
    def extract_helpful_votes(self, helpful_str: str) -> Tuple[int, int]:
        """
        Extract helpful votes from string format "x/y"
        
        Args:
            helpful_str: String like "15/20"
            
        Returns:
            Tuple (helpful_yes, helpful_total)
        """
        if not isinstance(helpful_str, str):
            return 0, 0
        
        try:
            parts = helpful_str.split('/')
            if len(parts) == 2:
                yes = int(parts[0])
                total = int(parts[1])
                return yes, total
        except:
            pass
        
        return 0, 0
    
    def clean_dataframe(self, 
                        df: pd.DataFrame,
                        text_column: str = 'review',
                        label_column: Optional[str] = 'sentiment',
                        rating_column: Optional[str] = 'rating',
                        helpful_column: Optional[str] = 'helpful') -> pd.DataFrame:
        """
        Clean entire dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column
            rating_column: Name of rating column
            helpful_column: Name of helpful column
            
        Returns:
            Cleaned dataframe
        """
        print("🧹 Cleaning dataframe...")
        df_clean = df.copy()
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"  • Removed {initial_len - len(df_clean)} duplicates")
        
        # Remove rows with missing text
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=[text_column])
        print(f"  • Removed {initial_len - len(df_clean)} rows with missing text")
        
        # Filter by text length
        min_len = self.config.get('min_review_length', 5)
        max_len = self.config.get('max_review_length', 1000)
        
        df_clean['review_length_raw'] = df_clean[text_column].astype(str).str.split().str.len()
        initial_len = len(df_clean)
        df_clean = df_clean[
            (df_clean['review_length_raw'] >= min_len) & 
            (df_clean['review_length_raw'] <= max_len)
        ]
        print(f"  • Filtered by length: kept {len(df_clean)} rows (min={min_len}, max={max_len})")
        
        # Clean text
        print(f"  • Cleaning text column '{text_column}'...")
        df_clean['review_clean'] = df_clean[text_column].astype(str).apply(self.preprocess_text)
        df_clean['review_length'] = df_clean['review_clean'].str.split().str.len()
        
        # Extract helpful votes if column exists
        if helpful_column and helpful_column in df_clean.columns:
            print(f"  • Extracting helpful votes from '{helpful_column}'...")
            helpful_data = df_clean[helpful_column].apply(self.extract_helpful_votes)
            df_clean['helpful_yes'] = helpful_data.apply(lambda x: x[0])
            df_clean['helpful_total'] = helpful_data.apply(lambda x: x[1])
            df_clean['helpful_ratio'] = df_clean.apply(
                lambda row: row['helpful_yes'] / row['helpful_total'] if row['helpful_total'] > 0 else 0,
                axis=1
            )
        
        # Create sentiment labels from ratings if label column not provided
        if label_column not in df_clean.columns and rating_column in df_clean.columns:
            print(f"  • Creating sentiment labels from '{rating_column}'...")
            # 4-5 stars: positive (1), 1-2 stars: negative (0), 3 stars: neutral (exclude)
            df_clean = df_clean[df_clean[rating_column] != 3]  # Remove neutral
            df_clean['sentiment'] = (df_clean[rating_column] >= 4).astype(int)
            print(f"    • Positive: {df_clean['sentiment'].sum():,}")
            print(f"    • Negative: {len(df_clean) - df_clean['sentiment'].sum():,}")
        
        # Tạo cột sentiment từ label nếu chưa có
        if 'sentiment' not in df_clean.columns:
            if 'label' in df_clean.columns:
                print(f"  • Creating sentiment column from 'label'")
                # Chuyển label 1,2 thành 0,1 (1=negative, 2=positive)
                df_clean['sentiment'] = (df_clean['label'] == 2).astype(int)
                print(f"    • Positive (label=2): {df_clean['sentiment'].sum():,}")
                print(f"    • Negative (label=1): {len(df_clean) - df_clean['sentiment'].sum():,}")
            else:
                print(f"  • Warning: No label column found, using default sentiment")
                df_clean['sentiment'] = 1  # Mặc định là positive
        
        self.cleaned_data = df_clean
        print(f"✅ Cleaning complete. Final shape: {df_clean.shape}")
        print(f"   Columns: {df_clean.columns.tolist()}")
        
        return df_clean
    
    def get_texts_and_labels(self, 
                             df: Optional[pd.DataFrame] = None,
                             text_column: str = 'review_clean',
                             label_column: str = 'sentiment') -> Tuple[List[str], np.ndarray]:
        """
        Extract texts and labels from dataframe
        
        Args:
            df: Dataframe (uses self.cleaned_data if None)
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple (texts, labels)
        """
        if df is None:
            df = self.cleaned_data
            
        if df is None:
            raise ValueError("No data available")
        
        texts = df[text_column].tolist()
        labels = df[label_column].values
        
        return texts, labels
    
    def save_cleaned_data(self, output_path: str):
        """Save cleaned data to file"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data to save")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned data to: {output_path}")