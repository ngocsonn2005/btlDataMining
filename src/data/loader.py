# -*- coding: utf-8 -*-
"""
Data Loader Module
Handles loading and basic inspection of raw data
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import yaml


class DataLoader:
    """
    Class for loading and inspecting raw data
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader with configuration
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.raw_data = None
        self.data_info = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_csv(self, 
                 file_path: str, 
                 encoding: str = 'utf-8',
                 nrows: Optional[int] = None,
                 skiprows: Optional[int] = None,
                 **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding
            nrows: Number of rows to read (for large files)
            skiprows: Number of rows to skip (for large files)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        print(f"📂 Loading data from: {file_path}")
        
        # Thông báo nếu chỉ đọc một phần
        if nrows:
            print(f"  • Chỉ đọc {nrows:,} dòng đầu tiên (do dung lượng lớn)")
        
        try:
            # Kiểm tra nếu là file train/test gốc (không có header)
            if 'train.csv' in file_path or 'test.csv' in file_path:
                # File không có header, đọc với header=None và gán tên cột
                df = pd.read_csv(
                    file_path, 
                    encoding=encoding, 
                    header=None,
                    names=['label', 'title', 'review_text'],
                    nrows=nrows,
                    skiprows=skiprows,
                    **kwargs
                )
                print(f"  • Đọc file không header, gán tên cột: {df.columns.tolist()}")
            else:
                # File bình thường có header
                df = pd.read_csv(file_path, encoding=encoding, nrows=nrows, skiprows=skiprows, **kwargs)
            
            self.raw_data = df
            
            # Thông báo kích thước file gốc nếu chỉ đọc một phần
            if nrows:
                # Ước tính tổng số dòng (chỉ để thông báo)
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        total_lines = sum(1 for _ in f) - 1  # Trừ header
                    print(f"  • Tổng số dòng trong file: {total_lines:,}")
                    print(f"  • Đã đọc {len(df):,}/{total_lines:,} dòng ({len(df)/total_lines*100:.1f}%)")
                except:
                    pass
            else:
                print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def load_csv_in_chunks(self,
                          file_path: str,
                          chunksize: int = 10000,
                          encoding: str = 'utf-8',
                          **kwargs) -> pd.DataFrame:
        """
        Load data in chunks and combine (for very large files)
        
        Args:
            file_path: Path to CSV file
            chunksize: Number of rows per chunk
            encoding: File encoding
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with sampled data
        """
        print(f"📂 Loading data in chunks from: {file_path}")
        print(f"  • Chunksize: {chunksize:,} rows")
        
        chunks = []
        total_rows = 0
        
        try:
            # Xác định định dạng file
            if 'train.csv' in file_path or 'test.csv' in file_path:
                # File không header
                for chunk in pd.read_csv(file_path, 
                                        encoding=encoding,
                                        header=None,
                                        names=['label', 'title', 'review_text'],
                                        chunksize=chunksize,
                                        **kwargs):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    print(f"  • Đã đọc {total_rows:,} dòng...")
                    
                    # Dừng sau 5 chunks để test (có thể điều chỉnh)
                    if len(chunks) >= 5:
                        print(f"  • Dừng sau {len(chunks)} chunks để test")
                        break
            else:
                # File có header
                for chunk in pd.read_csv(file_path, 
                                        encoding=encoding,
                                        chunksize=chunksize,
                                        **kwargs):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    print(f"  • Đã đọc {total_rows:,} dòng...")
                    
                    # Dừng sau 5 chunks để test
                    if len(chunks) >= 5:
                        print(f"  • Dừng sau {len(chunks)} chunks để test")
                        break
            
            df = pd.concat(chunks, ignore_index=True)
            self.raw_data = df
            
            print(f"✅ Loaded {len(df):,} rows from {len(chunks)} chunks")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data in chunks: {e}")
            raise
    
    def load_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate sample data for testing (when real data is not available)
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample data
        """
        print(f"🔧 Generating {n_samples} sample reviews...")
        
        np.random.seed(42)
        
        # Sample positive reviews
        positive_templates = [
            "Great product! {feature} is amazing.",
            "I love this {product}. It's {adjective}.",
            "Excellent quality, {feature} works perfectly.",
            "Best purchase ever! Highly recommended.",
            "This {product} exceeded my expectations.",
            "Amazing {feature}, worth every penny.",
            "Five stars! {product} is {adjective}.",
            "Very satisfied with this purchase.",
            "Perfect {product} for my needs.",
            "Outstanding quality and fast shipping."
        ]
        
        # Sample negative reviews
        negative_templates = [
            "Terrible product. {feature} broke after {days} days.",
            "Waste of money. {product} is {adjective}.",
            "Poor quality, {feature} doesn't work.",
            "Very disappointed with this purchase.",
            "Avoid this {product}. It's {adjective}.",
            "{feature} is defective. Requesting refund.",
            "One star! {product} is completely useless.",
            "Not worth the price. {feature} failed.",
            "Cheap materials, {product} broke easily.",
            "Terrible customer service and bad product."
        ]
        
        # Sample neutral reviews
        neutral_templates = [
            "Average {product}. {feature} is okay.",
            "Decent quality for the price.",
            "It's {adjective}, but nothing special.",
            "Works as expected, no complaints.",
            "Standard {product}, does the job.",
            "Not bad, but not great either.",
            "Middle of the road {product}.",
            "It's okay. {feature} could be better.",
            "Acceptable quality, fair price.",
            "Just what I expected from this brand."
        ]
        
        features = ["battery life", "screen quality", "build quality", 
                   "sound", "camera", "performance", "design", 
                   "durability", "ease of use", "value"]
        
        products = ["phone", "laptop", "headphones", "tablet", 
                   "speaker", "camera", "watch", "charger", 
                   "case", "accessory"]
        
        adjectives = ["excellent", "fantastic", "terrible", "mediocre", 
                     "awesome", "horrible", "decent", "superb", 
                     "disappointing", "satisfactory"]
        
        days = ["few", "couple of", "several", "two", "three", 
                "five", "ten", "several", "many", "numerous"]
        
        data = []
        
        for i in range(n_samples):
            # Generate random rating (1-5) with imbalance (more 4-5 stars)
            rand = np.random.random()
            if rand < 0.55:  # 55% 5-star
                rating = 5
                templates = positive_templates
            elif rand < 0.77:  # 22% 4-star
                rating = 4
                templates = positive_templates
            elif rand < 0.88:  # 11% 3-star
                rating = 3
                templates = neutral_templates
            elif rand < 0.95:  # 7% 2-star
                rating = 2
                templates = negative_templates
            else:  # 5% 1-star
                rating = 1
                templates = negative_templates
            
            # Generate sentiment label
            sentiment = 1 if rating >= 4 else 0
            
            # Generate review text
            template = np.random.choice(templates)
            review = template.format(
                feature=np.random.choice(features),
                product=np.random.choice(products),
                adjective=np.random.choice(adjectives),
                days=np.random.choice(days)
            )
            
            # Generate title
            if rating >= 4:
                title = np.random.choice([
                    "Great product!", "Excellent!", "Love it!", 
                    "Highly recommended", "Perfect!"
                ])
            elif rating == 3:
                title = np.random.choice([
                    "It's okay", "Average product", "Not bad", 
                    "Decent", "Middle of the road"
                ])
            else:
                title = np.random.choice([
                    "Disappointed", "Terrible", "Waste of money", 
                    "Avoid!", "Not worth it"
                ])
            
            # Generate helpful votes
            helpful_total = np.random.randint(1, 100)
            helpful_yes = np.random.randint(0, helpful_total + 1)
            
            data.append({
                'rating': rating,
                'sentiment': sentiment,
                'title': title,
                'review_text': review,
                'helpful': f"{helpful_yes}/{helpful_total}"
            })
        
        df = pd.DataFrame(data)
        self.raw_data = df
        print(f"✅ Generated {len(df):,} sample reviews")
        return df
    
    def inspect_data(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Inspect data and return information
        
        Args:
            df: DataFrame to inspect (uses self.raw_data if None)
            
        Returns:
            Dictionary with data information
        """
        if df is None:
            df = self.raw_data
            
        if df is None:
            raise ValueError("No data to inspect. Load data first.")
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Value counts for categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            info['categorical_counts'] = {}
            for col in cat_cols[:3]:  # Limit to first 3 categorical columns
                info['categorical_counts'][col] = df[col].value_counts().head(10).to_dict()
        
        self.data_info = info
        return info
    
    def print_info(self, info: Optional[Dict[str, Any]] = None):
        """Print data information in a readable format"""
        if info is None:
            info = self.data_info
            
        if not info:
            print("No information available. Run inspect_data() first.")
            return
        
        print("\n" + "="*60)
        print("📊 DATA INFORMATION")
        print("="*60)
        
        print(f"\n📈 Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
        print(f"💾 Memory usage: {info['memory_usage']:.2f} MB")
        print(f"🔄 Duplicate rows: {info['duplicates']:,}")
        
        print("\n📋 Columns:")
        for i, col in enumerate(info['columns'], 1):
            dtype = info['dtypes'][col]
            missing = info['missing_values'][col]
            missing_pct = info['missing_percent'][col]
            print(f"  {i:2d}. {col:20} | {str(dtype):10} | Missing: {missing:6,} ({missing_pct:.1f}%)")
        
        if 'numeric_stats' in info:
            print("\n📊 Numeric Columns Statistics:")
            for col, stats in info['numeric_stats'].items():
                print(f"  {col}:")
                for stat, value in stats.items():
                    print(f"    {stat:10}: {value:.2f}")
        
        if 'categorical_counts' in info:
            print("\n🔤 Categorical Columns (top 10 values):")
            for col, counts in info['categorical_counts'].items():
                print(f"  {col}:")
                for val, count in list(counts.items())[:5]:
                    print(f"    '{val[:30]}...' : {count:,}")