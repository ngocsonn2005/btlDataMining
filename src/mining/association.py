# -*- coding: utf-8 -*-
"""
Association Rule Mining Module
Handles mining of association rules from text data
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import os


class AssociationRuleMiner:
    """
    Class for mining association rules from text data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AssociationRuleMiner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.transactions = None
        self.onehot_df = None
        self.frequent_itemsets = None
        self.rules = None
        
    def prepare_transactions(self, 
                             texts: List[str],
                             top_words: int = 1000,
                             min_word_freq: int = 10) -> List[List[str]]:
        """
        Prepare transactions from texts (each document is a transaction, words are items)
        
        Args:
            texts: List of text strings
            top_words: Number of top words to keep
            min_word_freq: Minimum word frequency
            
        Returns:
            List of transactions (each transaction is a list of words)
        """
        print("🔄 Preparing transactions...")
        
        # Tokenize texts
        transactions = [text.split() for text in texts]
        
        # Filter words by frequency
        word_freq = {}
        for trans in transactions:
            for word in set(trans):  # Use set to avoid counting duplicates in same doc
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words_list = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_words]
        top_words_set = set([word for word, freq in top_words_list if freq >= min_word_freq])
        
        # Filter transactions
        filtered_transactions = []
        for trans in transactions:
            filtered = [word for word in trans if word in top_words_set]
            if filtered:
                filtered_transactions.append(list(set(filtered)))  # Remove duplicates within transaction
        
        self.transactions = filtered_transactions
        print(f"  • Number of transactions: {len(self.transactions):,}")
        print(f"  • Number of unique items: {len(top_words_set):,}")
        
        return self.transactions
    
    def encode_transactions(self, transactions: Optional[List[List[str]]] = None) -> pd.DataFrame:
        """
        Encode transactions to one-hot format
        
        Args:
            transactions: List of transactions (uses self.transactions if None)
            
        Returns:
            One-hot encoded dataframe
        """
        if transactions is None:
            transactions = self.transactions
            
        if transactions is None:
            raise ValueError("No transactions available")
        
        print("🔄 Encoding transactions...")
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.onehot_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"  • Shape: {self.onehot_df.shape}")
        return self.onehot_df
    
    def mine_frequent_itemsets(self,
                               min_support: float = 0.01,
                               max_len: int = 3,
                               algorithm: str = 'fpgrowth') -> pd.DataFrame:
        """
        Mine frequent itemsets
        
        Args:
            min_support: Minimum support
            max_len: Maximum itemset length
            algorithm: Algorithm to use ('apriori' or 'fpgrowth')
            
        Returns:
            DataFrame of frequent itemsets
        """
        if self.onehot_df is None:
            raise ValueError("No encoded data available. Run encode_transactions() first.")
        
        print(f"🔄 Mining frequent itemsets (min_support={min_support}, algorithm={algorithm})...")
        start_time = time.time()
        
        if algorithm.lower() == 'apriori':
            self.frequent_itemsets = apriori(
                self.onehot_df,
                min_support=min_support,
                use_colnames=True,
                max_len=max_len
            )
        else:  # fpgrowth
            self.frequent_itemsets = fpgrowth(
                self.onehot_df,
                min_support=min_support,
                use_colnames=True,
                max_len=max_len
            )
        
        elapsed_time = time.time() - start_time
        print(f"  • Found {len(self.frequent_itemsets):,} frequent itemsets")
        print(f"  • Time: {elapsed_time:.2f} seconds")
        
        return self.frequent_itemsets
    
    def generate_rules(self,
                       metric: str = 'lift',
                       min_threshold: float = 1.0) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets
        
        Args:
            metric: Metric to use ('lift', 'confidence', 'support')
            min_threshold: Minimum threshold for metric
            
        Returns:
            DataFrame of association rules
        """
        if self.frequent_itemsets is None:
            raise ValueError("No frequent itemsets available. Run mine_frequent_itemsets() first.")
        
        print(f"🔄 Generating rules (metric={metric}, min_threshold={min_threshold})...")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        # Sort by lift
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        print(f"  • Found {len(self.rules):,} rules")
        
        return self.rules
    
    @staticmethod
    def _frozenset_to_str(fs):
        """Convert frozenset to readable string"""
        return ", ".join(sorted(list(fs)))
    
    def add_readable_columns(self) -> pd.DataFrame:
        """Add readable columns to rules dataframe"""
        if self.rules is None:
            raise ValueError("No rules available")
        
        self.rules['antecedents_str'] = self.rules['antecedents'].apply(self._frozenset_to_str)
        self.rules['consequents_str'] = self.rules['consequents'].apply(self._frozenset_to_str)
        self.rules['rule_str'] = self.rules['antecedents_str'] + " → " + self.rules['consequents_str']
        
        return self.rules
    
    def filter_rules(self,
                     min_support: Optional[float] = None,
                     min_confidence: Optional[float] = None,
                     min_lift: Optional[float] = None,
                     max_antecedents: Optional[int] = None,
                     max_consequents: Optional[int] = None) -> pd.DataFrame:
        """
        Filter rules by various metrics
        
        Args:
            min_support: Minimum support
            min_confidence: Minimum confidence
            min_lift: Minimum lift
            max_antecedents: Maximum number of antecedents
            max_consequents: Maximum number of consequents
            
        Returns:
            Filtered rules dataframe
        """
        if self.rules is None:
            raise ValueError("No rules available")
        
        filtered = self.rules.copy()
        
        if min_support is not None:
            filtered = filtered[filtered['support'] >= min_support]
        
        if min_confidence is not None:
            filtered = filtered[filtered['confidence'] >= min_confidence]
        
        if min_lift is not None:
            filtered = filtered[filtered['lift'] >= min_lift]
        
        if max_antecedents is not None:
            filtered = filtered[filtered['antecedents'].apply(len) <= max_antecedents]
        
        if max_consequents is not None:
            filtered = filtered[filtered['consequents'].apply(len) <= max_consequents]
        
        print(f"  • Filtered from {len(self.rules)} to {len(filtered)} rules")
        
        return filtered
    
    def get_top_rules(self, n: int = 20, by: str = 'lift') -> pd.DataFrame:
        """Get top N rules by specified metric"""
        if self.rules is None:
            raise ValueError("No rules available")
        
        return self.rules.sort_values(by, ascending=False).head(n)
    
    def extract_insights(self, rules_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Extract insights from rules
        
        Args:
            rules_df: Rules dataframe (uses self.rules if None)
            
        Returns:
            Dictionary of insights
        """
        if rules_df is None:
            rules_df = self.rules
            
        if rules_df is None:
            raise ValueError("No rules available")
        
        insights = {
            'total_rules': len(rules_df),
            'rules_by_length': {},
            'top_lift_rules': [],
            'top_confidence_rules': [],
            'top_support_rules': [],
            'avg_lift': rules_df['lift'].mean(),
            'avg_confidence': rules_df['confidence'].mean(),
            'avg_support': rules_df['support'].mean(),
            'max_lift': rules_df['lift'].max(),
            'max_confidence': rules_df['confidence'].max(),
            'max_support': rules_df['support'].max()
        }
        
        # Rules by antecedent length
        for i in range(1, 4):
            mask = rules_df['antecedents'].apply(len) == i
            insights['rules_by_length'][f'{i}-item antecedents'] = mask.sum()
        
        # Top rules
        top_lift = rules_df.nlargest(5, 'lift')
        insights['top_lift_rules'] = [
            {
                'rule': f"{self._frozenset_to_str(row['antecedents'])} → {self._frozenset_to_str(row['consequents'])}",
                'lift': row['lift'],
                'confidence': row['confidence'],
                'support': row['support']
            }
            for _, row in top_lift.iterrows()
        ]
        
        return insights
    
    def save_results(self, output_dir: str):
        """Save results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.frequent_itemsets is not None:
            self.frequent_itemsets.to_csv(os.path.join(output_dir, 'frequent_itemsets.csv'), index=False)
        
        if self.rules is not None:
            self.rules.to_csv(os.path.join(output_dir, 'association_rules.csv'), index=False)
        
        print(f"✅ Saved association results to {output_dir}")