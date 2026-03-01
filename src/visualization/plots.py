# -*- coding: utf-8 -*-
"""
Visualization Module
Handles all plotting functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from wordcloud import WordCloud
import os


class Visualizer:
    """
    Class for creating visualizations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Visualizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Set default style
        style = self.config.get('plot_style', 'seaborn-v0_8-whitegrid')
        plt.style.use(style)
        
        self.figsize = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 100)
        self.color_palette = self.config.get('color_palette', 'viridis')
        
        sns.set_palette(self.color_palette)
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, label_col: str = 'sentiment'):
        """
        Plot sentiment distribution
        
        Args:
            df: Dataframe with sentiment labels
            label_col: Name of sentiment column
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Count plot
        ax = axes[0]
        counts = df[label_col].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(['Positive', 'Negative'], counts.values, color=colors)
        ax.set_title('Sentiment Distribution', fontsize=14)
        ax.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{count:,}', ha='center', va='bottom')
        
        # Pie chart
        ax = axes[1]
        ax.pie(counts.values, labels=['Positive', 'Negative'], autopct='%1.1f%%',
               colors=colors, startangle=90, explode=(0.05, 0))
        ax.set_title('Sentiment Proportion', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_rating_distribution(self, df: pd.DataFrame, rating_col: str = 'rating'):
        """
        Plot rating distribution
        
        Args:
            df: Dataframe with ratings
            rating_col: Name of rating column
        """
    # Kiểm tra xem cột có tồn tại không
        if rating_col not in df.columns:
            print(f"  ⚠️ Column '{rating_col}' not found, skipping rating distribution plot")
            # Trả về figure rỗng
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f"No column '{rating_col}'", 
                    ha='center', va='center', fontsize=14)
            ax.set_title('Rating Distribution (Data not available)')
            return fig
    
        fig, ax = plt.subplots(figsize=self.figsize)
        
        counts = df[rating_col].value_counts().sort_index()
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
        
        bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)])
        ax.set_title('Rating Distribution', fontsize=14)
        ax.set_xlabel('Rating (stars)')
        ax.set_ylabel('Count')
        ax.set_xticks(counts.index)
        
        # Add value labels
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_review_length_distribution(self, df: pd.DataFrame, length_col: str = 'review_length'):
        """
        Plot review length distribution
        
        Args:
            df: Dataframe with review lengths
            length_col: Name of length column
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax = axes[0]
        ax.hist(df[length_col], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_title('Review Length Distribution', fontsize=14)
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Frequency')
        ax.axvline(df[length_col].mean(), color='red', linestyle='--', label=f"Mean: {df[length_col].mean():.1f}")
        ax.axvline(df[length_col].median(), color='green', linestyle='--', label=f"Median: {df[length_col].median():.1f}")
        ax.legend()
        
        # Boxplot by sentiment
        ax = axes[1]
        if 'sentiment' in df.columns:
            data = [df[df['sentiment'] == 0][length_col], df[df['sentiment'] == 1][length_col]]
            bp = ax.boxplot(data, labels=['Negative', 'Positive'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#e74c3c', '#2ecc71']):
                patch.set_facecolor(color)
            ax.set_title('Review Length by Sentiment', fontsize=14)
            ax.set_ylabel('Number of Words')
        
        plt.tight_layout()
        return fig
    
    def plot_wordcloud(self, texts: List[str], title: str = 'Word Cloud', max_words: int = 100):
        """
        Generate word cloud from texts
        
        Args:
            texts: List of text strings
            title: Title for the plot
            max_words: Maximum number of words
        """
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(all_text)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_top_words(self, 
                       feature_names: List[str],
                       coefficients: np.ndarray,
                       title: str = 'Top Words',
                       n_words: int = 20):
        """
        Plot top words by coefficient magnitude
        
        Args:
            feature_names: List of feature names
            coefficients: Coefficients array
            title: Plot title
            n_words: Number of top words to show
        """
        # Get top words by absolute coefficient
        top_indices = np.argsort(np.abs(coefficients))[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_coefs = coefficients[top_indices]
        
        # Color by sign
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_coefs]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(top_words)), top_coefs, color=colors)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(title, fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for i, (bar, coef) in enumerate(zip(bars, top_coefs)):
            ax.text(coef + (0.01 if coef > 0 else -0.05), i, f'{coef:.3f}',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str] = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
        """
        if labels is None:
            labels = ['Negative', 'Positive']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, results_df: pd.DataFrame, metric: str = 'F1-score'):
        """
        Plot model comparison bar chart
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to compare
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = results_df['Model'].tolist()
        scores = results_df[metric].tolist()
        colors = sns.color_palette(self.color_palette, len(models))
        
        bars = ax.bar(models, scores, color=colors)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14)
        ax.set_ylabel(metric)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_cluster_scatter(self, 
                             X_2d: np.ndarray, 
                             labels: np.ndarray,
                             title: str = 'Cluster Visualization',
                             highlight_noise: bool = True):
        """
        Plot 2D scatter plot of clusters
        
        Args:
            X_2d: 2D projection of data
            labels: Cluster labels
            title: Plot title
            highlight_noise: Whether to highlight noise points
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(self.color_palette, len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            if label == -1 and highlight_noise:
                # Noise points in gray
                mask = labels == label
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                          c='gray', marker='x', s=30, alpha=0.5, label='Noise')
            else:
                mask = labels == label
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                          c=[colors(i)], s=10, alpha=0.7, label=f'Cluster {label}')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette_analysis(self, 
                                 silhouette_vals: np.ndarray,
                                 labels: np.ndarray,
                                 title: str = 'Silhouette Analysis'):
        """
        Plot silhouette analysis
        
        Args:
            silhouette_vals: Silhouette values for each sample
            labels: Cluster labels
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_clusters = len(np.unique(labels))
        y_lower = 10
        
        for i in range(n_clusters):
            # Get silhouette values for cluster i
            cluster_vals = silhouette_vals[labels == i]
            cluster_vals.sort()
            
            size = len(cluster_vals)
            y_upper = y_lower + size
            
            color = plt.cm.get_cmap(self.color_palette)(i / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                            facecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, str(i))
            
            y_lower = y_upper + 10
        
        ax.axvline(x=np.mean(silhouette_vals), color='red', linestyle='--',
                  label=f'Average: {np.mean(silhouette_vals):.3f}')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_profiles(self, profile_df: pd.DataFrame):
        """
        Plot cluster profiles
        
        Args:
            profile_df: DataFrame with cluster profiles
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Cluster sizes
        ax = axes[0, 0]
        colors = sns.color_palette(self.color_palette, len(profile_df))
        bars = ax.bar(profile_df['cluster_id'], profile_df['size'], color=colors)
        ax.set_title('Cluster Sizes', fontsize=12)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        
        for bar, size in zip(bars, profile_df['size']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{size:,}', ha='center', va='bottom', fontsize=9)
        
        # Percentages
        ax = axes[0, 1]
        ax.pie(profile_df['percentage'], labels=[f'Cluster {c}' for c in profile_df['cluster_id']],
              autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Cluster Proportions', fontsize=12)
        
        # Top words (if available)
        ax = axes[1, 0]
        if 'top_words' in profile_df.columns:
            y_pos = np.arange(len(profile_df))
            ax.barh(y_pos, profile_df['size'], color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"C{c}\n{w[:20]}..." for c, w in zip(profile_df['cluster_id'], profile_df['top_words'])])
            ax.set_xlabel('Size')
            ax.set_title('Clusters with Top Words', fontsize=12)
        
        # Hide empty subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_association_rules(self, 
                               rules_df: pd.DataFrame, 
                               top_n: int = 20,
                               metric: str = 'lift'):
        """
        Plot top association rules
        
        Args:
            rules_df: Rules dataframe
            top_n: Number of top rules to show
            metric: Metric to sort by
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_rules = rules_df.nlargest(top_n, metric)
        
        if 'rule_str' not in top_rules.columns:
            if 'antecedents_str' in top_rules.columns and 'consequents_str' in top_rules.columns:
                top_rules['rule_str'] = top_rules['antecedents_str'] + ' → ' + top_rules['consequents_str']
            else:
                raise ValueError("Rules dataframe must have rule_str or antecedents_str/consequents_str")
        
        # Truncate long rules
        rules_display = [r[:50] + '...' if len(r) > 50 else r for r in top_rules['rule_str']]
        
        colors = plt.cm.get_cmap(self.color_palette)(np.linspace(0.2, 0.8, len(top_rules)))
        bars = ax.barh(range(len(rules_display)), top_rules[metric].values, color=colors)
        ax.set_yticks(range(len(rules_display)))
        ax.set_yticklabels(rules_display)
        ax.set_xlabel(metric.capitalize())
        ax.set_title(f'Top {top_n} Rules by {metric.capitalize()}', fontsize=14)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_rules[metric].values)):
            ax.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_rule_network(self, 
                          rules_df: pd.DataFrame, 
                          top_n: int = 50,
                          min_lift: float = 1.2):
        """
        Plot network graph of association rules
        
        Args:
            rules_df: Rules dataframe
            top_n: Maximum number of rules to include
            min_lift: Minimum lift to include
        """
        # Filter rules
        filtered = rules_df[rules_df['lift'] >= min_lift]
        if len(filtered) > top_n:
            filtered = filtered.nlargest(top_n, 'lift')
        
        if 'antecedents_str' not in filtered.columns or 'consequents_str' not in filtered.columns:
            raise ValueError("Rules must have antecedents_str and consequents_str")
        
        # Create graph
        G = nx.DiGraph()
        
        for _, row in filtered.iterrows():
            ant = row['antecedents_str']
            cons = row['consequents_str']
            lift = row['lift']
            
            G.add_node(ant)
            G.add_node(cons)
            G.add_edge(ant, cons, weight=lift, lift=lift)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8, ax=ax)
        
        # Draw edges with width proportional to lift
        edges = G.edges(data=True)
        widths = [min(d['lift'] * 2, 5) for (_, _, d) in edges]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color='gray', 
                              arrows=True, arrowsize=15, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title('Association Rules Network', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, 
                           train_sizes: List[int],
                           train_scores: np.ndarray,
                           test_scores: np.ndarray,
                           title: str = 'Learning Curve'):
        """
        Plot learning curve
        
        Args:
            train_sizes: Training sizes
            train_scores: Training scores
            test_scores: Test scores
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                        alpha=0.1, color='orange')
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation Score')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_semi_supervised_comparison(self, results_df: pd.DataFrame):
        """
        Plot comparison of semi-supervised vs supervised
        
        Args:
            results_df: Results dataframe
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(results_df['labeled_percent'], results_df['test_f1'], 
               'b-o', label='Semi-supervised', linewidth=2)
        ax.plot(results_df['labeled_percent'], results_df['supervised_f1'], 
               'r--s', label='Supervised', linewidth=2)
        
        # Fill improvement area
        ax.fill_between(results_df['labeled_percent'], 
                        results_df['supervised_f1'], 
                        results_df['test_f1'],
                        alpha=0.3, color='green', label='Improvement')
        
        ax.set_xlabel('Labeled Data (%)', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('Semi-supervised vs Supervised Learning', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig, filename: str, output_dir: str = 'outputs/figures/'):
        """Save figure to disk"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        print(f"✅ Saved figure: {path}")
        return path