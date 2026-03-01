# -*- coding: utf-8 -*-
"""
Report Generator Module
Handles generation of summary reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
from datetime import datetime


class ReportGenerator:
    """
    Class for generating summary reports
    """
    
    def __init__(self, output_dir: str = 'outputs/reports/'):
        """
        Initialize ReportGenerator
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_classification_report(self,
                                       results_df: pd.DataFrame,
                                       metrics: Dict[str, Any],
                                       model_name: str = 'best_model') -> str:
        """
        Generate classification report
        
        Args:
            results_df: DataFrame with results
            metrics: Metrics dictionary
            model_name: Name of the model
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'classification_report_{model_name}_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLASSIFICATION MODEL REPORT\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Model comparison
            f.write("MODEL COMPARISON\n")
            f.write("-"*40 + "\n")
            f.write(results_df.to_string())
            f.write("\n\n")
            
            # Best model metrics
            f.write("BEST MODEL METRICS\n")
            f.write("-"*40 + "\n")
            for key, value in metrics.items():
                if key != 'per_class' and key != 'confusion_matrix':
                    if isinstance(value, float):
                        f.write(f"{key:20}: {value:.4f}\n")
                    else:
                        f.write(f"{key:20}: {value}\n")
            
            # Per-class metrics
            if 'per_class' in metrics:
                f.write("\nPER-CLASS METRICS\n")
                f.write("-"*40 + "\n")
                for cls, cls_metrics in metrics['per_class'].items():
                    f.write(f"\nClass {cls}:\n")
                    for m, v in cls_metrics.items():
                        f.write(f"  {m:10}: {v:.4f}\n")
            
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                f.write("\nCONFUSION MATRIX\n")
                f.write("-"*40 + "\n")
                cm = np.array(metrics['confusion_matrix'])
                f.write("      Predicted\n")
                f.write("      ")
                for i in range(cm.shape[1]):
                    f.write(f"{i:6d}")
                f.write("\n")
                f.write("True\n")
                for i in range(cm.shape[0]):
                    f.write(f"  {i:2d}  ")
                    for j in range(cm.shape[1]):
                        f.write(f"{cm[i, j]:6d}")
                    f.write("\n")
        
        print(f"✅ Generated classification report: {report_path}")
        return report_path
    
    def generate_clustering_report(self,
                                   profile_df: pd.DataFrame,
                                   metrics: Dict[str, Any],
                                   method: str = 'kmeans') -> str:
        """
        Generate clustering report
        
        Args:
            profile_df: Cluster profiles dataframe
            metrics: Metrics dictionary
            method: Clustering method
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'clustering_report_{method}_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLUSTERING REPORT\n")
            f.write(f"Method: {method}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Metrics
            f.write("CLUSTERING METRICS\n")
            f.write("-"*40 + "\n")
            for key, value in metrics.items():
                if key not in ['cluster_sizes']:
                    if isinstance(value, float):
                        f.write(f"{key:20}: {value:.4f}\n")
                    else:
                        f.write(f"{key:20}: {value}\n")
            
            # Cluster sizes
            if 'cluster_sizes' in metrics:
                f.write("\nCLUSTER SIZES\n")
                f.write("-"*40 + "\n")
                for cls, size in metrics['cluster_sizes'].items():
                    f.write(f"Cluster {cls}: {size:,} samples\n")
            
            # Cluster profiles
            f.write("\nCLUSTER PROFILES\n")
            f.write("-"*40 + "\n")
            f.write(profile_df.to_string())
            f.write("\n")
        
        print(f"✅ Generated clustering report: {report_path}")
        return report_path
    
    def generate_association_report(self,
                                    rules_df: pd.DataFrame,
                                    insights: Dict[str, Any],
                                    top_n: int = 20) -> str:
        """
        Generate association rules report
        
        Args:
            rules_df: Rules dataframe
            insights: Insights dictionary
            top_n: Number of top rules to show
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'association_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ASSOCIATION RULES REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*40 + "\n")
            for key, value in insights.items():
                if key not in ['top_lift_rules', 'top_confidence_rules', 'top_support_rules']:
                    if isinstance(value, float):
                        f.write(f"{key:25}: {value:.4f}\n")
                    else:
                        f.write(f"{key:25}: {value}\n")
            
            # Rules by length
            if 'rules_by_length' in insights:
                f.write("\nRULES BY ANTECEDENT LENGTH\n")
                f.write("-"*40 + "\n")
                for length, count in insights['rules_by_length'].items():
                    f.write(f"{length:25}: {count:,}\n")
            
            # Top rules by lift
            if 'top_lift_rules' in insights:
                f.write("\nTOP 5 RULES BY LIFT\n")
                f.write("-"*40 + "\n")
                for i, rule in enumerate(insights['top_lift_rules'], 1):
                    f.write(f"\n{i}. {rule['rule']}\n")
                    f.write(f"   Lift: {rule['lift']:.2f}, Confidence: {rule['confidence']:.2f}, Support: {rule['support']:.4f}\n")
            
            # Top N rules
            f.write(f"\nTOP {top_n} RULES\n")
            f.write("-"*40 + "\n")
            display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
            if all(col in rules_df.columns for col in display_cols):
                top_rules = rules_df.nlargest(top_n, 'lift')[display_cols]
                f.write(top_rules.to_string())
        
        print(f"✅ Generated association report: {report_path}")
        return report_path
    
    def generate_semi_supervised_report(self,
                                        results_df: pd.DataFrame,
                                        method: str = 'self_training') -> str:
        """
        Generate semi-supervised learning report
        
        Args:
            results_df: Results dataframe
            method: Method name
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'semi_supervised_{method}_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SEMI-SUPERVISED LEARNING REPORT\n")
            f.write(f"Method: {method}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXPERIMENT RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(results_df.to_string())
            f.write("\n\n")
            
            # Best improvement
            best_idx = results_df['improvement'].idxmax()
            best_row = results_df.loc[best_idx]
            f.write("BEST IMPROVEMENT\n")
            f.write("-"*40 + "\n")
            f.write(f"Labeled {best_row['labeled_percent']}%:\n")
            f.write(f"  • Semi-supervised F1: {best_row['test_f1']:.4f}\n")
            f.write(f"  • Supervised F1: {best_row['supervised_f1']:.4f}\n")
            f.write(f"  • Improvement: {best_row['improvement']:.4f}\n")
        
        print(f"✅ Generated semi-supervised report: {report_path}")
        return report_path
    
    def generate_summary_report(self,
                                classification_results: Optional[pd.DataFrame] = None,
                                clustering_results: Optional[pd.DataFrame] = None,
                                association_results: Optional[Dict] = None,
                                semi_supervised_results: Optional[pd.DataFrame] = None) -> str:
        """
        Generate comprehensive summary report
        
        Args:
            classification_results: Classification results
            clustering_results: Clustering results
            association_results: Association results
            semi_supervised_results: Semi-supervised results
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'project_summary_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SENTIMENT ANALYSIS PROJECT - FINAL SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Classification summary
            if classification_results is not None:
                f.write("CLASSIFICATION\n")
                f.write("-"*40 + "\n")
                f.write("Model Comparison:\n")
                f.write(classification_results.to_string())
                f.write("\n\n")
            
            # Clustering summary
            if clustering_results is not None:
                f.write("CLUSTERING\n")
                f.write("-"*40 + "\n")
                f.write("Cluster Profiles:\n")
                f.write(clustering_results.to_string())
                f.write("\n\n")
            
            # Association rules summary
            if association_results is not None:
                f.write("ASSOCIATION RULES\n")
                f.write("-"*40 + "\n")
                for key, value in association_results.items():
                    if key not in ['top_lift_rules']:
                        if isinstance(value, float):
                            f.write(f"{key:25}: {value:.4f}\n")
                        else:
                            f.write(f"{key:25}: {value}\n")
                f.write("\n")
            
            # Semi-supervised summary
            if semi_supervised_results is not None:
                f.write("SEMI-SUPERVISED LEARNING\n")
                f.write("-"*40 + "\n")
                f.write(semi_supervised_results.to_string())
                f.write("\n")
        
        print(f"✅ Generated summary report: {report_path}")
        return report_path