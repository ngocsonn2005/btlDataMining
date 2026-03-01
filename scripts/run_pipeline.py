#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Full Pipeline Script
Executes the entire sentiment analysis pipeline
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationRuleMiner
from src.mining.clustering import TextClusterer
from src.models.supervised import SupervisedClassifier
from src.models.semi_supervised import SemiSupervisedClassifier
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.report import ReportGenerator
from src.visualization.plots import Visualizer


def load_config(config_path='configs/params.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main pipeline execution"""
    
    print("="*80)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*80)
    
    # Load configuration
    config = load_config()
    print(f"\n✅ Loaded configuration from configs/params.yaml")
    
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # ============================================================
    # STEP 1: DATA LOADING
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)
    
    loader = DataLoader('configs/params.yaml')
    
    # Cấu hình số dòng cần đọc (có thể điều chỉnh)
    # Đặt thành None để đọc toàn bộ file
    NROWS_TO_READ = 50000  # Đọc 50,000 dòng đầu tiên
    # NROWS_TO_READ = None  # Bỏ comment dòng này để đọc toàn bộ file
    
    # Try to load real data, fall back to sample data if not available
    data_path = config['data']['raw_train_path']
    if os.path.exists(data_path):
        if NROWS_TO_READ:
            print(f"\n⚠️ Chỉ đọc {NROWS_TO_READ:,} dòng do dung lượng lớn")
            df = loader.load_csv(data_path, nrows=NROWS_TO_READ)
        else:
            df = loader.load_csv(data_path)
    else:
        print(f"⚠️ Data file not found: {data_path}")
        print("📊 Generating sample data for testing...")
        df = loader.load_sample_data(n_samples=10000)
    
    info = loader.inspect_data()
    loader.print_info(info)
    
    # ============================================================
    # STEP 2: DATA CLEANING
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: DATA CLEANING")
    print("="*60)
    
    cleaner = DataCleaner(config['preprocessing'])
    df_clean = cleaner.clean_dataframe(
        df,
        text_column=config['preprocessing'].get('text_column', 'review_text'),  # Sửa từ 'review' thành 'review_text'
        rating_column=config['preprocessing'].get('rating_column', 'rating'),
        helpful_column='helpful' if 'helpful' in df.columns else None
    )
    
    # Save cleaned data
    cleaner.save_cleaned_data('data/processed/cleaned_data.csv')
    
    # Extract texts and labels
    texts, labels = cleaner.get_texts_and_labels()
    
    # ============================================================
    # STEP 3: FEATURE ENGINEERING
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    feature_builder = FeatureBuilder(config['features'])
    
    # TF-IDF features
    tfidf_params = config['features']['tfidf']
    X_tfidf = feature_builder.build_tfidf_features(
        texts,
        max_features=tfidf_params['max_features'],
        ngram_range=tuple(tfidf_params['ngram_range']),
        min_df=tfidf_params['min_df'],
        max_df=tfidf_params['max_df']
    )
    
    # Word2Vec features (optional - can be slow on large data)
    use_word2vec = False
    if use_word2vec:
        w2v_params = config['features']['word2vec']
        X_w2v = feature_builder.build_word2vec_features(
            texts,
            vector_size=w2v_params['vector_size'],
            window=w2v_params['window'],
            min_count=w2v_params['min_count'],
            workers=w2v_params['workers'],
            epochs=w2v_params['epochs']
        )
        X_combined = feature_builder.combine_features([X_tfidf, X_w2v])
    else:
        X_combined = X_tfidf
    
    # Scale features
    X_scaled = feature_builder.scale_features(X_combined, method='standard')
    
    # Save feature models
    feature_builder.save_models('outputs/models/')
    
    # ============================================================
    # STEP 4: ASSOCIATION RULES MINING
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: ASSOCIATION RULES MINING")
    print("="*60)
    
    assoc_config = config['association']
    assoc_miner = AssociationRuleMiner(assoc_config)
    
    # Prepare transactions (use positive reviews only)
    pos_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1][:5000]  # Limit for performance
    transactions = assoc_miner.prepare_transactions(
        pos_texts,
        top_words=500,
        min_word_freq=5
    )
    
    # Encode transactions
    onehot_df = assoc_miner.encode_transactions()
    
    # Mine frequent itemsets
    frequent_itemsets = assoc_miner.mine_frequent_itemsets(
        min_support=assoc_config['min_support'],
        max_len=assoc_config['max_len'],
        algorithm='fpgrowth'
    )
    
    # Generate rules
    rules = assoc_miner.generate_rules(
        metric=assoc_config['metric'],
        min_threshold=assoc_config['min_threshold']
    )
    
    # Add readable columns
    rules = assoc_miner.add_readable_columns()
    
    # Filter rules
    filtered_rules = assoc_miner.filter_rules(
        min_support=assoc_config.get('filter_min_support'),
        min_confidence=assoc_config.get('filter_min_confidence'),
        min_lift=assoc_config.get('filter_min_lift'),
        max_antecedents=assoc_config.get('filter_max_antecedents'),
        max_consequents=assoc_config.get('filter_max_consequents')
    )
    
    # Extract insights
    insights = assoc_miner.extract_insights(filtered_rules)
    
    # Save results
    assoc_miner.save_results('outputs/tables/')
    
    # ============================================================
    # STEP 5: CLUSTERING
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: CLUSTERING")
    print("="*60)
    
    cluster_config = config['clustering']
    clusterer = TextClusterer(cluster_config)
    
    # Find optimal k
    k_results = clusterer.find_optimal_k(
        X_scaled,
        k_min=cluster_config['k_min'],
        k_max=cluster_config['k_max'],
        random_state=cluster_config['random_state']
    )
    
    # Determine number of clusters
    n_clusters = cluster_config.get('n_clusters')
    if n_clusters is None:
        n_clusters = int(k_results.loc[k_results['silhouette'].idxmax(), 'k'])
    
    print(f"\n📊 Using k={n_clusters} for clustering")
    
    # Fit K-Means
    labels_cluster = clusterer.fit_kmeans(
        X_scaled,
        n_clusters=n_clusters,
        random_state=cluster_config['random_state']
    )
    
    # Get topic words
    topic_words = clusterer.get_topic_words(
        X_scaled,
        feature_builder.feature_names,
        n_words=10
    )
    
    # Create cluster profiles
    profiles = clusterer.profile_clusters(
        X_scaled,
        feature_builder.feature_names,
        texts,
        n_examples=3
    )
    
    # Save model
    clusterer.save_model('outputs/models/kmeans_model.pkl')
    
    # ============================================================
    # STEP 6: SUPERVISED CLASSIFICATION
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: SUPERVISED CLASSIFICATION")
    print("="*60)
    
    classifier = SupervisedClassifier(config['classification'])
    
    # Split data
    X_train, X_test, y_train, y_test = classifier.split_data(
        X_scaled, labels,
        test_size=0.2,
        random_state=config['classification']['random_state']
    )
    
    # Train models
    results = {}
    
    # Naive Bayes
    if 'naive_bayes' in config['classification']['models']:
        results['naive_bayes'] = classifier.train_naive_bayes(X_train, y_train, X_test, y_test)
    
    # Logistic Regression
    if 'logistic_regression' in config['classification']['models']:
        results['logistic_regression'] = classifier.train_logistic_regression(
            X_train, y_train, X_test, y_test,
            C=1.0, max_iter=1000
        )
    
    # SVM
    if 'svm' in config['classification']['models']:
        results['svm'] = classifier.train_svm(X_train, y_train, X_test, y_test, C=1.0)
    
    # Random Forest
    if 'random_forest' in config['classification']['models']:
        results['random_forest'] = classifier.train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=100, max_depth=10
        )
    
    # Compare models
    comparison_df = classifier.compare_models()
    print("\n" + comparison_df.to_string())
    
    # Save best model
    if classifier.best_model_name:
        classifier.save_model(
            classifier.best_model_name,
            f'outputs/models/best_{classifier.best_model_name}.pkl'
        )
    
    # ============================================================
    # STEP 7: SEMI-SUPERVISED LEARNING
    # ============================================================
    print("\n" + "="*60)
    print("STEP 7: SEMI-SUPERVISED LEARNING")
    print("="*60)
    
    semi_config = config['semi_supervised']
    semi_classifier = SemiSupervisedClassifier(semi_config)
    
    # Use best model as base estimator
    from sklearn.linear_model import LogisticRegression
    base_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    
    # Run self-training experiment
    self_training_results = semi_classifier.run_experiment(
        X_scaled, labels,
        base_estimator,
        labeled_percents=semi_config['labeled_percents'],
        method='self_training',
        random_state=semi_config['random_state']
    )
    
    print("\n" + self_training_results.to_string())
    
    # Save results
    semi_classifier.save_results('outputs/tables/')
    
    # ============================================================
    # STEP 8: GENERATE REPORTS
    # ============================================================
    print("\n" + "="*60)
    print("STEP 8: GENERATING REPORTS")
    print("="*60)
    
    report_gen = ReportGenerator('outputs/reports/')
    
    # Classification report
    if classifier.best_model_name:
        best_metrics = results[classifier.best_model_name]
        report_gen.generate_classification_report(
            comparison_df,
            best_metrics,
            classifier.best_model_name
        )
    
    # Clustering report
    cluster_metrics = MetricsCalculator.clustering_metrics(X_scaled, labels_cluster)
    report_gen.generate_clustering_report(profiles, cluster_metrics, 'kmeans')
    
    # Association report
    report_gen.generate_association_report(filtered_rules, insights, top_n=20)
    
    # Semi-supervised report
    report_gen.generate_semi_supervised_report(self_training_results, 'self_training')
    
    # Summary report
    report_gen.generate_summary_report(
        classification_results=comparison_df,
        clustering_results=profiles,
        association_results=insights,
        semi_supervised_results=self_training_results
    )
    
    # ============================================================
    # STEP 9: VISUALIZATIONS
    # ============================================================
    print("\n" + "="*60)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("="*60)
    
    viz = Visualizer(config['visualization'])
    
    # EDA plots
    fig = viz.plot_sentiment_distribution(df_clean)
    viz.save_figure(fig, 'sentiment_distribution.png')
    
    try:
        fig = viz.plot_rating_distribution(df_clean)
        viz.save_figure(fig, 'rating_distribution.png')
    except Exception as e:
        print(f"  ⚠️ Could not plot rating distribution: {e}")
    
    fig = viz.plot_review_length_distribution(df_clean)
    viz.save_figure(fig, 'review_length_distribution.png')
    
    # Word cloud
    fig = viz.plot_wordcloud(texts[:5000], title='Word Cloud - All Reviews')
    viz.save_figure(fig, 'wordcloud_all.png')
    
    # Association rules
    fig = viz.plot_association_rules(filtered_rules, top_n=20, metric='lift')
    viz.save_figure(fig, 'top_rules.png')
    
    fig = viz.plot_rule_network(filtered_rules, top_n=30, min_lift=1.2)
    viz.save_figure(fig, 'rule_network.png')
    
    # Clustering
    X_2d = clusterer.project_2d(X_scaled, method='pca')
    fig = viz.plot_cluster_scatter(X_2d, labels_cluster, title='K-Means Clusters')
    viz.save_figure(fig, 'cluster_scatter.png')
    
    fig = viz.plot_cluster_profiles(profiles)
    viz.save_figure(fig, 'cluster_profiles.png')
    
    # Classification
    if classifier.best_model_name:
        fig = viz.plot_model_comparison(comparison_df, metric='F1-score')
        viz.save_figure(fig, 'model_comparison.png')
        
        # Confusion matrix
        best_metrics = results[classifier.best_model_name]
        if 'confusion_matrix' in best_metrics:
            cm = np.array(best_metrics['confusion_matrix'])
            fig = viz.plot_confusion_matrix(cm)
            viz.save_figure(fig, f'confusion_matrix_{classifier.best_model_name}.png')
    
    # Semi-supervised
    fig = viz.plot_semi_supervised_comparison(self_training_results)
    viz.save_figure(fig, 'semi_supervised_comparison.png')
    
    # ============================================================
    # PIPELINE COMPLETE
    # ============================================================
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\n📁 Outputs saved to:")
    print("  • Data: data/processed/")
    print("  • Tables: outputs/tables/")
    print("  • Figures: outputs/figures/")
    print("  • Models: outputs/models/")
    print("  • Reports: outputs/reports/")
    print("\n🏁 Done!")


if __name__ == "__main__":
    main()