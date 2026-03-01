# -*- coding: utf-8 -*-
"""
Streamlit Dashboard for Sentiment Analysis Project
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import os
import sys
from datetime import datetime
import joblib
import re
import glob
from collections import Counter

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-text {
        background-color: #e8f4f8;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def find_file_paths():
    """Tìm tất cả các đường dẫn có thể chứa dữ liệu"""
    
    paths = {
        'project_root': os.getcwd(),
        'parent_dir': os.path.dirname(os.getcwd()),
        'data_mining_root': 'E:/Data Mining'
    }
    
    search_locations = []
    
    # Thêm các đường dẫn có thể
    for name, base_path in paths.items():
        # outputs/tables trong từng thư mục
        search_locations.append(os.path.join(base_path, 'outputs', 'tables'))
        search_locations.append(os.path.join(base_path, 'btlDataMining', 'outputs', 'tables'))
    
    # Thêm đường dẫn tuyệt đối
    search_locations.extend([
        'E:/Data Mining/outputs/tables',
        'E:/Data Mining/btlDataMining/outputs/tables',
        'outputs/tables',
        '../outputs/tables'
    ])
    
    # Loại bỏ duplicates và kiểm tra tồn tại
    valid_locations = []
    for loc in set(search_locations):
        if os.path.exists(loc):
            valid_locations.append(loc)
            print(f"📁 Found directory: {loc}")
    
    return valid_locations


def find_reports_paths():
    """Tìm đường dẫn chứa reports"""
    
    search_locations = [
        'outputs/reports',
        '../outputs/reports',
        'E:/Data Mining/outputs/reports',
        'E:/Data Mining/btlDataMining/outputs/reports'
    ]
    
    valid_locations = []
    for loc in search_locations:
        if os.path.exists(loc):
            valid_locations.append(loc)
    
    return valid_locations


@st.cache_data
def load_all_data():
    """Load tất cả dữ liệu từ mọi vị trí có thể"""
    
    data = {}
    found_files = []
    
    # Tìm các thư mục chứa dữ liệu
    table_dirs = find_file_paths()
    report_dirs = find_reports_paths()
    
    # Định nghĩa các file cần tìm
    files_to_find = {
        'association': ['association_rules.csv'],
        'frequent_itemsets': ['frequent_itemsets.csv'],
        'semi_self': ['self_training_results.csv', 'semi_supervised_self.csv'],
        'classification': ['model_comparison.csv', 'classification_results.csv'],
        'clustering': ['cluster_profiles.csv', 'clustering_results.csv'],
        'executive': ['executive_summary.csv']
    }
    
    # Tìm trong tất cả các thư mục tables
    for table_dir in table_dirs:
        print(f"\n📁 Scanning: {table_dir}")
        for data_type, filenames in files_to_find.items():
            if data_type in data:
                continue  # Đã tìm thấy
            for filename in filenames:
                file_path = os.path.join(table_dir, filename)
                if os.path.exists(file_path):
                    try:
                        data[data_type] = pd.read_csv(file_path)
                        found_files.append(f"✅ {data_type}: {file_path}")
                        print(f"  ✅ Found: {filename}")
                        break
                    except Exception as e:
                        print(f"  ❌ Error loading {filename}: {e}")
    
    # Tìm trong reports để lấy thông tin clustering
    if 'clustering' not in data:
        for report_dir in report_dirs:
            if os.path.exists(report_dir):
                cluster_reports = glob.glob(os.path.join(report_dir, '*clustering*.txt')) + \
                                 glob.glob(os.path.join(report_dir, '*cluster*.txt'))
                
                if cluster_reports:
                    # Lấy file mới nhất
                    latest_report = max(cluster_reports, key=os.path.getctime)
                    try:
                        with open(latest_report, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # Parse cluster sizes
                            cluster_sizes = re.findall(r'Cluster (\d+): (\d+,?\d*) samples', content)
                            if cluster_sizes:
                                cluster_data = []
                                total = sum(int(s.replace(',', '')) for _, s in cluster_sizes)
                                
                                for cluster_id, size in cluster_sizes:
                                    size_val = int(size.replace(',', ''))
                                    
                                    # Xác định tên cluster dựa trên cluster_id
                                    if cluster_id == '0':
                                        name = "Book/Fiction Reviews"
                                        desc = "Reviews about books, fiction, novels, science fiction"
                                    elif cluster_id == '1':
                                        name = "Product/CD Reviews"
                                        desc = "Reviews about products, CDs, albums, songs"
                                    else:
                                        name = f"Cluster {cluster_id}"
                                        desc = f"Reviews in cluster {cluster_id}"
                                    
                                    # Tìm top words nếu có
                                    top_words_match = re.search(rf'Cluster {cluster_id}.*?top_words: (.*?)(?:\n|$)', content, re.DOTALL)
                                    top_words = top_words_match.group(1).strip() if top_words_match else ""
                                    
                                    cluster_data.append({
                                        'cluster': int(cluster_id),
                                        'name': name,
                                        'description': desc,
                                        'top_words': top_words,
                                        'size': size_val,
                                        'percentage': round(size_val / total * 100, 1)
                                    })
                                
                                data['clustering'] = pd.DataFrame(cluster_data)
                                found_files.append(f"✅ clustering: parsed from {latest_report}")
                                break
                    except Exception as e:
                        print(f"  ❌ Error parsing report: {e}")
    
    # Tìm trong classification reports
    if 'classification' not in data:
        for report_dir in report_dirs:
            class_reports = glob.glob(os.path.join(report_dir, '*classification*.txt'))
            if class_reports:
                latest_report = max(class_reports, key=os.path.getctime)
                try:
                    with open(latest_report, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Parse model comparison
                        model_lines = re.findall(r'(\d+)\s+(\w+(?:_\w+)?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', content)
                        if model_lines:
                            class_data = []
                            for line in model_lines:
                                class_data.append({
                                    'Model': line[1].replace('_', ' ').title(),
                                    'Accuracy': float(line[2]),
                                    'Precision': float(line[3]),
                                    'Recall': float(line[4]),
                                    'F1-Score': float(line[5])
                                })
                            data['classification'] = pd.DataFrame(class_data)
                            found_files.append(f"✅ classification: parsed from {latest_report}")
                            break
                except:
                    pass
    
    # Hiển thị kết quả tìm kiếm
    if found_files:
        st.sidebar.success("✅ Data files found:")
        for msg in found_files[:5]:  # Chỉ hiển thị 5 file đầu
            st.sidebar.info(msg)
    else:
        st.sidebar.warning("⚠️ No data files found. Using sample data.")
        data = create_sample_data()
    
    return data


def create_sample_data():
    """Tạo dữ liệu mẫu từ các file có sẵn"""
    
    data = {}
    
    # Thử đọc association_rules từ nhiều vị trí
    for path in ['outputs/tables/association_rules.csv', 
                 '../outputs/tables/association_rules.csv',
                 'E:/Data Mining/outputs/tables/association_rules.csv']:
        if os.path.exists(path):
            try:
                assoc_df = pd.read_csv(path)
                # Đảm bảo có các cột cần thiết
                if 'antecedents_str' not in assoc_df.columns:
                    assoc_df['antecedents_str'] = assoc_df.get('antecedents', '').astype(str)
                if 'consequents_str' not in assoc_df.columns:
                    assoc_df['consequents_str'] = assoc_df.get('consequents', '').astype(str)
                data['association'] = assoc_df
                break
            except:
                pass
    
    # Thử đọc frequent_itemsets
    for path in ['outputs/tables/frequent_itemsets.csv',
                 '../outputs/tables/frequent_itemsets.csv']:
        if os.path.exists(path):
            try:
                data['frequent_itemsets'] = pd.read_csv(path)
                break
            except:
                pass
    
    # Thử đọc self_training_results
    for path in ['outputs/tables/self_training_results.csv',
                 '../outputs/tables/self_training_results.csv']:
        if os.path.exists(path):
            try:
                semi_df = pd.read_csv(path)
                # Đảm bảo có các cột cần thiết
                if 'test_f1' in semi_df.columns:
                    semi_df['f1'] = semi_df['test_f1']
                if 'supervised_f1' not in semi_df.columns and 'supervised_f1' not in semi_df.columns:
                    semi_df['supervised_f1'] = semi_df.get('supervised_f1', semi_df.get('supervised_f1', 0))
                data['semi_self'] = semi_df
                break
            except:
                pass
    
    # Tạo classification data nếu chưa có
    if 'classification' not in data:
        data['classification'] = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'XGBoost', 'LSTM'],
            'Accuracy': [0.83, 0.77, 0.77, 0.82, 0.85],
            'Precision': [0.83, 0.74, 0.77, 0.82, 0.85],
            'Recall': [0.83, 0.83, 0.80, 0.82, 0.85],
            'F1-Score': [0.83, 0.78, 0.78, 0.82, 0.85],
            'Train Time (s)': [1.1, 4.5, 0.9, 2.8, 8.5]
        })
    
    # Tạo clustering data nếu chưa có
    if 'clustering' not in data:
        data['clustering'] = pd.DataFrame({
            'cluster': [0, 1],
            'name': ['Book Reviews', 'Product Reviews'],
            'description': ['Reviews about books, fiction, novels', 'Reviews about products, CDs, albums'],
            'size': [1151, 48849],
            'percentage': [2.3, 97.7],
            'top_words': ['book, read, story, fiction, novel', 'product, cd, buy, album, song']
        })
    
    # Tạo association data nếu chưa có
    if 'association' not in data:
        data['association'] = pd.DataFrame({
            'antecedents_str': ['book', 'read'],
            'consequents_str': ['read', 'book'],
            'support': [0.1817, 0.1817],
            'confidence': [0.5477, 0.7819],
            'lift': [2.3575, 2.3575]
        })
    
    # Tạo semi-supervised data nếu chưa có
    if 'semi_self' not in data:
        data['semi_self'] = pd.DataFrame({
            'labeled_percent': [5, 10, 20, 30, 50],
            'f1': [0.7205, 0.7654, 0.8029, 0.8126, 0.8236],
            'supervised_f1': [0.7192, 0.7716, 0.8068, 0.8167, 0.8249],
            'improvement': [0.0013, -0.0061, -0.0039, -0.0041, -0.0013]
        })
    
    return data


def display_sidebar(data):
    """Display sidebar with navigation"""
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/amazon.png", width=80)
        st.markdown("## 📊 Sentiment Analysis")
        st.markdown("**Amazon Reviews Project**")
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Select page:",
            ["🏠 Overview", 
             "🔗 Association Rules", 
             "🔍 Clustering", 
             "🤖 Classification",
             "🔄 Semi-Supervised",
             "📋 Reports"]
        )
        
        st.markdown("---")
        
        # Data status
        st.markdown("### 💾 Data Status")
        if data:
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                if 'classification' in data:
                    st.success("✅ Classification")
                if 'clustering' in data:
                    st.success("✅ Clustering")
            with status_col2:
                if 'association' in data:
                    st.success("✅ Association")
                if 'semi_self' in data:
                    st.success("✅ Semi-Sup")
        else:
            st.warning("⚠️ No data")
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ Info")
        st.markdown("**Course:** Data Mining")
        st.markdown("**Group:** 7")
        st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        return page


def display_overview(data):
    """Overview page"""
    
    st.markdown("<h1 class='main-header'>📊 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("### Amazon Customer Reviews Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'classification' in data and not data['classification'].empty:
                best_f1 = data['classification']['F1-Score'].max()
                st.metric("Best F1-Score", f"{best_f1:.4f}")
            else:
                st.metric("Best F1-Score", "0.8302")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'clustering' in data and not data['clustering'].empty:
                n_clusters = len(data['clustering'])
                st.metric("Number of Clusters", n_clusters)
            else:
                st.metric("Number of Clusters", "2")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'association' in data and not data['association'].empty:
                n_rules = len(data['association'])
                st.metric("Association Rules", n_rules)
            else:
                st.metric("Association Rules", "2")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'semi_self' in data and not data['semi_self'].empty:
                best_imp = data['semi_self']['improvement'].max()
                st.metric("Best Improvement", f"{best_imp:.4f}")
            else:
                st.metric("Best Improvement", "0.0013")
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Objectives")
        st.markdown("""
        - **Association Rule Mining**: Discover patterns in reviews
        - **Clustering**: Group reviews into meaningful topics
        - **Classification**: Predict sentiment (positive/negative)
        - **Semi-Supervised Learning**: Handle limited labeled data
        """)
    
    with col2:
        st.markdown("### 🔧 Technologies Used")
        st.markdown("""
        - Python 3.9+ | Pandas | NumPy
        - Scikit-learn | XGBoost | TensorFlow
        - NLTK | Gensim | WordCloud
        - MLxtend (Association Rules)
        - Streamlit | Plotly | Matplotlib
        """)
    
    # Key findings
    st.markdown("### 📊 Key Findings")
    
    findings = []
    
    # Classification findings
    if 'classification' in data and not data['classification'].empty:
        best_idx = data['classification']['F1-Score'].idxmax()
        best = data['classification'].iloc[best_idx]
        findings.append(f"🏆 **Best Model**: {best['Model']} (F1-Score = {best['F1-Score']:.4f})")
    
    # Clustering findings
    if 'clustering' in data and not data['clustering'].empty:
        cluster_info = data['clustering'].iloc[0] if len(data['clustering']) > 0 else None
        if cluster_info is not None:
            cluster_pct = cluster_info.get('percentage', 2.3)
            findings.append(f"📚 **Clustering**: {len(data['clustering'])} clusters - {cluster_info['name']} ({cluster_pct}%)")
    
    # Association findings
    if 'association' in data and not data['association'].empty:
        top_rule = data['association'].iloc[0]
        ant = top_rule.get('antecedents_str', f"Rule 1")
        cons = top_rule.get('consequents_str', f"Rule 1")
        lift = top_rule.get('lift', 2.36)
        findings.append(f"🔗 **Strongest Rule**: '{ant} → {cons}' (lift = {lift:.2f})")
    
    # Semi-supervised findings
    if 'semi_self' in data and not data['semi_self'].empty:
        best_idx = data['semi_self']['improvement'].idxmax()
        best_imp = data['semi_self'].iloc[best_idx]
        findings.append(f"🔄 **Semi-Supervised**: {best_imp['improvement']:.4f} improvement with {best_imp['labeled_percent']}% labeled data")
    
    # Default findings if none
    if not findings:
        findings = [
            "🏆 **Best Model**: Logistic Regression (F1-Score = 0.8302)",
            "📚 **Clustering**: 2 clusters - Book Reviews (2.3%)",
            "🔗 **Strongest Rule**: 'book → read' (lift = 2.36)",
            "🔄 **Semi-Supervised**: 0.0013 improvement with 5% labeled data"
        ]
    
    for finding in findings:
        st.markdown(f"<div class='insight-text'>{finding}</div>", unsafe_allow_html=True)


def display_association(data):
    """Association Rules page"""
    
    st.markdown("<h2 class='sub-header'>🔗 Association Rules Mining</h2>", unsafe_allow_html=True)
    
    if 'association' in data and not data['association'].empty:
        rules_df = data['association'].copy()
        
        st.markdown(f"#### 📊 Total Rules: {len(rules_df)}")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Lift", f"{rules_df['lift'].mean():.3f}")
        with col2:
            st.metric("Average Confidence", f"{rules_df['confidence'].mean():.3f}")
        with col3:
            st.metric("Average Support", f"{rules_df['support'].mean():.3f}")
        
        st.markdown("---")
        
        # Display rules
        st.markdown("#### 📋 Association Rules")
        
        # Format for display
        if 'antecedents_str' in rules_df.columns and 'consequents_str' in rules_df.columns:
            display_df = rules_df[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
            display_df.columns = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
            
            # Tạo cột Rule để hiển thị
            display_df['Rule'] = display_df['Antecedent'] + ' → ' + display_df['Consequent']
        else:
            display_df = rules_df[['support', 'confidence', 'lift']].copy()
            display_df['Rule'] = [f"Rule {i+1}" for i in range(len(rules_df))]
        
        # Sort by lift
        display_df = display_df.sort_values('Lift', ascending=False)
        
        st.dataframe(display_df[['Rule', 'Support', 'Confidence', 'Lift']].style.highlight_max(subset=['Lift'], color='lightgreen'), 
                    use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.markdown("#### 📈 Rule Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Lift distribution - dùng bar chart đơn giản
            fig = px.bar(
                display_df,
                x='Rule',
                y='Lift',
                title='Lift by Rule',
                color='Lift',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Support vs Confidence scatter
            fig = px.scatter(
                display_df,
                x='Support',
                y='Confidence',
                size='Lift',
                color='Lift',
                hover_name='Rule',
                title='Support vs Confidence (size = Lift)',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Rules CSV",
            data=csv,
            file_name="association_rules.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No association rules data found.")


def display_clustering(data):
    """Clustering page"""
    
    st.markdown("<h2 class='sub-header'>🔍 Clustering Analysis</h2>", unsafe_allow_html=True)
    
    if 'clustering' in data and not data['clustering'].empty:
        cluster_df = data['clustering'].copy()
        
        # Đảm bảo có cột percentage
        if 'percentage' not in cluster_df.columns:
            total = cluster_df['size'].sum()
            cluster_df['percentage'] = (cluster_df['size'] / total * 100).round(1)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", len(cluster_df))
        with col2:
            total_samples = cluster_df['size'].sum()
            st.metric("Total Samples", f"{total_samples:,}")
        with col3:
            largest_idx = cluster_df['size'].idxmax()
            largest_cluster = cluster_df.loc[largest_idx]
            st.metric("Largest Cluster", f"{largest_cluster['name']} ({largest_cluster['size']:,})")
        
        st.markdown("---")
        
        # Cluster profiles
        st.markdown("#### 📊 Cluster Profiles")
        
        for _, row in cluster_df.iterrows():
            with st.expander(f"Cluster {int(row['cluster'])}: {row['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Size:** {row['size']:,} samples")
                    st.write(f"**Percentage:** {row['percentage']:.1f}%")
                with col2:
                    if 'description' in row and pd.notna(row['description']):
                        st.write(f"**Description:** {row['description']}")
                    if 'top_words' in row and pd.notna(row['top_words']):
                        st.write(f"**Top words:** {row['top_words']}")
        
        # Visualizations
        st.markdown("---")
        st.markdown("#### 📈 Cluster Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(
                cluster_df,
                values='size',
                names='name',
                title='Cluster Size Distribution',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            fig = px.bar(
                cluster_df,
                x='name',
                y='size',
                title='Cluster Sizes',
                color='name',
                text='size',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        csv = cluster_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cluster Profiles CSV",
            data=csv,
            file_name="cluster_profiles.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No clustering data found.")


def display_classification(data):
    """Classification page"""
    
    st.markdown("<h2 class='sub-header'>🤖 Classification Models</h2>", unsafe_allow_html=True)
    
    if 'classification' in data and not data['classification'].empty:
        clf_df = data['classification'].copy()
        
        # Best model
        best_idx = clf_df['F1-Score'].idxmax()
        best = clf_df.loc[best_idx]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best['Model'])
        with col2:
            st.metric("Accuracy", f"{best['Accuracy']:.4f}")
        with col3:
            st.metric("F1-Score", f"{best['F1-Score']:.4f}")
        with col4:
            if 'Train Time (s)' in clf_df.columns:
                st.metric("Training Time", f"{best['Train Time (s)']:.2f}s")
            else:
                st.metric("Training Time", "N/A")
        
        st.markdown("---")
        
        # Model comparison table
        st.markdown("#### 📋 Model Comparison")
        st.dataframe(
            clf_df.style.highlight_max(subset=['F1-Score'], color='lightgreen'),
            use_container_width=True
        )
        
        # Visualizations
        st.markdown("---")
        st.markdown("#### 📈 Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart comparison
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            available_metrics = [m for m in metrics if m in clf_df.columns]
            
            fig = go.Figure()
            for metric in available_metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=clf_df['Model'],
                    y=clf_df[metric],
                    text=clf_df[metric].round(4),
                    textposition='outside'
                ))
            fig.update_layout(
                title='Model Performance Comparison',
                barmode='group',
                yaxis_range=[0, 1],
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training time comparison
            if 'Train Time (s)' in clf_df.columns:
                fig = px.bar(
                    clf_df,
                    x='Model',
                    y='Train Time (s)',
                    title='Training Time Comparison',
                    color='Train Time (s)',
                    color_continuous_scale='viridis',
                    text='Train Time (s)'
                )
                fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        csv = clf_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Model Comparison CSV",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No classification data found.")


def display_semi_supervised(data):
    """Semi-Supervised Learning page"""
    
    st.markdown("<h2 class='sub-header'>🔄 Semi-Supervised Learning</h2>", unsafe_allow_html=True)
    
    if 'semi_self' in data and not data['semi_self'].empty:
        semi_df = data['semi_self'].copy()
        
        # Đảm bảo có các cột cần thiết
        if 'f1' not in semi_df.columns and 'test_f1' in semi_df.columns:
            semi_df['f1'] = semi_df['test_f1']
        
        if 'supervised_f1' not in semi_df.columns and 'supervised_f1' in semi_df.columns:
            semi_df['supervised_f1'] = semi_df['supervised_f1']
        
        if 'labeled_percent' not in semi_df.columns and 'labeled_percent' in semi_df.columns:
            semi_df['labeled_percent'] = semi_df['labeled_percent']
        
        # Learning curve
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=semi_df['labeled_percent'],
            y=semi_df['f1'],
            mode='lines+markers',
            name='Self-Training',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=semi_df['labeled_percent'],
            y=semi_df['supervised_f1'],
            mode='lines+markers',
            name='Supervised Baseline',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Fill improvement area
        fig.add_trace(go.Scatter(
            x=list(semi_df['labeled_percent']) + list(semi_df['labeled_percent'])[::-1],
            y=list(semi_df['supervised_f1']) + list(semi_df['f1'])[::-1],
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Improvement',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Learning Curve: Self-Training vs Supervised',
            xaxis_title='Labeled Data (%)',
            yaxis_title='F1-Score',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Improvement analysis
        st.markdown("---")
        st.markdown("#### 📊 Improvement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Improvement table
            display_semi = semi_df[['labeled_percent', 'f1', 'supervised_f1', 'improvement']].copy()
            display_semi.columns = ['% Labeled', 'Self-Training F1', 'Supervised F1', 'Improvement']
            st.dataframe(
                display_semi.style.highlight_max(subset=['Improvement'], color='lightgreen'),
                use_container_width=True
            )
        
        with col2:
            # Improvement bar chart
            fig = px.bar(
                semi_df,
                x='labeled_percent',
                y='improvement',
                title='Improvement over Supervised Baseline',
                color='improvement',
                color_continuous_scale='RdYlGn',
                text='improvement',
                labels={'labeled_percent': 'Labeled Data (%)', 'improvement': 'Improvement'}
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        
        # Best improvement
        best_idx = semi_df['improvement'].idxmax()
        best = semi_df.loc[best_idx]
        st.markdown(f"""
        <div class='insight-text'>
        <strong>🏆 Best Improvement:</strong> {best['improvement']:.4f} at {best['labeled_percent']}% labeled data
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        csv = semi_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Semi-Supervised Results CSV",
            data=csv,
            file_name="semi_supervised_results.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No semi-supervised data found.")


def display_reports(data):
    """Reports page"""
    
    st.markdown("<h2 class='sub-header'>📋 Reports & Summary</h2>", unsafe_allow_html=True)
    
    # Executive summary
    if 'executive' in data and not data['executive'].empty:
        st.markdown("#### 📊 Executive Summary")
        st.dataframe(data['executive'], use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    st.markdown("#### 🔑 Key Insights")
    
    insights = []
    
    if 'association' in data and not data['association'].empty:
        rules_df = data['association']
        insights.append(f"🔗 **Association Rules**: Found {len(rules_df)} rules with average lift {rules_df['lift'].mean():.2f}")
        ant = rules_df.iloc[0].get('antecedents_str', f"Rule 1")
        cons = rules_df.iloc[0].get('consequents_str', f"Rule 1")
        lift = rules_df.iloc[0].get('lift', 2.36)
        insights.append(f"   • Top rule: '{ant} → {cons}' (lift={lift:.2f})")
    
    if 'clustering' in data and not data['clustering'].empty:
        cluster_df = data['clustering'].copy()
        if 'percentage' not in cluster_df.columns:
            total = cluster_df['size'].sum()
            cluster_df['percentage'] = (cluster_df['size'] / total * 100).round(1)
        
        insights.append(f"🔍 **Clustering**: {len(cluster_df)} clusters identified")
        for _, row in cluster_df.iterrows():
            insights.append(f"   • Cluster {int(row['cluster'])}: {row['name']} - {row['size']:,} reviews ({row['percentage']:.1f}%)")
    
    if 'classification' in data and not data['classification'].empty:
        clf_df = data['classification']
        best_idx = clf_df['F1-Score'].idxmax()
        best = clf_df.loc[best_idx]
        insights.append(f"🤖 **Classification**: Best model is {best['Model']} with F1-Score = {best['F1-Score']:.4f}")
    
    if 'semi_self' in data and not data['semi_self'].empty:
        semi_df = data['semi_self']
        best_idx = semi_df['improvement'].idxmax()
        best_imp = semi_df.loc[best_idx]
        insights.append(f"🔄 **Semi-Supervised**: Best improvement of {best_imp['improvement']:.4f} with {best_imp['labeled_percent']}% labeled data")
    
    # Default insights if none
    if not insights:
        insights = [
            "🔗 **Association Rules**: Found 2 rules with average lift 2.36",
            "   • Top rule: 'book → read' (lift=2.36)",
            "🔍 **Clustering**: 2 clusters identified",
            "   • Cluster 0: Book Reviews - 1,151 reviews (2.3%)",
            "   • Cluster 1: Product Reviews - 48,849 reviews (97.7%)",
            "🤖 **Classification**: Best model is Logistic Regression with F1-Score = 0.8302",
            "🔄 **Semi-Supervised**: Best improvement of 0.0013 with 5% labeled data"
        ]
    
    for insight in insights:
        st.markdown(f"<div class='insight-text'>{insight}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("#### 💡 Recommendations")
    
    recs = [
        "✅ Use **Logistic Regression** for production sentiment classification",
        "✅ Implement **bundle recommendations** based on association rules (book → read)",
        "✅ Apply **cluster-specific marketing** strategies for Book and Product reviews",
        "✅ Use **semi-supervised learning** when labeling new data is expensive",
        "✅ Monitor **negative reviews** in both clusters for early issue detection"
    ]
    
    for rec in recs:
        st.markdown(f"- {rec}")
    
    st.markdown("---")
    
    # Download all reports
    st.markdown("#### 📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'classification' in data and not data['classification'].empty:
            csv = data['classification'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Classification Results",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'clustering' in data and not data['clustering'].empty:
            cluster_download = data['clustering'].copy()
            if 'percentage' not in cluster_download.columns:
                total = cluster_download['size'].sum()
                cluster_download['percentage'] = (cluster_download['size'] / total * 100).round(1)
            csv = cluster_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="🔍 Clustering Profiles",
                data=csv,
                file_name="cluster_profiles.csv",
                mime="text/csv"
            )
    
    with col3:
        if 'association' in data and not data['association'].empty:
            csv = data['association'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="🔗 Association Rules",
                data=csv,
                file_name="association_rules.csv",
                mime="text/csv"
            )


def main():
    """Main function"""
    
    # Load data
    with st.spinner("🔄 Loading data from multiple locations..."):
        data = load_all_data()
    
    # Sidebar
    page = display_sidebar(data)
    
    # Main content
    if page == "🏠 Overview":
        display_overview(data)
    elif page == "🔗 Association Rules":
        display_association(data)
    elif page == "🔍 Clustering":
        display_clustering(data)
    elif page == "🤖 Classification":
        display_classification(data)
    elif page == "🔄 Semi-Supervised":
        display_semi_supervised(data)
    elif page == "📋 Reports":
        display_reports(data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p>© 2026 Sentiment Analysis Project - Data Mining | Group 7</p>
        <p>Dashboard created with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()