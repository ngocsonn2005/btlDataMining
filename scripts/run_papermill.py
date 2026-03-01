#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Papermill Script
Executes all notebooks in sequence using papermill
"""

import papermill as pm
import os
from datetime import datetime


def main():
    """Run all notebooks with papermill"""
    
    print("="*80)
    print("RUNNING NOTEBOOKS WITH PAPERMILL")
    print("="*80)
    
    # Create output directory
    os.makedirs("notebooks/outputs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Notebooks to run in order
    notebooks = [
        "01_eda.ipynb",
        "02_preprocess_feature.ipynb",
        "03_association_rules.ipynb",
        "04_clustering.ipynb",
        "05_classification.ipynb",
        "06_semi_supervised.ipynb",
        "07_evaluation_report.ipynb"
    ]
    
    for notebook in notebooks:
        print(f"\n📓 Running {notebook}...")
        
        input_path = f"notebooks/{notebook}"
        output_path = f"notebooks/outputs/{timestamp}_{notebook}"
        
        try:
            pm.execute_notebook(
                input_path,
                output_path,
                kernel_name="python3",
                log_output=True
            )
            print(f"✅ Completed {notebook}")
        except Exception as e:
            print(f"❌ Error in {notebook}: {e}")
            # Continue with next notebook
            continue
    
    print("\n" + "="*80)
    print("✅ ALL NOTEBOOKS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()