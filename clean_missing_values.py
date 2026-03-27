#!/usr/bin/env python3
"""
Quick fix for missing values in the training dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_missing_values():
    """Clean missing values in the feature dataset"""
    
    print("🧹 CLEANING MISSING VALUES")
    print("=" * 30)
    
    # Load data
    data_path = "../R3_Features_output/cpl_features_fast/cpl_features_combined.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"📊 Original dataset: {df.shape}")
    
    # Check missing values
    missing_counts = df.isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    
    if len(features_with_missing) == 0:
        print("✅ No missing values found")
        return
    
    print(f"⚠️ Features with missing values:")
    for feature, count in features_with_missing.items():
        percentage = count / len(df) * 100
        print(f"   {feature}: {count:,} ({percentage:.1f}%)")
    
    # Fix bowler_workload_efficiency specifically
    if 'bowler_workload_efficiency' in features_with_missing:
        
        # Strategy: Fill with median value for teams with similar total_death_overs
        print(f"\n🔧 Fixing bowler_workload_efficiency...")
        
        # Calculate median grouped by death overs (similar bowling workload context)
        if 'total_death_overs' in df.columns:
            # Group by death overs and fill with group median
            df['bowler_workload_efficiency'] = df.groupby('total_death_overs')['bowler_workload_efficiency'].transform(
                lambda x: x.fillna(x.median())
            )
            
            # If still missing, fill with overall median
            overall_median = df['bowler_workload_efficiency'].median()
            df['bowler_workload_efficiency'].fillna(overall_median, inplace=True)
            
        else:
            # Fallback: fill with overall median
            median_value = df['bowler_workload_efficiency'].median()
            df['bowler_workload_efficiency'].fillna(median_value, inplace=True)
            print(f"   Filled with median: {median_value:.4f}")
    
    # Handle any other missing values
    feature_columns = [col for col in df.columns 
                      if col not in ['target', 'match_id', 'team_uuid', 'date', 'venue']]
    
    for col in feature_columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                # Numerical: fill with median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   {col}: filled with median {median_val:.4f}")
            else:
                # Categorical: fill with mode
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0
                df[col].fillna(mode_val, inplace=True)
                print(f"   {col}: filled with mode {mode_val}")
    
    # Verify no missing values remain
    final_missing = df.isnull().sum().sum()
    
    if final_missing == 0:
        print(f"✅ All missing values fixed!")
    else:
        print(f"⚠️ {final_missing} missing values remain")
    
    # Save cleaned data
    output_path = "../R3_Features_output/cpl_features_fast/cpl_features_combined_clean.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"💾 Cleaned data saved to: {output_path}")
    print(f"📊 Final dataset: {df.shape}")
    
    return output_path

if __name__ == "__main__":
    clean_missing_values()
