#!/usr/bin/env python3
"""
Fixed XGBoost High-Recall Training for CEM Elite Team Prediction
Fixes data type issues to ensure all features are proper scalars for XGBoost

Key Fixes:
- Ensure all feature values are scalars (not arrays)
- Robust data type handling
- Comprehensive error handling for array processing
"""

import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_manageable_dataset(max_elite=2000, max_non_elite=20000):
    """Load a manageable subset of data for efficient training"""
    print("📊 Loading manageable dataset for fast training...")
    start_time = time.time()
    
    # Load data files
    elite_files = list(Path("team_output/elite_teams").glob("*.parquet"))
    non_elite_files = list(Path("team_output/non_elite_teams").glob("*.parquet"))
    
    if not elite_files or not non_elite_files:
        raise FileNotFoundError("Parquet files not found in team_output/")
    
    print(f"   Found {len(elite_files)} elite files, {len(non_elite_files)} non-elite files")
    
    # Load all elite teams (they're manageable in size)
    print("   Loading elite teams...")
    elite_dfs = []
    total_elite = 0
    for file in elite_files:
        df = pd.read_parquet(file)
        df['is_elite'] = 1
        elite_dfs.append(df)
        total_elite += len(df)
        print(f"     {file.name}: {len(df)} teams")
        if total_elite >= max_elite:
            break
    
    df_elite = pd.concat(elite_dfs, ignore_index=True)
    if len(df_elite) > max_elite:
        df_elite = df_elite.head(max_elite)
    
    # Load manageable subset of non-elite teams
    print("   Loading non-elite teams...")
    non_elite_dfs = []
    total_non_elite = 0
    
    for file in non_elite_files[:2]:  # Only first 2 files to avoid memory issues
        print(f"     Loading {file.name}...")
        df = pd.read_parquet(file)
        df['is_elite'] = 0
        
        # Take only what we need
        remaining_needed = max_non_elite - total_non_elite
        if remaining_needed <= 0:
            break
            
        df_sample = df.head(min(len(df), remaining_needed))
        non_elite_dfs.append(df_sample)
        total_non_elite += len(df_sample)
        print(f"     Added {len(df_sample)} teams (total: {total_non_elite})")
        
        if total_non_elite >= max_non_elite:
            break
    
    df_non_elite = pd.concat(non_elite_dfs, ignore_index=True)
    
    # Combine datasets
    df_combined = pd.concat([df_elite, df_non_elite], ignore_index=True)
    
    load_time = time.time() - start_time
    print(f"✅ Dataset loaded in {load_time:.1f}s")
    print(f"   Elite teams: {len(df_elite):,}")
    print(f"   Non-elite teams: {len(df_non_elite):,}")
    print(f"   Total teams: {len(df_combined):,}")
    print(f"   Elite rate: {len(df_elite)/len(df_combined):.3%}")
    
    return df_combined

def safe_scalar_from_array(arr):
    """Safely extract a scalar value from an array-like object"""
    try:
        if arr is None:
            return 0.0
        
        # Convert to numpy array
        arr_np = np.asarray(arr, dtype=float)
        
        # Handle empty arrays
        if arr_np.size == 0:
            return 0.0
        
        # Flatten if multi-dimensional
        arr_flat = arr_np.flatten()
        
        # Return first value if array, otherwise the scalar
        if arr_flat.size == 1:
            return float(arr_flat[0])
        else:
            return float(arr_flat[0])  # Take first value if multiple
            
    except (ValueError, TypeError, IndexError):
        return 0.0

def extract_robust_features(df):
    """Extract features with robust scalar conversion"""
    print("🔧 Extracting features with robust scalar conversion...")
    start_time = time.time()
    
    n_teams = len(df)
    features_dict = {}
    
    print("   Processing player statistics arrays...")
    
    # Extract features from player arrays
    array_cols = ['avg_fantasy_points_last5_array', 'avg_balls_faced_last5_array', 
                  'avg_overs_bowled_last5_array', 'last10_fantasy_scores_array']
    
    for col in array_cols:
        if col in df.columns:
            print(f"     Processing {col}...")
            
            # Extract statistics for this column
            col_data = df[col]
            
            # Calculate mean
            means = []
            stds = []
            maxs = []
            mins = []
            medians = []
            
            for arr in col_data:
                try:
                    if arr is None or len(arr) == 0:
                        means.append(0.0)
                        stds.append(0.0)
                        maxs.append(0.0)
                        mins.append(0.0)
                        medians.append(0.0)
                    else:
                        arr_clean = np.array(arr, dtype=float)
                        means.append(float(np.mean(arr_clean)))
                        stds.append(float(np.std(arr_clean)))
                        maxs.append(float(np.max(arr_clean)))
                        mins.append(float(np.min(arr_clean)))
                        medians.append(float(np.median(arr_clean)))
                except:
                    means.append(0.0)
                    stds.append(0.0)
                    maxs.append(0.0)
                    mins.append(0.0)
                    medians.append(0.0)
            
            features_dict[f'{col}_mean'] = np.array(means, dtype=float)
            features_dict[f'{col}_std'] = np.array(stds, dtype=float)
            features_dict[f'{col}_max'] = np.array(maxs, dtype=float)
            features_dict[f'{col}_min'] = np.array(mins, dtype=float)
            features_dict[f'{col}_median'] = np.array(medians, dtype=float)
        else:
            print(f"     Warning: {col} not found, using zeros")
            for stat in ['mean', 'std', 'max', 'min', 'median']:
                features_dict[f'{col}_{stat}'] = np.zeros(n_teams, dtype=float)
    
    # Ownership percentage
    print("   Processing ownership percentages...")
    if 'ownership_percentage' in df.columns:
        own_means = []
        own_stds = []
        own_maxs = []
        own_mins = []
        own_medians = []
        
        for arr in df['ownership_percentage']:
            try:
                if arr is None or len(arr) == 0:
                    own_means.append(0.0)
                    own_stds.append(0.0)
                    own_maxs.append(0.0)
                    own_mins.append(0.0)
                    own_medians.append(0.0)
                else:
                    arr_clean = np.array(arr, dtype=float)
                    own_means.append(float(np.mean(arr_clean)))
                    own_stds.append(float(np.std(arr_clean)))
                    own_maxs.append(float(np.max(arr_clean)))
                    own_mins.append(float(np.min(arr_clean)))
                    own_medians.append(float(np.median(arr_clean)))
            except:
                own_means.append(0.0)
                own_stds.append(0.0)
                own_maxs.append(0.0)
                own_mins.append(0.0)
                own_medians.append(0.0)
        
        features_dict['ownership_mean'] = np.array(own_means, dtype=float)
        features_dict['ownership_std'] = np.array(own_stds, dtype=float)
        features_dict['ownership_max'] = np.array(own_maxs, dtype=float)
        features_dict['ownership_min'] = np.array(own_mins, dtype=float)
        features_dict['ownership_median'] = np.array(own_medians, dtype=float)
    else:
        for stat in ['mean', 'std', 'max', 'min', 'median']:
            features_dict[f'ownership_{stat}'] = np.zeros(n_teams, dtype=float)
    
    # Simple categorical encodings
    print("   Processing categorical features...")
    
    def safe_hash_encode(series, mod_val):
        """Safely encode categorical variables using hash"""
        result = []
        for val in series:
            try:
                hash_val = hash(str(val) if val is not None else 'unknown') % mod_val
                result.append(float(hash_val))
            except:
                result.append(0.0)
        return np.array(result, dtype=float)
    
    features_dict['venue_encoded'] = safe_hash_encode(df.get('venue', ['unknown'] * n_teams), 1000)
    features_dict['league_encoded'] = safe_hash_encode(df.get('league', ['unknown'] * n_teams), 100)
    features_dict['pitch_encoded'] = safe_hash_encode(df.get('pitch_type', ['unknown'] * n_teams), 50)
    features_dict['toss_encoded'] = safe_hash_encode(df.get('toss_decision', ['unknown'] * n_teams), 10)
    features_dict['weather_encoded'] = safe_hash_encode(df.get('weather_conditions', ['unknown'] * n_teams), 20)
    
    # String length features
    capt_lens = []
    vice_lens = []
    for i in range(n_teams):
        try:
            capt = df.iloc[i].get('captain', '')
            vice = df.iloc[i].get('vice_captain', '')
            capt_lens.append(float(len(str(capt)) if capt is not None else 0))
            vice_lens.append(float(len(str(vice)) if vice is not None else 0))
        except:
            capt_lens.append(0.0)
            vice_lens.append(0.0)
    
    features_dict['captain_len'] = np.array(capt_lens, dtype=float)
    features_dict['vice_captain_len'] = np.array(vice_lens, dtype=float)
    
    # Numeric features with defaults
    features_dict['season_year'] = np.array([float(df.iloc[i].get('season_year', 2024)) for i in range(n_teams)], dtype=float)
    features_dict['match_number'] = np.array([float(df.iloc[i].get('match_number', 1)) for i in range(n_teams)], dtype=float)
    features_dict['team_count'] = np.array([float(df.iloc[i].get('team_count', 11)) for i in range(n_teams)], dtype=float)
    
    # Combine all features into DataFrame and ensure all are float64
    features_df = pd.DataFrame(features_dict)
    
    # Ensure all columns are proper float64
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0).astype(np.float64)
    
    extract_time = time.time() - start_time
    print(f"✅ Feature extraction completed in {extract_time:.1f}s")
    print(f"   Feature shape: {features_df.shape}")
    print(f"   All features are numeric: {features_df.dtypes.apply(lambda x: x.kind in 'biufc').all()}")
    
    # Verify no problematic values
    feature_array = features_df.values
    print(f"   Feature array dtype: {feature_array.dtype}")
    print(f"   Contains NaN: {np.isnan(feature_array).any()}")
    print(f"   Contains Inf: {np.isinf(feature_array).any()}")
    
    return feature_array

def recall_at_percentile(y_true, y_pred_proba, percentile=90):
    """Calculate recall at top percentile (primary metric for Stage 1 filtering)"""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
        
    threshold = np.percentile(y_pred_proba, percentile)
    top_predictions = y_pred_proba >= threshold
    
    elite_indices = np.where(y_true == 1)[0]
    elite_in_top = np.sum(top_predictions[elite_indices])
    recall = elite_in_top / len(elite_indices)
    
    return recall

def train_fixed_xgboost():
    """Execute XGBoost training with fixed data types"""
    print("🚀 CEM Elite Prediction - Fixed XGBoost High-Recall Training")
    print("🎯 Target: 85-90% elite recall in top 10%-15% predictions")
    print("🔧 Fixed data type issues for XGBoost compatibility")
    print("=" * 80)
    
    training_start = time.time()
    
    # Phase 1: Data Loading
    print("\n📋 PHASE 1: DATA LOADING")
    df = load_manageable_dataset(max_elite=2000, max_non_elite=20000)
    
    # Phase 2: Robust Feature Extraction  
    print("\n🔧 PHASE 2: ROBUST FEATURE EXTRACTION")
    X = extract_robust_features(df)
    y = df['is_elite'].values.astype(int)
    
    print(f"   Final dataset: {X.shape[0]:,} teams, {X.shape[1]} features")
    print(f"   Elite rate: {np.mean(y):.3%}")
    print(f"   Memory usage: ~{X.nbytes / 1024 / 1024:.1f} MB")
    
    # Phase 3: Train/Validation Split
    print("\n🔄 PHASE 3: TRAIN/VALIDATION SPLIT")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train):,} teams, {np.sum(y_train)} elite ({np.mean(y_train):.3%})")
    print(f"   Val: {len(X_val):,} teams, {np.sum(y_val)} elite ({np.mean(y_val):.3%})")
    
    # Phase 4: High-Recall XGBoost Training
    print("\n🤖 PHASE 4: HIGH-RECALL XGBOOST TRAINING")
    
    # Calculate class weight for high recall
    elite_count = np.sum(y_train)
    non_elite_count = len(y_train) - elite_count
    pos_weight = (non_elite_count / elite_count) * 5  # 5x boost for recall
    
    print(f"   scale_pos_weight: {pos_weight:.1f} (boosting elite team importance)")
    print(f"   Training on {len(X_train):,} teams...")
    
    # Create and train XGBoost model
    model = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    print(f"   Training completed in {train_time:.1f}s")
    
    # Phase 5: Performance Evaluation
    print("\n📊 PHASE 5: PERFORMANCE EVALUATION")
    
    # Get predictions
    eval_start = time.time()
    train_preds = model.predict_proba(X_train)[:, 1]
    val_preds = model.predict_proba(X_val)[:, 1]
    eval_time = time.time() - eval_start
    
    print(f"   Inference completed in {eval_time:.1f}s")
    print(f"   Inference speed: {len(X_val) / eval_time:.0f} teams/second")
    
    # Calculate key metrics
    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val, val_preds)
    
    # Calculate recall at different percentiles (key for Stage 1 filtering)
    val_recall_95 = recall_at_percentile(y_val, val_preds, 95)
    val_recall_90 = recall_at_percentile(y_val, val_preds, 90) 
    val_recall_85 = recall_at_percentile(y_val, val_preds, 85)
    val_recall_80 = recall_at_percentile(y_val, val_preds, 80)
    
    print(f"   Training AUC: {train_auc:.4f}")
    print(f"   Validation AUC: {val_auc:.4f}")
    print(f"   Recall@95%: {val_recall_95:.1%} (top 5%)")
    print(f"   Recall@90%: {val_recall_90:.1%} (top 10%)")
    print(f"   Recall@85%: {val_recall_85:.1%} (top 15%)")  
    print(f"   Recall@80%: {val_recall_80:.1%} (top 20%)")
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = [f"feature_{i:02d}" for i in range(len(feature_importance))]
    
    print("\n🏆 Top 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Phase 6: Save Results
    print("\n💾 PHASE 6: SAVING RESULTS")
    
    total_time = time.time() - training_start
    
    # Training statistics
    training_stats = {
        'timestamp': datetime.now().isoformat(),
        'training_teams': int(len(X_train)),
        'validation_teams': int(len(X_val)),
        'total_features': int(X.shape[1]),
        'elite_rate_train': float(np.mean(y_train)),
        'elite_rate_val': float(np.mean(y_val)),
        'scale_pos_weight': float(pos_weight),
        'training_auc': float(train_auc),
        'validation_auc': float(val_auc),
        'recall_at_95pct': float(val_recall_95),
        'recall_at_90pct': float(val_recall_90),
        'recall_at_85pct': float(val_recall_85),
        'recall_at_80pct': float(val_recall_80),
        'training_time_seconds': float(train_time),
        'inference_speed_teams_per_second': float(len(X_val) / eval_time),
        'total_time_seconds': float(total_time),
        'feature_importance': importance_df.to_dict('records'),
    }
    
    # Save model and stats
    model.save_model("xgboost_fixed_model.json")
    
    with open("training_stats_fixed.json", 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"   Model saved: xgboost_fixed_model.json")
    print(f"   Stats saved: training_stats_fixed.json")
    
    # Summary
    print("\n🎯 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"📊 Final Results:")
    print(f"   Training Teams: {len(X_train):,}")
    print(f"   Validation AUC: {val_auc:.4f}")
    print(f"   Elite Recall@90%: {val_recall_90:.1%} (Target: 85%+)")
    print(f"   Elite Recall@85%: {val_recall_85:.1%}")
    print(f"   Inference Speed: {len(X_val) / eval_time:.0f} teams/second")
    print(f"   Total Training Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    success_criteria = val_recall_90 >= 0.85 or val_recall_85 >= 0.90
    if success_criteria:
        print("✅ SUCCESS: High recall target achieved!")
    else:
        print("⚠️ PARTIAL SUCCESS: Reasonable performance for proof of concept")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Scale to larger dataset with proven approach")
    print(f"   2. Test model on production team filtering (200K teams)")
    print(f"   3. Integrate into CEM Stage 1 filtering pipeline")  
    print(f"   4. Begin DeepSets Stage 2 implementation")
    
    return model, training_stats

if __name__ == "__main__":
    try:
        model, stats = train_fixed_xgboost()
        print("\n🎉 Fixed training pipeline completed successfully!")
        exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 