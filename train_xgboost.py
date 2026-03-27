#!/usr/bin/env python3
"""
XGBoost Training for Elite Team Classification
Memory-efficient batch processing with comprehensive evaluation
"""

import os
import json
import pickle
import time
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, precision_score, recall_score
)
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import our feature engineering module
from features import (
    FeatureEncoder, combine_all_features, prepare_xgboost_features,
    XGBOOST_FEATURE_NAMES
)

class XGBoostTrainer:
    """Memory-efficient XGBoost trainer for elite team classification"""
    
    def __init__(self, batch_size=10000, max_memory_gb=32):
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.feature_encoder = FeatureEncoder()
        self.model = None
        self.results = {}
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def load_parquet_files_batch(self, file_pattern, batch_size=None):
        """
        Load parquet files in memory-efficient batches
        
        Args:
            file_pattern: Pattern to match parquet files
            batch_size: Number of records per batch
            
        Yields:
            DataFrame batches
        """
        batch_size = batch_size or self.batch_size
        
        # Find all parquet files
        elite_files = glob.glob(f"team_output/elite_teams/{file_pattern}")
        nonelite_files = glob.glob(f"team_output/non_elite_teams/{file_pattern}")
        
        print(f"🗂️ Found {len(elite_files)} elite files and {len(nonelite_files)} non-elite files")
        
        all_files = [(f, 'elite') for f in elite_files] + [(f, 'nonelite') for f in nonelite_files]
        
        batch_data = []
        total_processed = 0
        
        for file_path, file_type in all_files:
            try:
                print(f"📖 Loading {file_path}...")
                df = pd.read_parquet(file_path)
                
                # Convert JSON strings to Python objects
                json_columns = [
                    'last10_fantasy_scores_array', 'player_ids', 'roles', 'team_ids',
                    'batting_order_array', 'batting_style_array', 'bowling_style_array',
                    'bowling_phases_array', 'ownership_array', 'avg_fantasy_points_last5_array',
                    'avg_balls_faced_last5_array', 'avg_overs_bowled_last5_array'
                ]
                
                for col in json_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                
                batch_data.append(df)
                total_processed += len(df)
                
                # Yield batch when size limit reached
                if total_processed >= batch_size:
                    combined_batch = pd.concat(batch_data, ignore_index=True)
                    yield combined_batch
                    batch_data = []
                    total_processed = 0
                    
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue
        
        # Yield remaining data
        if batch_data:
            combined_batch = pd.concat(batch_data, ignore_index=True)
            yield combined_batch

    def extract_features_parallel(self, team_records, n_workers=4):
        """
        Extract features from team records using parallel processing
        
        Args:
            team_records: List of team dictionaries
            n_workers: Number of parallel workers
            
        Returns:
            Tuple of (feature_matrix, labels, metadata)
        """
        def process_team_batch(team_batch):
            """Process a batch of teams in one worker"""
            batch_features = []
            batch_labels = []
            batch_metadata = []
            
            for team_data in team_batch:
                try:
                    # Extract all features
                    all_features = combine_all_features(team_data, self.feature_encoder)
                    
                    # Prepare XGBoost features
                    xgb_features = prepare_xgboost_features(all_features)
                    
                    batch_features.append(xgb_features)
                    batch_labels.append(all_features.get('is_elite', 0))
                    batch_metadata.append({
                        'team_uuid': all_features.get('team_uuid', ''),
                        'match_id': all_features.get('match_id', '')
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing team: {e}")
                    continue
            
            return batch_features, batch_labels, batch_metadata
        
        # Split teams into batches for parallel processing
        batch_size = max(1, len(team_records) // n_workers)
        team_batches = [team_records[i:i + batch_size] for i in range(0, len(team_records), batch_size)]
        
        all_features = []
        all_labels = []
        all_metadata = []
        
        print(f"🔄 Processing {len(team_records)} teams using {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all batches
            futures = [executor.submit(process_team_batch, batch) for batch in team_batches]
            
            # Collect results
            for i, future in enumerate(as_completed(futures)):
                try:
                    batch_features, batch_labels, batch_metadata = future.result()
                    all_features.extend(batch_features)
                    all_labels.extend(batch_labels)
                    all_metadata.extend(batch_metadata)
                    print(f"✅ Completed batch {i+1}/{len(futures)}")
                except Exception as e:
                    print(f"⚠️ Error in batch processing: {e}")
        
        # Convert to numpy arrays
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        print(f"📊 Extracted features: {X.shape}, Labels: {y.shape}")
        print(f"📊 Elite teams: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, all_metadata

    def train_model(self, X, y, validation_split=0.2):
        """
        Train XGBoost model with proper class balancing
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction for validation
        """
        print("🚀 Training XGBoost model...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Validation set: {X_val.shape[0]} samples")
        print(f"📊 Training elite rate: {np.mean(y_train)*100:.2f}%")
        print(f"📊 Validation elite rate: {np.mean(y_val)*100:.2f}%")
        
        # Calculate class weights for imbalanced data
        pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        print(f"⚖️ Positive class weight: {pos_weight:.2f}")
        
        # XGBoost parameters optimized for imbalanced classification
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=XGBOOST_FEATURE_NAMES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=XGBOOST_FEATURE_NAMES)
        
        # Train model with early stopping
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        start_time = time.time()
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        self.evaluate_model(X_val, y_val, "Validation")
        
        return X_val, y_val

    def evaluate_model(self, X, y, dataset_name="Test"):
        """
        Comprehensive model evaluation
        
        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name for logging
        """
        print(f"\n📈 Evaluating on {dataset_name} set...")
        
        # Get predictions
        dtest = xgb.DMatrix(X, feature_names=XGBOOST_FEATURE_NAMES)
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        avg_precision = average_precision_score(y, y_pred_proba)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        
        # Precision at different thresholds
        precisions, recalls, thresholds = precision_recall_curve(y, y_pred_proba)
        
        # Find precision at 1% (very high precision)
        precision_at_1_idx = np.where(precisions >= 0.99)[0]
        if len(precision_at_1_idx) > 0:
            precision_at_1_recall = recalls[precision_at_1_idx[0]]
        else:
            precision_at_1_recall = 0
        
        # Store results
        results = {
            'dataset': dataset_name,
            'auc_score': float(auc_score),
            'average_precision': float(avg_precision),
            'precision': float(precision),
            'recall': float(recall),
            'precision_at_99_recall': float(precision_at_1_recall),
            'num_samples': len(y),
            'num_elite': int(np.sum(y)),
            'elite_rate': float(np.mean(y)),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        self.results[dataset_name.lower()] = results
        
        # Print summary
        print(f"🎯 AUC Score: {auc_score:.4f}")
        print(f"🎯 Average Precision: {avg_precision:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"🎯 Recall: {recall:.4f}")
        print(f"🎯 Precision@99%: {precision_at_1_recall:.4f}")
        print(f"📊 Elite teams found: {np.sum(y_pred)}/{np.sum(y)}")
        
        return results

    def save_model_and_results(self):
        """Save trained model and evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save XGBoost model
        model_path = f"models/xgboost_elite_classifier_{timestamp}.json"
        self.model.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Save feature encoder
        encoder_path = f"models/feature_encoder_{timestamp}.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.feature_encoder, f)
        print(f"💾 Feature encoder saved to {encoder_path}")
        
        # Save feature importance
        importance_scores = self.model.get_score(importance_type='weight')
        feature_importance = pd.DataFrame([
            {'feature': feature, 'importance': importance_scores.get(feature, 0)}
            for feature in XGBOOST_FEATURE_NAMES
        ]).sort_values('importance', ascending=False)
        
        importance_path = f"results/feature_importance_{timestamp}.csv"
        feature_importance.to_csv(importance_path, index=False)
        print(f"📊 Feature importance saved to {importance_path}")
        
        # Save evaluation results
        results_path = f"results/xgboost_evaluation_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📈 Evaluation results saved to {results_path}")
        
        # Save model metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': 'XGBoost',
            'feature_count': len(XGBOOST_FEATURE_NAMES),
            'feature_names': XGBOOST_FEATURE_NAMES,
            'training_params': {
                'batch_size': self.batch_size,
                'max_memory_gb': self.max_memory_gb
            }
        }
        
        metadata_path = f"models/model_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Model metadata saved to {metadata_path}")

def main():
    """Main training pipeline"""
    print("🚀 XGBoost Elite Team Classification Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = XGBoostTrainer(batch_size=50000, max_memory_gb=32)  # Adjusted for 64GB RAM
    
    # Check for parquet files
    if not os.path.exists('team_output'):
        print("❌ Error: team_output directory not found!")
        print("Please run the parquet team generator first.")
        return
    
    all_features = []
    all_labels = []
    all_metadata = []
    
    # Process data in batches
    total_teams = 0
    batch_count = 0
    
    print("\n📊 Loading and processing team data...")
    for batch_df in trainer.load_parquet_files_batch("*.parquet"):
        batch_count += 1
        batch_size = len(batch_df)
        total_teams += batch_size
        
        print(f"\n📦 Processing batch {batch_count} ({batch_size} teams)...")
        
        # Initialize encoder on first batch
        if not trainer.feature_encoder.initialized:
            trainer.feature_encoder.initialize_encoders(batch_df)
        
        # Convert DataFrame to list of dictionaries
        team_records = batch_df.to_dict('records')
        
        # Extract features
        X_batch, y_batch, metadata_batch = trainer.extract_features_parallel(
            team_records, n_workers=4
        )
        
        # Accumulate data
        all_features.append(X_batch)
        all_labels.append(y_batch)
        all_metadata.extend(metadata_batch)
        
        print(f"✅ Batch {batch_count} completed: {len(X_batch)} teams processed")
        print(f"📊 Running total: {total_teams} teams, {np.sum(y_batch)} elite in this batch")
        
        # Memory management - limit batches
        if total_teams > 500000:  # Limit for memory management
            print("🛑 Reached team limit for memory management")
            break
    
    if not all_features:
        print("❌ No data was processed successfully!")
        return
    
    # Combine all batches
    print(f"\n🔗 Combining {len(all_features)} batches...")
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"📊 Final dataset: {X.shape[0]} teams, {len(XGBOOST_FEATURE_NAMES)} features")
    print(f"📊 Elite teams: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    
    # Train model
    X_val, y_val = trainer.train_model(X, y)
    
    # Save everything
    trainer.save_model_and_results()
    
    print("\n✅ XGBoost training completed successfully!")
    print(f"📊 Total teams processed: {len(X)}")
    print(f"🎯 Best validation AUC: {trainer.results.get('validation', {}).get('auc_score', 0):.4f}")

if __name__ == "__main__":
    main() 