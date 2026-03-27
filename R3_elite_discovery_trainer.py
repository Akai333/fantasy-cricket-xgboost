#!/usr/bin/env python3
"""
R3 Elite Discovery Trainer - Updated for 140 Features with Target Variable
Addresses overfitting issues from previous pipeline iterations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
import json
from datetime import datetime
import joblib
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')

class R3EliteDiscoveryTrainer:
    """
    Elite Discovery Trainer for 140-feature CPL dataset with target variable
    Implements anti-overfitting strategies and clean data filtering
    """
    
    def __init__(self):
        """Initialize trainer with anti-overfitting configuration"""
        self.feature_columns = None
        self.target_column = 'target'
        self.model = None
        self.training_results = {}
        
        # Anti-overfitting XGBoost parameters (based on R3 pipeline learnings)
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,                    # REDUCED: Prevent memorization
            'learning_rate': 0.05,             # REDUCED: Slower learning
            'n_estimators': 3000,              # More iterations with slower learning
            'subsample': 0.7,                  # REDUCED: More row sampling
            'colsample_bytree': 0.7,          # REDUCED: More feature sampling
            'colsample_bylevel': 0.8,         # Additional feature sampling per level
            'reg_alpha': 2.0,                  # INCREASED: Strong L1 regularization
            'reg_lambda': 10.0,                # REDUCED from 25: Balanced L2 regularization
            'min_child_weight': 5,             # INCREASED: Prevent overfitting on small groups
            'gamma': 1.0,                      # INCREASED: Minimum split loss
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
        
        # Elite discovery weight scheme (from R3 pipeline)
        self.weight_scheme = {
            'normal': {'threshold': 0.7, 'weight': 1.0},      # Below 0.7
            'good': {'threshold': 0.85, 'weight': 5.0},       # 0.7-0.85
            'very_good': {'threshold': 0.92, 'weight': 15.0}, # 0.85-0.92
            'elite': {'threshold': 0.96, 'weight': 50.0},     # 0.92-0.96
            'super_elite': {'threshold': 1.0, 'weight': 200.0} # 0.96+
        }
        
        print("🎯 R3 Elite Discovery Trainer Initialized")
        print(f"📊 Target: 140 features + target variable")
        print(f"🛡️ Anti-overfitting: Strong regularization + early stopping")
    
    def load_data(self, data_path):
        """Load CPL features with target variable and apply clean data filtering"""
        
        print(f"\n📁 Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        
        print(f"📊 Initial dataset: {len(df):,} teams, {len(df.columns)} columns")
        
        # Verify target variable exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        # Apply clean data filtering (teams with statistical data)
        df_clean = self._filter_clean_data(df)
        
        # Separate features and target
        feature_columns = [col for col in df_clean.columns 
                          if col not in [self.target_column, 'match_id', 'team_uuid', 'date', 'venue']]
        
        X = df_clean[feature_columns]
        y = df_clean[self.target_column]
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        print(f"✅ Clean dataset: {len(X):,} teams, {len(X.columns)} features")
        print(f"🎯 Target range: {y.min():.4f} to {y.max():.4f}")
        print(f"🎯 Target mean: {y.mean():.4f} ± {y.std():.4f}")
        
        return X, y, df_clean
    
    def _filter_clean_data(self, df):
        """Filter teams with valid statistical data (based on R3 pipeline criteria)"""
        
        print(f"\n🧹 Applying clean data filtering...")
        initial_count = len(df)
        
        # Define statistical data columns (from R3 pipeline documentation)
        statistical_columns = [
            'avg_fantasy_points_last5_array_mean',
            'last10_fantasy_scores_array_mean', 
            'avg_balls_faced_last5_array_mean',
            'avg_overs_bowled_last5_array_mean'
        ]
        
        # Check which columns exist in the dataset
        existing_stat_cols = [col for col in statistical_columns if col in df.columns]
        
        if not existing_stat_cols:
            print("⚠️ No statistical columns found, using all data")
            return df
        
        print(f"📋 Filtering on {len(existing_stat_cols)} statistical columns: {existing_stat_cols}")
        
        # Filter teams with non-zero statistical data
        mask = df[existing_stat_cols].gt(0).all(axis=1)
        df_clean = df[mask].copy()
        
        filtered_count = initial_count - len(df_clean)
        filter_percentage = (filtered_count / initial_count) * 100
        
        print(f"🗑️ Filtered out: {filtered_count:,} teams ({filter_percentage:.1f}%)")
        print(f"✅ Clean teams: {len(df_clean):,} teams with valid statistical data")
        
        return df_clean
    
    def create_sample_weights(self, y):
        """Create sample weights for elite discovery (from R3 pipeline)"""
        
        weights = np.ones(len(y))
        weight_counts = {}
        
        for category, config in self.weight_scheme.items():
            if category == 'normal':
                mask = y < config['threshold']
            elif category == 'super_elite':
                mask = y >= self.weight_scheme['elite']['threshold']
            else:
                prev_threshold = 0.0
                for prev_cat, prev_config in self.weight_scheme.items():
                    if prev_cat == category:
                        break
                    prev_threshold = prev_config['threshold']
                
                mask = (y >= prev_threshold) & (y < config['threshold'])
            
            weights[mask] = config['weight']
            weight_counts[category] = {
                'count': mask.sum(),
                'weight': config['weight'],
                'percentage': mask.sum() / len(y) * 100
            }
        
        print(f"\n⚖️ Sample weights distribution:")
        for category, stats in weight_counts.items():
            print(f"   {category:12}: {stats['count']:6,} teams ({stats['percentage']:5.1f}%) - weight {stats['weight']}")
        
        return weights
    
    def train_model(self, X, y, test_size=0.2, early_stopping_rounds=100):
        """Train XGBoost model with anti-overfitting measures"""
        
        print(f"\n🚀 Training XGBoost model...")
        print(f"📊 Features: {X.shape[1]}, Samples: {X.shape[0]:,}")
        
        # Create sample weights for elite discovery
        sample_weights = self.create_sample_weights(y)
        
        # Split data
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=test_size, random_state=42, stratify=None
        )
        
        print(f"📋 Train: {len(X_train):,} teams, Test: {len(X_test):,} teams")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
        dtest = xgb.DMatrix(X_test, label=y_test, weight=weights_test)
        
        # Training with early stopping
        start_time = time.time()
        
        # Use validation set for early stopping
        evallist = [(dtrain, 'train'), (dtest, 'test')]
        
        print(f"🎯 Training with early stopping (patience: {early_stopping_rounds})")
        print(f"🛡️ Regularization: L1={self.xgb_params['reg_alpha']}, L2={self.xgb_params['reg_lambda']}")
        
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.xgb_params['n_estimators'],
            evals=evallist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        training_time = time.time() - start_time
        
        print(f"⏱️ Training completed in {training_time:.1f} seconds")
        print(f"🎯 Best iteration: {self.model.best_iteration}")
        print(f"🎯 Best score: {self.model.best_score:.6f}")
        
        # Evaluate model
        train_pred = self.model.predict(dtrain)
        test_pred = self.model.predict(dtest)
        
        results = self._evaluate_predictions(
            y_train, train_pred, y_test, test_pred, 
            weights_train, weights_test, training_time
        )
        
        self.training_results = results
        
        return results
    
    def _evaluate_predictions(self, y_train, train_pred, y_test, test_pred, 
                            weights_train, weights_test, training_time):
        """Comprehensive model evaluation"""
        
        print(f"\n📊 MODEL EVALUATION RESULTS")
        print("=" * 50)
        
        # Basic metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Spearman correlation
        train_spearman, _ = spearmanr(y_train, train_pred)
        test_spearman, _ = spearmanr(y_test, test_pred)
        
        # Overfitting detection
        rmse_gap = test_rmse - train_rmse
        spearman_gap = train_spearman - test_spearman
        overfitting_ratio = test_rmse / train_rmse
        
        print(f"📈 OVERALL PERFORMANCE:")
        print(f"   Train RMSE: {train_rmse:.6f}")
        print(f"   Test RMSE:  {test_rmse:.6f}")
        print(f"   Train MAE:  {train_mae:.6f}")
        print(f"   Test MAE:   {test_mae:.6f}")
        print(f"   Train Spearman: {train_spearman:.6f}")
        print(f"   Test Spearman:  {test_spearman:.6f}")
        
        print(f"\n🛡️ OVERFITTING ANALYSIS:")
        print(f"   RMSE Gap: {rmse_gap:.6f}")
        print(f"   Spearman Gap: {spearman_gap:.6f}")
        print(f"   Overfitting Ratio: {overfitting_ratio:.3f}")
        
        if overfitting_ratio > 1.5:
            print("   ⚠️ SEVERE OVERFITTING DETECTED")
        elif overfitting_ratio > 1.2:
            print("   ⚠️ Moderate overfitting detected")
        else:
            print("   ✅ Overfitting under control")
        
        # Elite team analysis
        elite_results = self._analyze_elite_performance(y_test, test_pred)
        
        # Feature importance
        feature_importance = self._get_feature_importance()
        
        results = {
            'training_time': training_time,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_spearman': train_spearman,
            'test_spearman': test_spearman,
            'rmse_gap': rmse_gap,
            'spearman_gap': spearman_gap,
            'overfitting_ratio': overfitting_ratio,
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'elite_analysis': elite_results,
            'feature_importance': feature_importance
        }
        
        return results
    
    def _analyze_elite_performance(self, y_true, y_pred):
        """Analyze model performance on elite teams (>=0.92)"""
        
        elite_mask = y_true >= 0.92
        elite_count = elite_mask.sum()
        
        if elite_count == 0:
            print(f"\n🏆 ELITE TEAM ANALYSIS: No elite teams in test set")
            return {'elite_count': 0}
        
        elite_y_true = y_true[elite_mask]
        elite_y_pred = y_pred[elite_mask]
        
        elite_rmse = np.sqrt(mean_squared_error(elite_y_true, elite_y_pred))
        elite_mae = mean_absolute_error(elite_y_true, elite_y_pred)
        elite_spearman, _ = spearmanr(elite_y_true, elite_y_pred)
        
        print(f"\n🏆 ELITE TEAM ANALYSIS (>= 0.92):")
        print(f"   Elite teams in test: {elite_count}")
        print(f"   Elite RMSE: {elite_rmse:.6f}")
        print(f"   Elite MAE: {elite_mae:.6f}")
        print(f"   Elite Spearman: {elite_spearman:.6f}")
        
        # Precision at top predictions
        precision_results = {}
        for k in [10, 20, 50, 100]:
            top_k_indices = np.argsort(y_pred)[-k:]
            top_k_true = y_true.iloc[top_k_indices]
            elite_in_top_k = (top_k_true >= 0.92).sum()
            precision_k = elite_in_top_k / k
            precision_results[f'precision@{k}'] = precision_k
            print(f"   Precision@{k}: {precision_k:.3f} ({elite_in_top_k}/{k} elite teams)")
        
        return {
            'elite_count': elite_count,
            'elite_rmse': elite_rmse,
            'elite_mae': elite_mae,
            'elite_spearman': elite_spearman,
            **precision_results
        }
    
    def _get_feature_importance(self):
        """Get feature importance from trained model"""
        
        if self.model is None:
            return {}
        
        # Get importance scores
        importance = self.model.get_score(importance_type='gain')
        
        # Convert to sorted list
        feature_importance = []
        for feature, score in importance.items():
            feature_importance.append({
                'feature': feature,
                'importance': score,
                'feature_name': feature if feature in self.feature_columns else feature
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"\n🎯 TOP 15 FEATURE IMPORTANCE:")
        for i, feat in enumerate(feature_importance[:15], 1):
            print(f"   {i:2d}. {feat['feature']:35} {feat['importance']:.6f}")
        
        return feature_importance
    
    def save_model(self, output_dir):
        """Save trained model and results"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_file = output_dir / f"r3_elite_model_{timestamp}.joblib"
        joblib.dump(self.model, model_file)
        
        # Save results
        results_file = output_dir / f"r3_training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # Save feature importance
        if 'feature_importance' in self.training_results:
            importance_file = output_dir / f"r3_feature_importance_{timestamp}.csv"
            importance_df = pd.DataFrame(self.training_results['feature_importance'])
            importance_df.to_csv(importance_file, index=False)
        
        print(f"\n💾 MODEL SAVED:")
        print(f"   Model: {model_file}")
        print(f"   Results: {results_file}")
        print(f"   Feature Importance: {importance_file}")
        
        return model_file, results_file, importance_file

def main():
    """Main training pipeline"""
    
    print("🚀 R3 ELITE DISCOVERY TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize trainer
    trainer = R3EliteDiscoveryTrainer()
    
    # Data paths - try cleaned data first, then original
    data_paths = [
        "../R3_Features_output/cpl_features_fast/cpl_features_combined_clean.parquet",
        "../R3_Features_output/cpl_features_fast/cpl_features_combined.parquet"
    ]
    
    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = path
            break
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("🔄 Waiting for feature extraction to complete...")
        return
    
    try:
        # Load data
        X, y, df = trainer.load_data(data_path)
        
        # Train model
        results = trainer.train_model(X, y, test_size=0.2)
        
        # Save model
        output_dir = "../R3_Training_Results/R3_Final_Models"
        trainer.save_model(output_dir)
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Test Spearman: {results['test_spearman']:.6f}")
        print(f"🛡️ Overfitting Ratio: {results['overfitting_ratio']:.3f}")
        
        if 'elite_analysis' in results and results['elite_analysis']['elite_count'] > 0:
            elite_spearman = results['elite_analysis']['elite_spearman']
            print(f"🏆 Elite Team Spearman: {elite_spearman:.6f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
