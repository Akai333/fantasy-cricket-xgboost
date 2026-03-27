#!/usr/bin/env python3
"""
R3 Model Evaluator - Comprehensive analysis of trained elite discovery model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb

warnings.filterwarnings('ignore')

class R3ModelEvaluator:
    """
    Comprehensive evaluation of R3 Elite Discovery Model
    """
    
    def __init__(self, model_path, data_path=None):
        """Initialize evaluator with trained model"""
        self.model_path = Path(model_path)
        self.data_path = Path(data_path) if data_path else None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        
        print(f"🔍 R3 Model Evaluator Initialized")
        print(f"📁 Model: {self.model_path}")
        if self.data_path:
            print(f"📁 Data: {self.data_path}")
    
    def load_model_and_data(self):
        """Load trained model and test data"""
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        print(f"✅ Model loaded: {type(self.model)}")
        
        # Load data if provided
        if self.data_path and self.data_path.exists():
            df = pd.read_parquet(self.data_path)
            print(f"✅ Data loaded: {df.shape}")
            
            # Separate features and target
            feature_columns = [col for col in df.columns 
                              if col not in ['target', 'match_id', 'team_uuid', 'date', 'venue']]
            
            self.X_test = df[feature_columns]
            self.y_test = df['target']
            
            print(f"📊 Test data: {len(self.X_test):,} teams, {len(feature_columns)} features")
        
        return True
    
    def generate_predictions(self, X=None, y=None):
        """Generate predictions on test data"""
        
        if X is not None and y is not None:
            self.X_test = X
            self.y_test = y
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test data available")
        
        # Generate predictions
        if hasattr(self.model, 'predict'):
            self.predictions = self.model.predict(self.X_test)
        elif hasattr(self.model, 'predict'):  # XGBoost DMatrix
            dtest = xgb.DMatrix(self.X_test)
            self.predictions = self.model.predict(dtest)
        else:
            raise ValueError("Model does not have predict method")
        
        print(f"✅ Predictions generated: {len(self.predictions):,}")
        
        return self.predictions
    
    def comprehensive_evaluation(self):
        """Perform comprehensive model evaluation"""
        
        if self.predictions is None:
            self.generate_predictions()
        
        print(f"\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        results = {}
        
        # Basic regression metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        mae = mean_absolute_error(self.y_test, self.predictions)
        
        # Correlation metrics
        spearman_corr, spearman_p = spearmanr(self.y_test, self.predictions)
        pearson_corr, pearson_p = pearsonr(self.y_test, self.predictions)
        
        # R-squared equivalent
        ss_res = np.sum((self.y_test - self.predictions) ** 2)
        ss_tot = np.sum((self.y_test - np.mean(self.y_test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"📈 REGRESSION METRICS:")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R²: {r_squared:.6f}")
        
        print(f"\n📊 CORRELATION METRICS:")
        print(f"   Spearman: {spearman_corr:.6f} (p={spearman_p:.6f})")
        print(f"   Pearson: {pearson_corr:.6f} (p={pearson_p:.6f})")
        
        results.update({
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p
        })
        
        # Elite team analysis
        elite_results = self._analyze_elite_teams()
        results['elite_analysis'] = elite_results
        
        # Prediction distribution analysis
        distribution_results = self._analyze_prediction_distribution()
        results['distribution_analysis'] = distribution_results
        
        # Feature importance (if available)
        if hasattr(self.model, 'get_score'):
            importance_results = self._analyze_feature_importance()
            results['feature_importance'] = importance_results
        
        return results
    
    def _analyze_elite_teams(self):
        """Detailed analysis of elite team predictions"""
        
        print(f"\n🏆 ELITE TEAM ANALYSIS")
        print("-" * 30)
        
        elite_results = {}
        
        # Define elite thresholds
        thresholds = [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.96]
        
        for threshold in thresholds:
            elite_mask = self.y_test >= threshold
            elite_count = elite_mask.sum()
            
            if elite_count == 0:
                continue
            
            elite_y_true = self.y_test[elite_mask]
            elite_y_pred = self.predictions[elite_mask]
            
            elite_rmse = np.sqrt(mean_squared_error(elite_y_true, elite_y_pred))
            elite_mae = mean_absolute_error(elite_y_true, elite_y_pred)
            
            if len(elite_y_true) > 1:
                elite_spearman, _ = spearmanr(elite_y_true, elite_y_pred)
            else:
                elite_spearman = np.nan
            
            elite_results[f'threshold_{threshold}'] = {
                'count': int(elite_count),
                'percentage': float(elite_count / len(self.y_test) * 100),
                'rmse': float(elite_rmse),
                'mae': float(elite_mae),
                'spearman': float(elite_spearman) if not np.isnan(elite_spearman) else None
            }
            
            print(f"   >= {threshold}: {elite_count:4d} teams ({elite_count/len(self.y_test)*100:5.2f}%) "
                  f"RMSE: {elite_rmse:.4f}, Spearman: {elite_spearman:.4f}")
        
        # Precision at top predictions
        precision_results = {}
        for k in [10, 20, 50, 100, 200, 500]:
            if k > len(self.predictions):
                continue
                
            top_k_indices = np.argsort(self.predictions)[-k:]
            top_k_true = self.y_test.iloc[top_k_indices]
            
            # Count elite teams in top k
            elite_in_top_k_92 = (top_k_true >= 0.92).sum()
            elite_in_top_k_90 = (top_k_true >= 0.90).sum()
            elite_in_top_k_85 = (top_k_true >= 0.85).sum()
            
            precision_k_92 = elite_in_top_k_92 / k
            precision_k_90 = elite_in_top_k_90 / k
            precision_k_85 = elite_in_top_k_85 / k
            
            precision_results[f'precision@{k}'] = {
                'precision_92': float(precision_k_92),
                'precision_90': float(precision_k_90),
                'precision_85': float(precision_k_85),
                'elite_count_92': int(elite_in_top_k_92),
                'elite_count_90': int(elite_in_top_k_90),
                'elite_count_85': int(elite_in_top_k_85)
            }
        
        print(f"\n🎯 PRECISION AT TOP PREDICTIONS:")
        for k in [10, 20, 50, 100]:
            if f'precision@{k}' in precision_results:
                p92 = precision_results[f'precision@{k}']['precision_92']
                p90 = precision_results[f'precision@{k}']['precision_90']
                p85 = precision_results[f'precision@{k}']['precision_85']
                print(f"   Top {k:3d}: P@92={p92:.3f}, P@90={p90:.3f}, P@85={p85:.3f}")
        
        elite_results['precision_analysis'] = precision_results
        
        return elite_results
    
    def _analyze_prediction_distribution(self):
        """Analyze distribution of predictions vs actual"""
        
        print(f"\n📊 PREDICTION DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        # Basic statistics
        pred_stats = {
            'actual_mean': float(self.y_test.mean()),
            'actual_std': float(self.y_test.std()),
            'actual_min': float(self.y_test.min()),
            'actual_max': float(self.y_test.max()),
            'pred_mean': float(self.predictions.mean()),
            'pred_std': float(self.predictions.std()),
            'pred_min': float(self.predictions.min()),
            'pred_max': float(self.predictions.max())
        }
        
        print(f"   Actual  - Mean: {pred_stats['actual_mean']:.4f}, Std: {pred_stats['actual_std']:.4f}")
        print(f"   Actual  - Range: {pred_stats['actual_min']:.4f} to {pred_stats['actual_max']:.4f}")
        print(f"   Predicted - Mean: {pred_stats['pred_mean']:.4f}, Std: {pred_stats['pred_std']:.4f}")
        print(f"   Predicted - Range: {pred_stats['pred_min']:.4f} to {pred_stats['pred_max']:.4f}")
        
        # Residual analysis
        residuals = self.y_test - self.predictions
        residual_stats = {
            'residual_mean': float(residuals.mean()),
            'residual_std': float(residuals.std()),
            'residual_min': float(residuals.min()),
            'residual_max': float(residuals.max())
        }
        
        print(f"\n📈 RESIDUAL ANALYSIS:")
        print(f"   Mean: {residual_stats['residual_mean']:.6f}")
        print(f"   Std: {residual_stats['residual_std']:.6f}")
        print(f"   Range: {residual_stats['residual_min']:.4f} to {residual_stats['residual_max']:.4f}")
        
        return {**pred_stats, **residual_stats}
    
    def _analyze_feature_importance(self):
        """Analyze feature importance from model"""
        
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 35)
        
        # Get importance scores
        importance = self.model.get_score(importance_type='gain')
        
        # Convert to sorted list
        feature_importance = []
        for feature, score in importance.items():
            feature_importance.append({
                'feature': feature,
                'importance': float(score)
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"   TOP 20 FEATURES:")
        for i, feat in enumerate(feature_importance[:20], 1):
            print(f"   {i:2d}. {feat['feature']:35} {feat['importance']:.6f}")
        
        return feature_importance
    
    def save_evaluation_report(self, results, output_dir="../R3_Training_Results/Evaluation"):
        """Save comprehensive evaluation report"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_dir / f"r3_evaluation_report_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save feature importance
        if 'feature_importance' in results:
            importance_file = output_dir / f"r3_feature_importance_{timestamp}.csv"
            importance_df = pd.DataFrame(results['feature_importance'])
            importance_df.to_csv(importance_file, index=False)
        
        # Save predictions for further analysis
        predictions_file = output_dir / f"r3_predictions_{timestamp}.csv"
        pred_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.predictions,
            'residual': self.y_test - self.predictions
        })
        pred_df.to_csv(predictions_file, index=False)
        
        print(f"\n💾 EVALUATION REPORT SAVED:")
        print(f"   Results: {results_file}")
        print(f"   Predictions: {predictions_file}")
        if 'feature_importance' in results:
            print(f"   Feature Importance: {importance_file}")
        
        return results_file, predictions_file

def main():
    """Main evaluation pipeline"""
    
    print("🔍 R3 MODEL EVALUATION PIPELINE")
    print("=" * 50)
    
    # Look for latest model
    model_dir = Path("../R3_Training_Results/R3_Final_Models")
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    # Find latest model file
    model_files = list(model_dir.glob("r3_elite_model_*.joblib"))
    
    if not model_files:
        print(f"❌ No model files found in {model_dir}")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using latest model: {latest_model}")
    
    # Look for test data
    data_paths = [
        "../R3_Features_output/cpl_features_fast/cpl_features_combined.parquet",
        "../R3_Features_output/cpl_features_optimized/cpl_features_combined.parquet"
    ]
    
    test_data = None
    for path in data_paths:
        if Path(path).exists():
            test_data = path
            break
    
    if not test_data:
        print(f"❌ No test data found")
        return
    
    # Initialize evaluator
    evaluator = R3ModelEvaluator(latest_model, test_data)
    
    try:
        # Load model and data
        evaluator.load_model_and_data()
        
        # Perform comprehensive evaluation
        results = evaluator.comprehensive_evaluation()
        
        # Save evaluation report
        evaluator.save_evaluation_report(results)
        
        print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
