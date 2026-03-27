#!/usr/bin/env python3
"""
R3 Exact Feature Trainer - Fixed Version
Train model with exact 137 features using same parameters as successful model
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

class R3ExactFeatureTrainer:
    """
    Train XGBoost model using exactly the 137 features from the successful model
    Uses same training approach as R3_elite_discovery_trainer.py
    """
    
    def __init__(self):
        """Initialize trainer with exact parameters from successful model"""
        self.target_column = 'target'
        self.model = None
        self.training_results = {}
        
        # EXACT 137 features from the successful model (from CSV)
        self.required_features = [
            'allrounder_quality_score', 'bf_squad_chosen_vs_available_ar', 'ar_selection_efficiency',
            'choice_efficiency_overall', 'ch_squad_chosen_vs_available_ar', 'ch_squad_chosen_vs_available_bat',
            'total_allrounders_count', 'top_players_missed', 'captain_batting_position', 'bowling_quality_score',
            'batting_order_relative_to_squad_pool', 'team_composition_efficiency', 'avg_overs_bowled_last5_array_max',
            'captain_venue_leverage', 'batsman_form_vs_venue_max', 'best_batsman_vs_venue_ceiling',
            'bowler_form_vs_venue_max', 'top_order_perf_ratio', 'wk_selection_efficiency', 'captain_opp_rank_in_pool',
            'captain_recent_form', 'captain_fp_per_ball', 'total_death_overs', 'avg_fantasy_points_last5_array_max',
            'team_variance_vs_pool', 'batsman_form_vs_venue_mean', 'bf_squad_chosen_vs_available_bat',
            'squad_ar_quality_vs_specialist_quality', 'total_middle_overs', 'team_avg_score',
            'captain_perf_rank_in_pool', 'last10_fantasy_scores_array_max', 'best_bowler_vs_venue_ceiling',
            'risk_adjusted_chemistry', 'captain_avg_fp', 'total_powerplay_overs', 'batting_predictability',
            'bf_squad_chosen_vs_available_wk', 'bowl_selection_efficiency', 'captain_role',
            'avg_balls_faced_last5_array_max', 'team_volatility_vs_bf_squad', 'batting_heavy_team',
            'weak_team_player_count', 'avg_overs_bowled_last5_array_mean', 'opener_quality',
            'last10_fantasy_scores_array_std', 'captain_opportunity_vs_alternatives', 'team_max_vs_bf_squad_avg',
            'team_max_vs_ch_squad_avg', 'bowler_perf_ratio', 'captain_role_opportunity_multiplier',
            'captaincy_quality_score', 'ch_squad_chosen_vs_available_wk', 'top_order_opp_ratio',
            'team_avg_vs_ch_squad_avg', 'bowling_variety_score', 'bat_first_heavy', 'ar_total_opportunity_score',
            'intra_team_synergy', 'squad_bat_strength_differential', 'average_selection_rank',
            'avg_balls_faced_last5_array_std', 'finisher_quality', 'bowling_uniqueness', 'top_heavy_team',
            'bowler_opp_ratio', 'top_order_strength', 'avg_fantasy_points_last5_array_std',
            'lower_order_strength', 'bowler_form_vs_venue_mean', 'team_max_score',
            'opportunity_vs_historical_performance_balance', 'last10_fantasy_scores_array_mean',
            'bf_squad_chosen_vs_available_bowl', 'specialist_heavy', 'vice_captain_batting_position',
            'deep_batting', 'death_coverage', 'role_synergy_vs_competition_score', 'tail_contribution',
            'vc_opp_rank_in_pool', 'avg_overs_bowled_last5_array_std', 'vc_avg_fp', 'bowler_workload_efficiency',
            'team_opportunity_concentration_score', 'all_batsmen_opp_ratio', 'vc_venue_leverage',
            'ar_batting_opportunity_vs_pure_batsmen', 'middle_coverage', 'bf_squad_utilization',
            'ch_squad_utilization', 'ch_squad_chosen_vs_available_bowl', 'top_talent_capture_rate',
            'team_strength_variance', 'batting_consistency', 'vc_fp_per_ball', 'powerplay_coverage',
            'phase_balance', 'role_specialization', 'performance_vs_available', 'strong_team_player_count',
            'middle_order_strength', 'batting_order_opportunity_capture', 'vice_captain_recent_form',
            'ar_vs_specialist_opportunity_trade', 'squad_lh_batsmen_availability_vs_selection',
            'vc_perf_rank_in_pool', 'squad_exploitation_efficiency', 'vc_opportunity_vs_remaining_pool',
            'avg_fantasy_points_last5_array_mean', 'team_avg_vs_bf_squad_avg', 'choice_sophistication_score',
            'our_role_selection_vs_squad_dominance', 'avg_balls_faced_last5_array_mean', 'bowling_heavy_team',
            'all_batsmen_perf_ratio', 'team_avg_vs_combined_squad_strength', 'style_based_advantage',
            'innings_balance_max', 'batting_order_risk', 'bat_selection_efficiency',
            'bowling_phase_opportunity_capture', 'batting_quality_score', 'role_balance',
            'ar_bowling_opportunity_vs_pure_bowlers', 'captain_team', 'chemistry_vs_talent',
            'cvc_boost_ratio', 'leadership_balance', 'leadership_spread', 'vice_captain_team',
            'innings_balance_avg', 'cvc_contribution_max', 'squad_pace_vs_spin_bowling_balance',
            'leadership_team_spread', 'cvc_role_diversity'
        ]
        
        # EXACT XGBoost parameters from successful model (R3_Pipeline_11Aug.md)
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,                    # Reduced complexity
            'learning_rate': 0.05,             # Slower learning  
            'n_estimators': 3000,              # More iterations
            'subsample': 0.7,                  # 70% row sampling
            'colsample_bytree': 0.7,          # 70% feature sampling
            'colsample_bylevel': 0.8,         # Additional feature sampling
            'reg_alpha': 2.0,                  # L1 regularization
            'reg_lambda': 10.0,                # L2 regularization
            'min_child_weight': 5,             # Prevent overfitting
            'gamma': 1.0,                      # Minimum split loss
            'random_state': 42,
            'verbosity': 1
        }
        
        # EXACT weight scheme from successful model
        self.weight_scheme = {
            'normal': {'threshold': 0.7, 'weight': 1.0},      # <0.7
            'good': {'threshold': 0.85, 'weight': 5.0},       # 0.7-0.85
            'very_good': {'threshold': 0.92, 'weight': 15.0}, # 0.85-0.92
            'elite': {'threshold': 0.96, 'weight': 50.0},     # 0.92-0.96
            'super_elite': {'threshold': 1.0, 'weight': 200.0} # 0.96+
        }
        
        print("🎯 R3 Exact Feature Trainer Initialized")
        print(f"📊 Target: {len(self.required_features)} exact features from successful model")
        print(f"🛡️ Same XGBoost parameters as successful model")
        print(f"🎯 Anti-overfitting: max_depth=4, lr=0.05, strong regularization")
    
    def load_data(self, data_path):
        """Load data and filter to exact 137 features"""
        print(f"📂 Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"📊 Loaded {len(df):,} teams")
        
        # Filter to clean teams (those with statistical data)
        clean_teams = df[df['target'] > 0].copy()
        print(f"🧹 Clean teams (with statistical data): {len(clean_teams):,}")
        
        # Filter to exact required features + target
        available_features = [f for f in self.required_features if f in clean_teams.columns]
        missing_features = [f for f in self.required_features if f not in clean_teams.columns]
        
        print(f"✅ Available features: {len(available_features)}/{len(self.required_features)}")
        if missing_features:
            print(f"❌ Missing features: {missing_features[:10]}...")  # Show first 10
            raise ValueError(f"Missing {len(missing_features)} required features")
        
        # Extract features and target
        X = clean_teams[self.required_features].copy()
        y = clean_teams[self.target_column].copy()
        
        print(f"🎯 Final dataset: {X.shape[0]:,} teams × {X.shape[1]} features")
        print(f"🎯 Target range: {y.min():.4f} to {y.max():.4f}")
        print(f"🎯 Elite teams (≥0.92): {len(y[y >= 0.92]):,} ({len(y[y >= 0.92])/len(y)*100:.1f}%)")
        
        return X, y, clean_teams
    
    def calculate_sample_weights(self, y):
        """Calculate sample weights for elite discovery (same as successful model)"""
        weights = np.ones(len(y))
        
        # Apply weight scheme
        weights[y < 0.7] = 1.0                     # Normal teams
        weights[(y >= 0.7) & (y < 0.85)] = 5.0    # Good teams
        weights[(y >= 0.85) & (y < 0.92)] = 15.0  # Very good teams
        weights[(y >= 0.92) & (y < 0.96)] = 50.0  # Elite teams
        weights[y >= 0.96] = 200.0                 # Super elite teams
        
        print(f"⚖️ Weight distribution:")
        print(f"   Normal (<0.7): {len(weights[weights == 1.0]):,} teams")
        print(f"   Good (0.7-0.85): {len(weights[weights == 5.0]):,} teams")
        print(f"   Very good (0.85-0.92): {len(weights[weights == 15.0]):,} teams")
        print(f"   Elite (0.92-0.96): {len(weights[weights == 50.0]):,} teams")
        print(f"   Super elite (0.96+): {len(weights[weights == 200.0]):,} teams")
        
        return weights
    
    def train_model(self, X, y, test_size=0.2, early_stopping_rounds=100):
        """Train XGBoost model using same approach as successful model"""
        print(f"\n🚀 TRAINING XGBOOST MODEL")
        print("=" * 50)
        
        # Split data (same random state as successful model)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape[0]:,} teams")
        print(f"📊 Test set: {X_test.shape[0]:,} teams")
        
        # Calculate sample weights
        weights_train = self.calculate_sample_weights(y_train)
        weights_test = self.calculate_sample_weights(y_test)
        
        # Create DMatrix for XGBoost (same as successful model)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
        dtest = xgb.DMatrix(X_test, label=y_test, weight=weights_test)
        
        # Training with early stopping (same as successful model)
        start_time = time.time()
        
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
        
        # Calculate metrics (same as successful model)
        train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
        test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_spearman = spearmanr(y_train, train_pred)[0]
        test_spearman = spearmanr(y_test, test_pred)[0]
        
        # Elite team analysis (≥0.92)
        elite_mask = y_test >= 0.92
        if np.sum(elite_mask) > 0:
            elite_y = y_test[elite_mask]
            elite_pred = test_pred[elite_mask]
            elite_spearman = spearmanr(elite_y, elite_pred)[0]
            elite_rmse = mean_squared_error(elite_y, elite_pred) ** 0.5
            elite_count = len(elite_y)
        else:
            elite_spearman = 0.0
            elite_rmse = 0.0
            elite_count = 0
        
        # Store results
        self.training_results = {
            'training_time': training_time,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_spearman': train_spearman,
            'test_spearman': test_spearman,
            'rmse_gap': abs(test_rmse - train_rmse),
            'spearman_gap': abs(test_spearman - train_spearman),
            'overfitting_ratio': test_rmse / train_rmse,
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_count': X.shape[1],
            'elite_analysis': {
                'elite_count': elite_count,
                'elite_spearman': elite_spearman,
                'elite_rmse': elite_rmse
            }
        }
        
        print(f"\n📊 TRAINING RESULTS:")
        print(f"⏱️ Training time: {training_time:.1f} seconds")
        print(f"🎯 Features used: {X.shape[1]}")
        print(f"🎯 Best iteration: {self.model.best_iteration}")
        print(f"📈 Train RMSE: {train_rmse:.6f}")
        print(f"📈 Test RMSE: {test_rmse:.6f}")
        print(f"📈 Train Spearman: {train_spearman:.6f}")
        print(f"📈 Test Spearman: {test_spearman:.6f}")
        print(f"🛡️ Overfitting ratio: {test_rmse/train_rmse:.3f}")
        print(f"🏆 Elite teams (≥0.92): {elite_count} teams")
        if elite_count > 0:
            print(f"🏆 Elite Spearman: {elite_spearman:.6f}")
        
        return self.training_results
    
    def save_model(self, output_dir):
        """Save the trained model and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_file = output_dir / f"r3_exact_features_model_{timestamp}.joblib"
        joblib.dump(self.model, model_file)
        
        # Save results
        results_file = output_dir / f"r3_exact_features_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # Save feature importance
        importance_file = output_dir / f"r3_exact_features_importance_{timestamp}.csv"
        if hasattr(self.model, 'get_score'):
            # Get feature importance from XGBoost model
            importance_dict = self.model.get_score(importance_type='gain')
            feature_importance = pd.DataFrame([
                {'feature': feature, 'importance': importance_dict.get(f'f{i}', 0.0)}
                for i, feature in enumerate(self.required_features)
            ]).sort_values('importance', ascending=False)
        else:
            # Fallback if importance not available
            feature_importance = pd.DataFrame({
                'feature': self.required_features,
                'importance': [0.0] * len(self.required_features)
            })
        
        feature_importance.to_csv(importance_file, index=False)
        
        print(f"\n💾 MODEL SAVED:")
        print(f"📁 Model: {model_file}")
        print(f"📁 Results: {results_file}")
        print(f"📁 Importance: {importance_file}")
        
        return model_file, results_file, importance_file

def main():
    """Main training pipeline"""
    print("🚀 R3 EXACT FEATURE TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize trainer
    trainer = R3ExactFeatureTrainer()
    
    # Data path - use cleaned data
    data_path = "../R3_Features_output/cpl_features_fast/cpl_features_combined_clean.parquet"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
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
        print(f"🎯 Exact features: {results['feature_count']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



