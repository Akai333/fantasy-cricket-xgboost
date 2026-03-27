#!/usr/bin/env python3
"""
R3 Exact Feature Trainer - Train model with exact 137 features from CSV
Uses the same features as the original successful model
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
    """
    
    def __init__(self):
        """Initialize trainer with exact feature list from CSV"""
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
        
        # Same XGBoost parameters as successful model
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 3000,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.8,
            'reg_alpha': 2.0,
            'reg_lambda': 10.0,
            'min_child_weight': 5,
            'gamma': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
        
        # Same weight scheme as successful model
        self.weight_scheme = {
            'normal': {'threshold': 0.7, 'weight': 1.0},
            'good': {'threshold': 0.85, 'weight': 5.0},
            'very_good': {'threshold': 0.92, 'weight': 15.0},
            'elite': {'threshold': 0.96, 'weight': 50.0},
            'super_elite': {'threshold': 1.0, 'weight': 200.0}
        }
        
        print("🎯 R3 Exact Feature Trainer Initialized")
        print(f"📊 Target: {len(self.required_features)} exact features from successful model")
        print(f"🛡️ Same parameters as successful model")
    
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
            print(f"❌ Missing features: {missing_features}")
            raise ValueError(f"Missing {len(missing_features)} required features")
        
        # Extract features and target
        X = clean_teams[self.required_features].copy()
        y = clean_teams[self.target_column].copy()
        
        print(f"🎯 Final dataset: {X.shape[0]:,} teams × {X.shape[1]} features")
        print(f"🎯 Target range: {y.min():.4f} to {y.max():.4f}")
        
        return X, y, clean_teams
    
    def calculate_sample_weights(self, y):
        """Calculate sample weights for elite discovery"""
        weights = np.ones(len(y))
        
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
        
        return weights
    
    def train_model(self, X, y, test_size=0.2):
        """Train XGBoost model with same parameters as successful model"""
        print(f"\n🚀 TRAINING XGBOOST MODEL")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        print(f"📊 Training set: {X_train.shape[0]:,} teams")
        print(f"📊 Test set: {X_test.shape[0]:,} teams")
        
        # Calculate sample weights
        sample_weights = self.calculate_sample_weights(y_train)
        print(f"⚖️ Sample weights calculated")
        
        # Train model
        start_time = time.time()
        
        self.model = xgb.XGBRegressor(**self.xgb_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=100,
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
        test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_spearman = spearmanr(y_train, train_pred)[0]
        test_spearman = spearmanr(y_test, test_pred)[0]
        
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
            'feature_count': X.shape[1]
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
        feature_importance = pd.DataFrame({
            'feature': self.required_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
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
