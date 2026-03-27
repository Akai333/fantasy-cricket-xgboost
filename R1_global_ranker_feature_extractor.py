#!/usr/bin/env python3
"""
R1 GLOBAL RANKER FEATURE EXTRACTOR
Stage 2 Match-Aware Ranking Model Feature Extraction

PURPOSE:
- Extract comprehensive features for Stage 2 XGBoost Ranker
- Leverage existing feature extraction capabilities 
- Optimize for ranking/precision tasks (not binary classification)
- Handle soft-labeled data (0.94-1.0 continuous labels)
- Support match-aware training with squad context

INTEGRATION:
- Uses ComprehensiveFeatureExtractorFixed (179 base features)
- Uses EnhancedCVCFeatureExtractor (61 C/VC features) 
- Adds ranking-specific features (15+ new features)
- Total: 255+ features for precision ranking

CRITICAL: No data leakage - only pre-match features used
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import existing extractors
try:
    from comprehensive_feature_extractor_fixed_v2 import ComprehensiveFeatureExtractorFixed
    from enhanced_cvc_feature_extractor import EnhancedCVCFeatureExtractor
except ImportError:
    # Fallback import from backup location
    sys.path.append('Cem_app Backup files/XGBOOST_Saved Model Backup/LATEST XGBoost backup 23-07-2025/')
    from comprehensive_feature_extractor_fixed_v2 import ComprehensiveFeatureExtractorFixed
    from enhanced_cvc_feature_extractor import EnhancedCVCFeatureExtractor

class R1GlobalRankerFeatureExtractor:
    """
    STAGE 2 RANKING FEATURE EXTRACTOR
    
    Combines best of existing extractors + ranking-specific enhancements:
    - 179 comprehensive features (squad context, choice quality, etc.)
    - 61 enhanced C/VC features (multiplier impact, field bias)
    - 15+ ranking-specific features (relative performance, score normalization)
    
    TOTAL: 255+ features optimized for match-aware ranking precision
    """
    
    def __init__(self):
        """Initialize with all sub-extractors"""
        print("🎯 Initializing R1 Global Ranker Feature Extractor...")
        
        # Core extractors
        self.comprehensive_extractor = ComprehensiveFeatureExtractorFixed()
        self.cvc_extractor = EnhancedCVCFeatureExtractor()
        
        # Ranking-specific settings
        self.target_feature_count = 255
        
        print("✅ Core extractors loaded:")
        print("   - Comprehensive: 179 features")
        print("   - Enhanced C/VC: 61 features") 
        print("   - Ranking-specific: 15+ features")
        print(f"🎯 Target total: {self.target_feature_count}+ features")
        
    def _round_calculated_features(self, features_dict):
        """Round calculated features to eliminate floating-point precision artifacts"""
        # Features that involve division or complex calculations that may have precision errors
        precision_sensitive_features = [
            'team_avg_ownership', 'team_weighted_ownership', 'contrarian_score', 
            'field_position_score', 'venue_complexity_score', 'captain_ownership_cown_ratio',
            'vc_ownership_vcown_ratio', 'cvc_weighted_ownership', 'cvc_multiplier_field_weight',
            'cvc_anti_field_score'
        ]
        
        for feature_name in precision_sensitive_features:
            if feature_name in features_dict:
                features_dict[feature_name] = round(features_dict[feature_name], 8)
        
        return features_dict
        
    def extract_all_features(self, team_data: pd.Series) -> Dict[str, float]:
        """
        Extract all features for ranking model
        
        RANKING FOCUS:
        - Relative performance vs absolute scores
        - Choice quality within match context
        - Squad-normalized metrics
        - Soft-label compatible features
        """
        features = {}
        
        try:
            # === 1. COMPREHENSIVE BASE FEATURES (179) ===
            comprehensive_features = self.comprehensive_extractor.extract_all_features(team_data)
            features.update(comprehensive_features)
            
            # === 2. ENHANCED C/VC FEATURES (61) ===
            cvc_features = self.cvc_extractor.extract_enhanced_cvc_features(team_data)
            features.update(cvc_features)
            
            # === 3. RANKING-SPECIFIC FEATURES (15+) ===
            ranking_features = self._extract_ranking_specific_features(team_data)
            features.update(ranking_features)
            
            # === 4. FEATURE VALIDATION ===
            features = self._validate_and_clean_features(features)
            
            print(f"✅ Extracted {len(features)} features for ranking model")
            return features
            
        except Exception as e:
            print(f"❌ Error in feature extraction: {e}")
            # Return safe defaults
            return {f'feature_{i}': 0.0 for i in range(self.target_feature_count)}
    
    def _extract_ranking_specific_features(self, team_data: pd.Series) -> Dict[str, float]:
        """
        Extract features specifically designed for ranking optimization
        
        FOCUS AREAS:
        1. Relative performance metrics (vs squad/field)
        2. Score normalization features
        3. Ranking signals (choice efficiency, upside potential)
        4. Match context weighting
        """
        features = {}
        
        try:
            # NOTE: We DO NOT use soft_label to compute features (that would be data leakage!)
            # soft_label is only used as the target variable (y), never as input features (X)
            
            # === RELATIVE PERFORMANCE FEATURES ===
            # Team performance vs available squad
            selected_fp = self._safe_array_extract(team_data.get('avg_fantasy_points_last5_array', []))
            bf_squad_fp = self._safe_array_extract(team_data.get('batfirst_squad_avg_fantasy_points_last5', []))
            ch_squad_fp = self._safe_array_extract(team_data.get('chase_squad_avg_fantasy_points_last5', []))
            
            # Squad performance metrics
            all_squad_fp = [x for x in (bf_squad_fp + ch_squad_fp) if x > 0]
            selected_fp_clean = [x for x in selected_fp if x > 0]
            
            if len(all_squad_fp) > 0 and len(selected_fp_clean) > 0:
                squad_max = max(all_squad_fp)
                squad_mean = np.mean(all_squad_fp)
                squad_75th = np.percentile(all_squad_fp, 75)
                selected_mean = np.mean(selected_fp_clean)
                
                features['performance_vs_squad_max'] = self._safe_divide(selected_mean, squad_max, 0.0)
                features['performance_vs_squad_mean'] = self._safe_divide(selected_mean, squad_mean, 1.0)
                features['performance_vs_squad_75th'] = self._safe_divide(selected_mean, squad_75th, 1.0)
                
                # Elite player capture rate
                elite_threshold = np.percentile(all_squad_fp, 80)  # Top 20% of squad
                elite_players_in_squad = sum(1 for x in all_squad_fp if x >= elite_threshold)
                elite_players_selected = sum(1 for x in selected_fp_clean if x >= elite_threshold)
                features['elite_player_capture_rate'] = self._safe_divide(elite_players_selected, elite_players_in_squad, 0.0)
            else:
                features['performance_vs_squad_max'] = 0.5
                features['performance_vs_squad_mean'] = 1.0
                features['performance_vs_squad_75th'] = 1.0
                features['elite_player_capture_rate'] = 0.0
            
            # === CHOICE QUALITY RANKING FEATURES ===
            # Enhanced choice efficiency metrics for ranking
            features['choice_quality_score'] = self._calculate_choice_quality_score(team_data)
            features['upside_potential_score'] = self._calculate_upside_potential(team_data)
            features['consistency_vs_ceiling_balance'] = self._calculate_consistency_balance(team_data)
            
            # === OWNERSHIP AND FIELD POSITION ===
            ownership_array = self._safe_array_extract(team_data.get('ownership_array', []), 0.5, 11)
            cown_array = self._safe_array_extract(team_data.get('cown_array', []), 0.09, 11)
            vcown_array = self._safe_array_extract(team_data.get('vcown_array', []), 0.09, 11)
            
            # Field positioning for ranking
            avg_ownership = np.mean(ownership_array) if ownership_array else 0.5
            weighted_ownership = np.mean([2.0 * cown_array[0], 1.5 * vcown_array[1]] + ownership_array[2:]) if len(ownership_array) >= 11 else 0.5
            
            features['team_avg_ownership'] = avg_ownership
            features['team_weighted_ownership'] = weighted_ownership
            features['contrarian_score'] = max(0.0, 1.0 - weighted_ownership)  # Higher = more contrarian
            features['field_position_score'] = self._safe_divide(1.0, weighted_ownership, 2.0)  # Inverse ownership
            
            # === MATCH CONTEXT NORMALIZATION ===
            league = team_data.get('league', '').lower()
            venue = team_data.get('venue', '').lower()
            
            # League strength indicators (for normalization)
            features['major_league_indicator'] = float(any(major in league for major in ['ipl', 'bbl', 'cpl', 'psl']))
            features['venue_complexity_score'] = len(venue) / 20.0  # Proxy for venue analysis depth
            
        except Exception as e:
            print(f"⚠️ Error in ranking-specific features: {e}")
            # Return safe defaults (removed soft_label derived features to prevent data leakage)
            ranking_feature_names = [
                'performance_vs_squad_max', 'performance_vs_squad_mean', 'performance_vs_squad_75th',
                'elite_player_capture_rate', 'choice_quality_score', 'upside_potential_score',
                'consistency_vs_ceiling_balance', 'team_avg_ownership', 'team_weighted_ownership',
                'contrarian_score', 'field_position_score', 'major_league_indicator', 'venue_complexity_score'
            ]
            features = {name: 0.0 for name in ranking_feature_names}
        
        # Apply precision rounding to eliminate floating-point artifacts
        features = self._round_calculated_features(features)
        
        return features
    

    def _calculate_choice_quality_score(self, team_data: pd.Series) -> float:
        """Calculate comprehensive choice quality for ranking"""
        try:
            # Get existing choice quality features
            choice_efficiency = team_data.get('choice_efficiency_overall', 0.0)
            top_talent_capture = team_data.get('top_talent_capture_rate', 0.0)
            opportunity_cost = team_data.get('opportunity_cost', 1.0)
            
            # Combine into ranking score
            choice_score = (choice_efficiency + top_talent_capture + (1.0 - opportunity_cost)) / 3.0
            return min(1.0, max(0.0, choice_score))
        except:
            return 0.5
    
    def _calculate_upside_potential(self, team_data: pd.Series) -> float:
        """Calculate team upside potential for ranking"""
        try:
            # Use last10 scores to assess ceiling vs floor
            last10_scores = team_data.get('last10_fantasy_scores_array', [])
            if self._safe_array_check(last10_scores):
                # Calculate variance and max potential
                all_scores = []
                for player_scores in last10_scores:
                    if isinstance(player_scores, list) and len(player_scores) > 0:
                        all_scores.extend([x for x in player_scores if not np.isnan(x)])
                
                if len(all_scores) > 0:
                    max_score = np.max(all_scores)
                    mean_score = np.mean(all_scores)
                    upside_ratio = self._safe_divide(max_score, mean_score, 1.0)
                    return min(3.0, upside_ratio)  # Cap at 3x upside
            
            return 1.0  # Default neutral upside
        except:
            return 1.0
    
    def _calculate_consistency_balance(self, team_data: pd.Series) -> float:
        """Calculate consistency vs ceiling balance for ranking"""
        try:
            avg_fp = self._safe_array_extract(team_data.get('avg_fantasy_points_last5_array', []))
            if len(avg_fp) > 0:
                team_std = np.std(avg_fp)
                team_mean = np.mean(avg_fp)
                cv = self._safe_divide(team_std, team_mean, 0.0)  # Coefficient of variation
                
                # Balance score: prefer some variance (upside) but not too much (risk)
                optimal_cv = 0.3  # 30% CV is often optimal
                balance_score = 1.0 - abs(cv - optimal_cv) / optimal_cv
                return max(0.0, min(1.0, balance_score))
            
            return 0.5
        except:
            return 0.5
    
    def _safe_array_extract(self, array_data, default_value=0.0, target_length=11):
        """Safely extract numerical values from arrays"""
        try:
            if array_data is None:
                return [default_value] * target_length
            
            if hasattr(array_data, 'tolist'):
                array_data = array_data.tolist()
            
            clean_data = []
            for item in array_data:
                if item is not None and not (isinstance(item, float) and np.isnan(item)):
                    clean_data.append(float(item))
                else:
                    clean_data.append(default_value)
            
            # Ensure target length
            while len(clean_data) < target_length:
                clean_data.append(default_value)
            
            return clean_data[:target_length]
        except:
            return [default_value] * target_length
    
    def _safe_array_check(self, array_data) -> bool:
        """Check if array has valid data"""
        try:
            if array_data is None:
                return False
            if isinstance(array_data, (list, tuple)):
                return len(array_data) > 0
            if hasattr(array_data, 'size'):
                return array_data.size > 0
            return bool(array_data)
        except:
            return False
    
    def _safe_divide(self, numerator, denominator, default=0.0):
        """Safe division with error handling"""
        try:
            if denominator is None or np.isnan(denominator) or denominator == 0:
                return default
            if numerator is None or np.isnan(numerator):
                return default
            
            result = float(numerator) / float(denominator)
            return result if not np.isnan(result) else default
        except:
            return default
    
    def _validate_and_clean_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean extracted features"""
        cleaned = {}
        
        for name, value in features.items():
            try:
                # Convert to float and handle NaN/inf
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        cleaned[name] = 0.0
                    else:
                        cleaned[name] = float(value)
                else:
                    cleaned[name] = 0.0
            except:
                cleaned[name] = 0.0
        
        return cleaned

def extract_features_for_ranking_dataset(data_dir="R1_team_output", max_samples=None):
    """
    Extract features for entire ranking dataset
    Used for training R1 Global Ranker model
    """
    print("🎯 Extracting features for R1 Global Ranker training...")
    
    extractor = R1GlobalRankerFeatureExtractor()
    all_features = []
    all_labels = []
    all_match_ids = []
    
    try:
        # Load elite teams
        elite_files = [f for f in os.listdir(f"{data_dir}/elite_teams") if f.endswith('.parquet')]
        print(f"📊 Processing {len(elite_files)} elite team files...")
        
        for file in elite_files[:3] if max_samples else elite_files:  # Limit for testing
            df = pd.read_parquet(f"{data_dir}/elite_teams/{file}")
            print(f"   Processing {file}: {len(df)} teams")
            
            for _, team in df.iterrows():
                features = extractor.extract_all_features(team)
                all_features.append(features)
                all_labels.append(team.get('soft_label', 1.0))
                all_match_ids.append(team.get('match_id', ''))
        
        # Load non-elite teams
        nonelite_files = [f for f in os.listdir(f"{data_dir}/non_elite_teams") if f.endswith('.parquet')]
        print(f"📊 Processing {len(nonelite_files)} non-elite team files...")
        
        for file in nonelite_files[:3] if max_samples else nonelite_files:  # Limit for testing
            df = pd.read_parquet(f"{data_dir}/non_elite_teams/{file}")
            print(f"   Processing {file}: {len(df)} teams")
            
            for _, team in df.iterrows():
                features = extractor.extract_all_features(team)
                all_features.append(features)
                all_labels.append(team.get('soft_label', 0.0))
                all_match_ids.append(team.get('match_id', ''))
        
        # Convert to DataFrame (DO NOT add soft_label as feature - that's data leakage!)
        features_df = pd.DataFrame(all_features)
        # Store labels and match_ids separately for training, not as features
        features_df['target_label'] = all_labels  # Rename to make clear this is target, not feature
        features_df['match_id'] = all_match_ids
        
        print(f"✅ Feature extraction complete!")
        print(f"   📊 Total teams: {len(features_df)}")
        print(f"   📊 Features extracted: {len(features_df.columns)-2}")
        print(f"   📊 Unique matches: {features_df['match_id'].nunique()}")
        print(f"   📊 Elite teams: {sum(1 for x in all_labels if x >= 0.94)}")
        
        return features_df
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return None

# Test the extractor
if __name__ == "__main__":
    print("🧪 Testing R1 Global Ranker Feature Extractor...")
    
    # Test on small sample
    features_df = extract_features_for_ranking_dataset(max_samples=True)
    
    if features_df is not None:
        print(f"✅ Test successful! Extracted {len(features_df.columns)-2} features")
        print(f"📊 Sample feature statistics:")
        print(features_df.describe().iloc[:5, :10])  # Show first 5 stats, 10 features
    else:
        print("❌ Test failed!")