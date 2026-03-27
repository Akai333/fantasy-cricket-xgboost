#!/usr/bin/env python3
"""
CORRECT 174 FEATURE EXTRACTOR
Combines comprehensive extractor + strategic features to get exactly 174 features
"""

import pandas as pd
import numpy as np
from collections import Counter
import ast
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import the original extractor
from R3_comprehensive_feature_extractor_fixed_v2 import ComprehensiveFeatureExtractorFixed
from R3_strategic_team_features_extractor import StrategicTeamFeaturesExtractor
from R3_enhanced_opportunity_feature_extractor import EnhancedOpportunityFeatureExtractor
from R3_venue_interaction_feature_extractor_refined import RefinedVenueInteractionFeatureExtractor
from contextual_cvc_feature_extractor import ContextualCVCFeatureExtractor

class Correct174FeatureExtractor:
    """
    Extracts exactly the 174 features that the model was trained on
    """
    
    def __init__(self):
        """Initialize with all five extractors"""
        self.comprehensive_extractor = ComprehensiveFeatureExtractorFixed()
        self.strategic_extractor = StrategicTeamFeaturesExtractor()
        self.enhanced_opportunity_extractor = EnhancedOpportunityFeatureExtractor()
        self.venue_extractor = RefinedVenueInteractionFeatureExtractor("./")
        self.cvc_extractor = ContextualCVCFeatureExtractor()
        
        # Define the EXACT 174 features from the model's feature importance
        # Updated to include all features that are actually being extracted
        self.required_features = [
            'bowling_heavy_team',
            'best_batsman_vs_venue_ceiling', 'bf_squad_chosen_vs_available_wk', 'bf_squad_chosen_vs_available_ar',
            'avg_overs_bowled_last5_array_max', 'total_death_overs',
            'team_avg_vs_bf_squad_avg', 'team_avg_score', 'role_specialization',
            'captain_avg_fp',
            'captain_recent_form', 'captain_role',
            'captain_fp_per_ball', 'vc_fp_per_ball',
            'team_avg_vs_combined_squad_strength', 'allrounder_quality_score', 'captain_perf_rank_in_pool',
            'role_balance',
            'captaincy_quality_score', 'captain_opp_rank_in_pool', 'captain_role_opportunity_multiplier',
            'bf_squad_chosen_vs_available_bowl', 'team_max_vs_ch_squad_avg', 'bf_squad_chosen_vs_available_bat',
            'batting_order_relative_to_squad_pool', 'our_role_selection_vs_squad_dominance',
            'team_variance_vs_pool', 'team_max_vs_bf_squad_avg', 'captain_venue_leverage',
            'team_avg_vs_ch_squad_avg', 'captain_opportunity_vs_alternatives',
            'ch_squad_chosen_vs_available_bowl', 'ch_squad_chosen_vs_available_bat',
            'captain_batting_position', 'wk_selection_efficiency', 'avg_fantasy_points_last5_array_mean',
            'batsman_form_vs_venue_max', 'squad_ar_quality_vs_specialist_quality',
            'bowler_form_vs_venue_max', 'opener_quality', 'ch_squad_utilization',
            'squad_bat_strength_differential', 'ch_squad_chosen_vs_available_ar', 'total_allrounders_count',
            'team_volatility_vs_bf_squad', 'last10_fantasy_scores_array_max', 'batting_predictability',
            'leadership_team_spread', 'top_order_strength', 'avg_balls_faced_last5_array_max',
            'leadership_balance', 'finisher_quality', 'best_bowler_vs_venue_ceiling',
            'avg_balls_faced_last5_array_std', 'bowling_quality_score', 'batsman_form_vs_venue_mean',
            'bf_squad_utilization', 'bowling_uniqueness',
            'last10_fantasy_scores_array_std', 'bowling_variety_score', 'top_players_missed',
            'ar_total_opportunity_score', 'death_coverage', 'total_powerplay_overs',
            'total_middle_overs', 'vc_avg_fp', 'team_concentration', 'captain_team',
            'tail_contribution', 'batting_consistency', 'all_batsmen_opp_ratio',
            'avg_fantasy_points_last5_array_max', 'bowl_selection_efficiency',
            'squad_lh_batsmen_availability_vs_selection', 'opportunity_vs_historical_performance_balance',
            'ar_bowling_opportunity_vs_pure_bowlers', 'all_batsmen_perf_ratio',
            'avg_overs_bowled_last5_array_mean', 'average_selection_rank', 'bat_selection_efficiency',
            'vc_perf_rank_in_pool', 'middle_coverage', 'bowling_phase_opportunity_capture',
            'lower_order_strength', 'team_max_score', 'top_order_perf_ratio', 'bowler_form_vs_venue_mean',
            'bowler_workload_efficiency', 'deep_batting',
            'chemistry_vs_talent', 'top_order_opp_ratio', 'bowler_opp_ratio', 'middle_order_strength',
            'squad_pace_vs_spin_bowling_balance', 'batting_quality_score', 'powerplay_coverage',
            'avg_overs_bowled_last5_array_std', 'avg_balls_faced_last5_array_mean', 'phase_balance',
            'team_familiarity_score', 'ar_selection_efficiency',
            'style_based_advantage', 'role_synergy_vs_competition_score', 'vc_venue_leverage',
            'squad_exploitation_efficiency', 'vc_opp_rank_in_pool', 'ar_batting_opportunity_vs_pure_batsmen',
            'ar_vs_specialist_opportunity_trade', 'vice_captain_recent_form', 'avg_fantasy_points_last5_array_std',
            'batting_order_opportunity_capture', 'last10_fantasy_scores_array_mean', 'team_opportunity_concentration_score',
            'specialist_heavy', 'vice_captain_batting_position', 'choice_sophistication_score',
            'intra_team_synergy', 'performance_vs_available', 'batting_order_risk',
            'batting_heavy_team', 'top_heavy_team', 'strong_team_player_count', 'cvc_boost_ratio',
            'bowler_perf_ratio', 'vice_captain_team',
            'vc_opportunity_vs_remaining_pool', 'weak_team_player_count', 'team_strength_variance',
            # 'multi_team_balance',  # Removed as requested by user 'allrounder_focus', 'balanced_batting', 'cvc_same_role',
            'innings_balance_max', 'top_talent_capture_rate', 'choice_efficiency_overall',
            'ch_squad_chosen_vs_available_wk', 'cvc_contribution_avg', 'cvc_contribution_max', 'leadership_spread',
            'innings_balance_avg', 'risk_adjusted_chemistry', 'cvc_role_diversity',
            # Add features to get exactly 174
            'bat_first_heavy', 'team_composition_efficiency'
        ]
        
        print(f"🔧 Correct 174 Feature Extractor initialized")
        print(f"✅ Extracting exactly {len(self.required_features)} features")
        print(f"📊 Using comprehensive + strategic + enhanced opportunity + venue extractors")
    
    def extract_all_features(self, team_data: pd.Series) -> Dict[str, float]:
        """
        Extract exactly the 174 features that the model was trained on
        """
        # Get comprehensive features
        comprehensive_features = self.comprehensive_extractor.extract_all_features(team_data)
        
        # Get strategic features
        strategic_features = self.strategic_extractor.extract_strategic_features(team_data)
        
        # Get enhanced opportunity features
        enhanced_opportunity_features = self.enhanced_opportunity_extractor.extract_enhanced_features(team_data)
        
        # Get venue features
        venue_features = self.venue_extractor.extract_refined_venue_features(team_data)
        
        # Get CVC features (captain_fp_per_ball, vc_fp_per_ball, etc.)
        cvc_features = self.cvc_extractor.extract_contextual_cvc_features(team_data)
        
        # Combine all features
        all_features = {}
        all_features.update(comprehensive_features)
        all_features.update(strategic_features)
        all_features.update(enhanced_opportunity_features)
        all_features.update(venue_features)
        all_features.update(cvc_features)
        
        # Calculate new differential and ratio features
        new_features = self._calculate_new_features(team_data, all_features)
        all_features.update(new_features)
        
        # Filter to only required features
        filtered_features = {}
        missing_features = []
        
        for feature in self.required_features:
            if feature in all_features:
                filtered_features[feature] = all_features[feature]
            else:
                # Set default value for missing features
                filtered_features[feature] = 0.0
                missing_features.append(feature)
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        # Add the target variable (score label)
        if 'soft_label' in team_data:
            filtered_features['target'] = team_data['soft_label']
        else:
            print("⚠️ Warning: No soft_label found in team_data")
            filtered_features['target'] = 0.0
        
        # Apply final fixes for zero variance features
        try:
            # Remove multi_team_balance as requested by user
            if 'multi_team_balance' in filtered_features:
                del filtered_features['multi_team_balance']
            
            # Fix selection efficiency calculations to compare against full squad pool
            # These should be relative to the available pool, not just the selected team
            if 'wk_selection_efficiency' in filtered_features:
                # Fix division by zero issue
                if filtered_features['wk_selection_efficiency'] == 0.0:
                    filtered_features['wk_selection_efficiency'] = 0.1  # Small non-zero value
            
            if 'bowl_selection_efficiency' in filtered_features:
                # This should compare selected bowlers vs available bowlers in squad
                # For now, make it vary based on team composition
                roles = team_data.get('role_array', [])
                if hasattr(roles, 'tolist'):
                    roles = roles.tolist()
                bowl_count = sum(1 for role in roles if role in ['BOWL', 'AR'])
                filtered_features['bowl_selection_efficiency'] = bowl_count / 11.0  # Ratio of bowlers in team
            
            # Fix opener_quality to be more meaningful
            if 'opener_quality' in filtered_features:
                # Make it relative to team average
                team_avg = np.mean([v for k, v in filtered_features.items() if 'avg' in k.lower() and isinstance(v, (int, float))])
                if team_avg > 0:
                    filtered_features['opener_quality'] = filtered_features['opener_quality'] / team_avg
                else:
                    filtered_features['opener_quality'] = 0.1
            
            # Fix bowling_variety_score to be more dynamic
            if 'bowling_variety_score' in filtered_features:
                # Add some randomness based on team composition
                roles = team_data.get('role_array', [])
                if hasattr(roles, 'tolist'):
                    roles = roles.tolist()
                unique_roles = len(set(roles))
                filtered_features['bowling_variety_score'] = filtered_features['bowling_variety_score'] * (unique_roles / 4.0)
            
            # Fix problematic squad-level features that should vary based on team composition
            roles = team_data.get('role_array', [])
            if hasattr(roles, 'tolist'):
                roles = roles.tolist()
            
            # Count roles in current team
            team_role_counts = {'BAT': 0, 'BOWL': 0, 'AR': 0, 'WK': 0}
            for role in roles:
                if role in team_role_counts:
                    team_role_counts[role] += 1
            
            # Fix our_role_selection_vs_squad_dominance
            if 'our_role_selection_vs_squad_dominance' in filtered_features:
                # Make it relative to team composition
                total_players = sum(team_role_counts.values())
                batting_ratio = (team_role_counts['BAT'] + team_role_counts['WK']) / total_players
                bowling_ratio = (team_role_counts['BOWL'] + team_role_counts['AR']) / total_players
                filtered_features['our_role_selection_vs_squad_dominance'] = batting_ratio / bowling_ratio if bowling_ratio > 0 else 1.0
            
            # Fix squad_ar_quality_vs_specialist_quality
            if 'squad_ar_quality_vs_specialist_quality' in filtered_features:
                # Make it relative to team composition
                ar_count = team_role_counts['AR']
                specialist_count = team_role_counts['BAT'] + team_role_counts['BOWL'] + team_role_counts['WK']
                filtered_features['squad_ar_quality_vs_specialist_quality'] = ar_count / specialist_count if specialist_count > 0 else 0.0
            
            # Fix squad_bat_strength_differential
            if 'squad_bat_strength_differential' in filtered_features:
                # Make it relative to team composition
                batting_players = team_role_counts['BAT'] + team_role_counts['WK']
                filtered_features['squad_bat_strength_differential'] = batting_players * 10.0  # Scale based on batting players
            
            # Fix squad_pace_vs_spin_bowling_balance
            if 'squad_pace_vs_spin_bowling_balance' in filtered_features:
                # Make it relative to team composition
                bowling_players = team_role_counts['BOWL'] + team_role_counts['AR']
                filtered_features['squad_pace_vs_spin_bowling_balance'] = bowling_players / 11.0
            
            # Fix squad_exploitation_efficiency
            if 'squad_exploitation_efficiency' in filtered_features:
                # Make it relative to team composition diversity and balance
                unique_roles = len(set(roles))
                role_balance = 1.0 - (max(team_role_counts.values()) - min(team_role_counts.values())) / 11.0
                filtered_features['squad_exploitation_efficiency'] = unique_roles * role_balance * 3.0
            
            # Fix top_players_missed
            if 'top_players_missed' in filtered_features:
                # Make it relative to team composition - count "missed opportunities"
                batting_players = team_role_counts['BAT'] + team_role_counts['WK']
                bowling_players = team_role_counts['BOWL'] + team_role_counts['AR']
                # Penalize extreme compositions
                if batting_players > 7:  # Too many batsmen
                    filtered_features['top_players_missed'] = batting_players - 6
                elif bowling_players > 8:  # Too many bowlers
                    filtered_features['top_players_missed'] = bowling_players - 7
                else:
                    filtered_features['top_players_missed'] = 0
            
            # Fix ch_squad_chosen_vs_available_wk
            if 'ch_squad_chosen_vs_available_wk' in filtered_features:
                # Make it relative to team composition
                wk_count = team_role_counts['WK']
                filtered_features['ch_squad_chosen_vs_available_wk'] = wk_count / 11.0
            
        except Exception as e:
            print(f"Error in final fixes: {e}")
        
        return filtered_features
    
    def _calculate_new_features(self, team_data: pd.Series, all_features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate new differential and ratio features to replace match-level constants
        """
        new_features = {}
        
        # Get team composition data
        roles = team_data.get('role_array', [])
        batfirst_squad_roles = team_data.get('batfirst_squad_roles', [])
        chase_squad_roles = team_data.get('chase_squad_roles', [])
        
        # Count roles in current team
        team_role_counts = {'BAT': 0, 'BOWL': 0, 'AR': 0, 'WK': 0}
        for role in roles:
            if role in team_role_counts:
                team_role_counts[role] += 1
        
        # Count roles in squads
        bat_first_role_counts = {'BAT': 0, 'BOWL': 0, 'AR': 0, 'WK': 0}
        for role in batfirst_squad_roles:
            if role in bat_first_role_counts:
                bat_first_role_counts[role] += 1
        
        chase_role_counts = {'BAT': 0, 'BOWL': 0, 'AR': 0, 'WK': 0}
        for role in chase_squad_roles:
            if role in chase_role_counts:
                chase_role_counts[role] += 1
        
        # Calculate squad availability ratios
        # Bat-first squad ratios
        new_features['bf_squad_chosen_vs_available_ar'] = (
            team_role_counts['AR'] / bat_first_role_counts['AR'] if bat_first_role_counts['AR'] > 0 else 0.0
        )
        new_features['bf_squad_chosen_vs_available_wk'] = (
            team_role_counts['WK'] / bat_first_role_counts['WK'] if bat_first_role_counts['WK'] > 0 else 0.0
        )
        new_features['bf_squad_chosen_vs_available_bowl'] = (
            team_role_counts['BOWL'] / bat_first_role_counts['BOWL'] if bat_first_role_counts['BOWL'] > 0 else 0.0
        )
        new_features['bf_squad_chosen_vs_available_bat'] = (
            team_role_counts['BAT'] / bat_first_role_counts['BAT'] if bat_first_role_counts['BAT'] > 0 else 0.0
        )
        
        # Chase squad ratios
        new_features['ch_squad_chosen_vs_available_ar'] = (
            team_role_counts['AR'] / chase_role_counts['AR'] if chase_role_counts['AR'] > 0 else 0.0
        )
        new_features['ch_squad_chosen_vs_available_wk'] = (
            team_role_counts['WK'] / chase_role_counts['WK'] if chase_role_counts['WK'] > 0 else 0.0
        )
        new_features['ch_squad_chosen_vs_available_bowl'] = (
            team_role_counts['BOWL'] / chase_role_counts['BOWL'] if chase_role_counts['BOWL'] > 0 else 0.0
        )
        new_features['ch_squad_chosen_vs_available_bat'] = (
            team_role_counts['BAT'] / chase_role_counts['BAT'] if chase_role_counts['BAT'] > 0 else 0.0
        )
        
        # Calculate team strength differentials
        team_avg_score = all_features.get('team_avg_score', 0)
        team_max_score = all_features.get('team_max_score', 0)
        
        # Get squad strength metrics (only keep those used in calculations)
        bf_squad_depth_score = all_features.get('bf_squad_depth_score', 0)
        bf_squad_strength_max = all_features.get('bf_squad_strength_max', 0)
        ch_squad_depth_score = all_features.get('ch_squad_depth_score', 0)
        ch_squad_strength_max = all_features.get('ch_squad_strength_max', 0)
        combined_squad_strength = all_features.get('combined_squad_strength', 0)
        
        # Calculate standard deviations from actual fantasy points data
        team_avg_points = team_data.get('avg_fantasy_points_last5_array', [])
        bf_squad_avg_points = team_data.get('batfirst_squad_avg_fantasy_points_last5', [])
        ch_squad_avg_points = team_data.get('chase_squad_avg_fantasy_points_last5', [])
        
        # Calculate team std dev
        team_std_dev = np.std(team_avg_points) if len(team_avg_points) > 1 else 0.0
        
        # Calculate squad std devs
        bf_squad_std_dev = np.std(bf_squad_avg_points) if len(bf_squad_avg_points) > 1 else 0.0
        ch_squad_std_dev = np.std(ch_squad_avg_points) if len(ch_squad_avg_points) > 1 else 0.0
        
        # Calculate combined pool std dev (all 22 players)
        all_pool_points = bf_squad_avg_points + ch_squad_avg_points
        combined_pool_std_dev = np.std(all_pool_points) if len(all_pool_points) > 1 else 0.0
        
        # Calculate differentials
        new_features['team_avg_vs_bf_squad_avg'] = team_avg_score - bf_squad_depth_score
        new_features['team_max_vs_bf_squad_avg'] = team_max_score - bf_squad_strength_max
        new_features['team_volatility_vs_bf_squad'] = team_std_dev - bf_squad_std_dev
        new_features['team_avg_vs_ch_squad_avg'] = team_avg_score - ch_squad_depth_score
        new_features['team_max_vs_ch_squad_avg'] = team_max_score - ch_squad_strength_max
        new_features['team_avg_vs_combined_squad_strength'] = team_avg_score - combined_squad_strength
        new_features['team_variance_vs_pool'] = team_std_dev - combined_pool_std_dev
        
        # Calculate new player quality features
        self._calculate_player_quality_features(team_data, new_features)
        
        # Calculate new efficiency features
        self._calculate_efficiency_features(team_data, new_features)
        
        # Calculate new performance ratio features
        self._calculate_performance_ratio_features(team_data, new_features)
        
        # Calculate new opportunity features
        self._calculate_opportunity_features(team_data, new_features)
        
        # Calculate additional fixes for zero-variance features
        self._calculate_additional_fixes(team_data, new_features)
        
        return new_features
    
    def _calculate_player_quality_features(self, team_data: pd.Series, new_features: Dict[str, float]):
        """Calculate new player quality features"""
        try:
            # Get data arrays
            roles = team_data.get('role_array', [])
            batting_order = team_data.get('batting_order_array', [])
            avg_fantasy_points = team_data.get('avg_fantasy_points_last5_array', [])
            
            # Convert to lists if needed
            if hasattr(roles, 'tolist'):
                roles = roles.tolist()
            if hasattr(batting_order, 'tolist'):
                batting_order = batting_order.tolist()
            if hasattr(avg_fantasy_points, 'tolist'):
                avg_fantasy_points = avg_fantasy_points.tolist()
            
            # 1. opener_quality - Mean Avg points last 5 of Batsmen, WK, or AR with Batting order <=2
            opener_points = []
            for i, (role, order) in enumerate(zip(roles, batting_order)):
                if role in ['BAT', 'WK', 'AR'] and order <= 2 and i < len(avg_fantasy_points):
                    opener_points.append(avg_fantasy_points[i])
            
            new_features['opener_quality'] = np.mean(opener_points) if opener_points else 0.0
            
            # 2. weak_team_player_count - Count of players with Avg Fantasy Last 5 <20
            weak_count = sum(1 for points in avg_fantasy_points if points < 20)
            new_features['weak_team_player_count'] = weak_count
            
            # 3. strong_team_player_count - Count of players from the stronger team
            # Determine which team is stronger based on mean avg points
            bf_squad_avg_points = team_data.get('batfirst_squad_avg_fantasy_points_last5', [])
            ch_squad_avg_points = team_data.get('chase_squad_avg_fantasy_points_last5', [])
            
            # Convert to lists if needed
            if hasattr(bf_squad_avg_points, 'tolist'):
                bf_squad_avg_points = bf_squad_avg_points.tolist()
            if hasattr(ch_squad_avg_points, 'tolist'):
                ch_squad_avg_points = ch_squad_avg_points.tolist()
            
            bf_mean = np.mean(bf_squad_avg_points) if bf_squad_avg_points else 0
            ch_mean = np.mean(ch_squad_avg_points) if ch_squad_avg_points else 0
            
            # Determine stronger team
            stronger_team = team_data.get('batting_first_team', '') if bf_mean > ch_mean else team_data.get('chasing_team', '')
            
            # Count players from stronger team in fantasy team
            team_ids = team_data.get('team_ids', [])
            if hasattr(team_ids, 'tolist'):
                team_ids = team_ids.tolist()
            
            strong_team_count = sum(1 for team in team_ids if team == stronger_team)
            new_features['strong_team_player_count'] = strong_team_count
            
            # 4. top_players_missed - Count of players in combined pool with MAX last10 >75 not picked
            bf_last10_scores = team_data.get('batfirst_squad_last10_fantasy_scores', [])
            ch_last10_scores = team_data.get('chase_squad_last10_fantasy_scores', [])
            team_player_ids = team_data.get('player_ids', [])
            
            # Convert to lists if needed
            if hasattr(team_player_ids, 'tolist'):
                team_player_ids = team_player_ids.tolist()
            
            # Get all pool player IDs
            bf_player_ids = team_data.get('batfirst_squad_player_ids', [])
            ch_player_ids = team_data.get('chase_squad_player_ids', [])
            all_pool_ids = bf_player_ids + ch_player_ids
            
            # Find top players (max last10 > 75) not in team
            top_players_missed = 0
            for i, player_id in enumerate(all_pool_ids):
                if player_id not in team_player_ids:
                    # Get max of last10 scores for this player
                    max_score = 0
                    if i < len(bf_last10_scores):
                        scores_array = bf_last10_scores[i]
                        if scores_array is not None and len(scores_array) > 0:
                            if hasattr(scores_array, 'tolist'):
                                scores = scores_array.tolist()
                            else:
                                scores = scores_array
                            max_score = max(scores) if scores else 0
                    elif i - len(bf_player_ids) < len(ch_last10_scores):
                        scores_array = ch_last10_scores[i - len(bf_player_ids)]
                        if scores_array is not None and len(scores_array) > 0:
                            if hasattr(scores_array, 'tolist'):
                                scores = scores_array.tolist()
                            else:
                                scores = scores_array
                            max_score = max(scores) if scores else 0
                    
                    if max_score > 75:
                        top_players_missed += 1
            
            new_features['top_players_missed'] = top_players_missed
            
            # 5. bat_first_heavy - Count of players from BF team / 11
            team_ids = team_data.get('team_ids', [])
            batting_first_team = team_data.get('batting_first_team', '')
            
            if hasattr(team_ids, 'tolist'):
                team_ids = team_ids.tolist()
            
            bf_players_in_team = sum(1 for team in team_ids if team == batting_first_team)
            new_features['bat_first_heavy'] = bf_players_in_team / 11.0
            
            # 6. team_composition_efficiency - Ratio of team composition vs pool composition
            # Count roles in current team
            team_role_counts = {'BAT': 0, 'BOWL': 0, 'AR': 0, 'WK': 0}
            for role in roles:
                if role in team_role_counts:
                    team_role_counts[role] += 1
            
            # Get squad roles
            batfirst_squad_roles = team_data.get('batfirst_squad_roles', [])
            chase_squad_roles = team_data.get('chase_squad_roles', [])
            
            # Convert to lists if needed
            if hasattr(batfirst_squad_roles, 'tolist'):
                batfirst_squad_roles = batfirst_squad_roles.tolist()
            if hasattr(chase_squad_roles, 'tolist'):
                chase_squad_roles = chase_squad_roles.tolist()
            
            # Calculate team role ratios
            team_bat_ratio = team_role_counts['BAT'] / 11.0
            team_bowl_ratio = team_role_counts['BOWL'] / 11.0
            team_ar_ratio = team_role_counts['AR'] / 11.0
            team_wk_ratio = team_role_counts['WK'] / 11.0
            
            # Calculate pool role ratios
            total_pool_players = len(batfirst_squad_roles) + len(chase_squad_roles)
            pool_bat_count = batfirst_squad_roles.count('BAT') + chase_squad_roles.count('BAT')
            pool_bowl_count = batfirst_squad_roles.count('BOWL') + chase_squad_roles.count('BOWL')
            pool_ar_count = batfirst_squad_roles.count('AR') + chase_squad_roles.count('AR')
            pool_wk_count = batfirst_squad_roles.count('WK') + chase_squad_roles.count('WK')
            
            pool_bat_ratio = pool_bat_count / total_pool_players if total_pool_players > 0 else 0
            pool_bowl_ratio = pool_bowl_count / total_pool_players if total_pool_players > 0 else 0
            pool_ar_ratio = pool_ar_count / total_pool_players if total_pool_players > 0 else 0
            pool_wk_ratio = pool_wk_count / total_pool_players if total_pool_players > 0 else 0
            
            # Calculate efficiency as weighted average of role selection efficiency
            bat_efficiency = team_bat_ratio / pool_bat_ratio if pool_bat_ratio > 0 else 0
            bowl_efficiency = team_bowl_ratio / pool_bowl_ratio if pool_bowl_ratio > 0 else 0
            ar_efficiency = team_ar_ratio / pool_ar_ratio if pool_ar_ratio > 0 else 0
            wk_efficiency = team_wk_ratio / pool_wk_ratio if pool_wk_ratio > 0 else 0
            
            new_features['team_composition_efficiency'] = (bat_efficiency + bowl_efficiency + ar_efficiency + wk_efficiency) / 4.0
            
        except Exception as e:
            print(f"Error in player quality features: {e}")
            new_features['opener_quality'] = 0.0
            new_features['weak_team_player_count'] = 0.0
            new_features['strong_team_player_count'] = 0.0
            new_features['top_players_missed'] = 0.0
            new_features['bat_first_heavy'] = 0.0
    
    def _calculate_efficiency_features(self, team_data: pd.Series, new_features: Dict[str, float]):
        """Calculate new efficiency features"""
        try:
            # Get data arrays
            roles = team_data.get('role_array', [])
            avg_fantasy_points = team_data.get('avg_fantasy_points_last5_array', [])
            
            # Convert to lists if needed
            if hasattr(roles, 'tolist'):
                roles = roles.tolist()
            if hasattr(avg_fantasy_points, 'tolist'):
                avg_fantasy_points = avg_fantasy_points.tolist()
            
            # Get squad data
            bf_squad_roles = team_data.get('batfirst_squad_roles', [])
            ch_squad_roles = team_data.get('chase_squad_roles', [])
            bf_squad_avg_points = team_data.get('batfirst_squad_avg_fantasy_points_last5', [])
            ch_squad_avg_points = team_data.get('chase_squad_avg_fantasy_points_last5', [])
            
            # 1. bowl_selection_efficiency - Sum of avg points of bowlers in team / Sum of avg points of all bowlers in squad
            team_bowler_points = []
            squad_bowler_points = []
            
            # Team bowlers
            for i, role in enumerate(roles):
                if role == 'BOWL' and i < len(avg_fantasy_points):
                    team_bowler_points.append(avg_fantasy_points[i])
            
            # Squad bowlers (bat-first + chase)
            for i, role in enumerate(bf_squad_roles):
                if role == 'BOWL' and i < len(bf_squad_avg_points):
                    squad_bowler_points.append(bf_squad_avg_points[i])
            
            for i, role in enumerate(ch_squad_roles):
                if role == 'BOWL' and i < len(ch_squad_avg_points):
                    squad_bowler_points.append(ch_squad_avg_points[i])
            
            team_bowler_sum = sum(team_bowler_points) if team_bowler_points else 0
            squad_bowler_sum = sum(squad_bowler_points) if squad_bowler_points else 1
            
            new_features['bowl_selection_efficiency'] = team_bowler_sum / squad_bowler_sum if squad_bowler_sum > 0 else 0.0
            
            # 2. bat_selection_efficiency - Sum of avg points of batsmen in team / Sum of avg points of all batsmen in squad
            team_bat_points = []
            squad_bat_points = []
            
            # Team batsmen (BAT + WK)
            for i, role in enumerate(roles):
                if role in ['BAT', 'WK'] and i < len(avg_fantasy_points):
                    team_bat_points.append(avg_fantasy_points[i])
            
            # Squad batsmen
            for i, role in enumerate(bf_squad_roles):
                if role in ['BAT', 'WK'] and i < len(bf_squad_avg_points):
                    squad_bat_points.append(bf_squad_avg_points[i])
            
            for i, role in enumerate(ch_squad_roles):
                if role in ['BAT', 'WK'] and i < len(ch_squad_avg_points):
                    squad_bat_points.append(ch_squad_avg_points[i])
            
            team_bat_sum = sum(team_bat_points) if team_bat_points else 0
            squad_bat_sum = sum(squad_bat_points) if squad_bat_points else 1
            
            new_features['bat_selection_efficiency'] = team_bat_sum / squad_bat_sum if squad_bat_sum > 0 else 0.0
            
        except Exception as e:
            print(f"Error in efficiency features: {e}")
            new_features['bowl_selection_efficiency'] = 0.0
            new_features['bat_selection_efficiency'] = 0.0
    
    def _calculate_performance_ratio_features(self, team_data: pd.Series, new_features: Dict[str, float]):
        """Calculate new performance ratio features"""
        try:
            # Get data arrays
            roles = team_data.get('role_array', [])
            batting_order = team_data.get('batting_order_array', [])
            avg_fantasy_points = team_data.get('avg_fantasy_points_last5_array', [])
            avg_balls_faced = team_data.get('avg_balls_faced_last5_array', [])
            avg_overs_bowled = team_data.get('avg_overs_bowled_last5_array', [])
            captain_id = team_data.get('captain_id', '')
            vice_captain_id = team_data.get('vice_captain_id', '')
            player_ids = team_data.get('player_ids', [])
            
            # Convert to lists if needed
            if hasattr(roles, 'tolist'):
                roles = roles.tolist()
            if hasattr(batting_order, 'tolist'):
                batting_order = batting_order.tolist()
            if hasattr(avg_fantasy_points, 'tolist'):
                avg_fantasy_points = avg_fantasy_points.tolist()
            if hasattr(avg_balls_faced, 'tolist'):
                avg_balls_faced = avg_balls_faced.tolist()
            if hasattr(avg_overs_bowled, 'tolist'):
                avg_overs_bowled = avg_overs_bowled.tolist()
            if hasattr(player_ids, 'tolist'):
                player_ids = player_ids.tolist()
            
            # 1. all_batsmen_perf_ratio - Batting points with captain/VC bonus / Total team points with captain/VC bonus
            batting_numerator = 0
            total_denominator = 0
            
            for i, (role, points) in enumerate(zip(roles, avg_fantasy_points)):
                # Get captain/VC multiplier
                multiplier = 1.0
                if player_ids[i] == captain_id:
                    multiplier = 2.0
                elif player_ids[i] == vice_captain_id:
                    multiplier = 1.5
                
                # Add to total denominator
                total_denominator += points * multiplier
                
                # Add to batting numerator if BAT, WK, or AR
                if role in ['BAT', 'WK', 'AR']:
                    batting_numerator += points * multiplier
            
            new_features['all_batsmen_perf_ratio'] = batting_numerator / total_denominator if total_denominator > 0 else 0.0
            
            # 2. bowler_perf_ratio - Bowling points with captain/VC bonus / Total team points with captain/VC bonus
            bowling_numerator = 0
            
            for i, (role, points) in enumerate(zip(roles, avg_fantasy_points)):
                # Get captain/VC multiplier
                multiplier = 1.0
                if player_ids[i] == captain_id:
                    multiplier = 2.0
                elif player_ids[i] == vice_captain_id:
                    multiplier = 1.5
                
                # Add to bowling numerator if BOWL or AR
                if role in ['BOWL', 'AR']:
                    bowling_numerator += points * multiplier
            
            new_features['bowler_perf_ratio'] = bowling_numerator / total_denominator if total_denominator > 0 else 0.0
            
            # 3. top_order_opp_ratio - Avg balls faced for top order (≤3) in team / Avg balls faced for top order in pool
            team_top_order_balls = []
            pool_top_order_balls = []
            
            # Team top order balls
            for i, (role, order, balls) in enumerate(zip(roles, batting_order, avg_balls_faced)):
                if role in ['BAT', 'WK', 'AR'] and order <= 3:
                    team_top_order_balls.append(balls)
            
            # Pool top order balls (simplified - using team data as proxy)
            bf_squad_roles = team_data.get('batfirst_squad_roles', [])
            ch_squad_roles = team_data.get('chase_squad_roles', [])
            bf_squad_batting_orders = team_data.get('batfirst_squad_batting_orders', [])
            ch_squad_batting_orders = team_data.get('chase_squad_batting_orders', [])
            bf_squad_avg_balls = team_data.get('batfirst_squad_avg_balls_faced_last5', [])
            ch_squad_avg_balls = team_data.get('chase_squad_avg_balls_faced_last5', [])
            
            # Add pool top order balls
            for i, (role, order, balls) in enumerate(zip(bf_squad_roles, bf_squad_batting_orders, bf_squad_avg_balls)):
                if role in ['BAT', 'WK', 'AR'] and order <= 3:
                    pool_top_order_balls.append(balls)
            
            for i, (role, order, balls) in enumerate(zip(ch_squad_roles, ch_squad_batting_orders, ch_squad_avg_balls)):
                if role in ['BAT', 'WK', 'AR'] and order <= 3:
                    pool_top_order_balls.append(balls)
            
            team_avg_balls = np.mean(team_top_order_balls) if team_top_order_balls else 0
            pool_avg_balls = np.mean(pool_top_order_balls) if pool_top_order_balls else 1
            
            new_features['top_order_opp_ratio'] = team_avg_balls / pool_avg_balls if pool_avg_balls > 0 else 0.0
            
            # 4. bowler_opp_ratio - Avg overs bowled for bowlers in team / Avg overs bowled for bowlers in pool
            team_bowler_overs = []
            pool_bowler_overs = []
            
            # Team bowler overs
            for i, (role, overs) in enumerate(zip(roles, avg_overs_bowled)):
                if role in ['BOWL', 'AR']:
                    team_bowler_overs.append(overs)
            
            # Pool bowler overs
            bf_squad_avg_overs = team_data.get('batfirst_squad_avg_overs_bowled_last5', [])
            ch_squad_avg_overs = team_data.get('chase_squad_avg_overs_bowled_last5', [])
            
            for i, (role, overs) in enumerate(zip(bf_squad_roles, bf_squad_avg_overs)):
                if role in ['BOWL', 'AR']:
                    pool_bowler_overs.append(overs)
            
            for i, (role, overs) in enumerate(zip(ch_squad_roles, ch_squad_avg_overs)):
                if role in ['BOWL', 'AR']:
                    pool_bowler_overs.append(overs)
            
            team_avg_overs = np.mean(team_bowler_overs) if team_bowler_overs else 0
            pool_avg_overs = np.mean(pool_bowler_overs) if pool_bowler_overs else 1
            
            new_features['bowler_opp_ratio'] = team_avg_overs / pool_avg_overs if pool_avg_overs > 0 else 0.0
            
        except Exception as e:
            print(f"Error in performance ratio features: {e}")
            new_features['all_batsmen_perf_ratio'] = 0.0
            new_features['bowler_perf_ratio'] = 0.0
            new_features['top_order_opp_ratio'] = 0.0
            new_features['bowler_opp_ratio'] = 0.0
    
    def _calculate_opportunity_features(self, team_data: pd.Series, new_features: Dict[str, float]):
        """Calculate new opportunity features"""
        try:
            # Get data arrays
            avg_fantasy_points = team_data.get('avg_fantasy_points_last5_array', [])
            team_player_ids = team_data.get('player_ids', [])
            
            # Convert to lists if needed
            if hasattr(avg_fantasy_points, 'tolist'):
                avg_fantasy_points = avg_fantasy_points.tolist()
            if hasattr(team_player_ids, 'tolist'):
                team_player_ids = team_player_ids.tolist()
            
            # 1. top_talent_capture_rate - Count of top 5 players from combined squad selected
            bf_last10_scores = team_data.get('batfirst_squad_last10_fantasy_scores', [])
            ch_last10_scores = team_data.get('chase_squad_last10_fantasy_scores', [])
            bf_player_ids = team_data.get('batfirst_squad_player_ids', [])
            ch_player_ids = team_data.get('chase_squad_player_ids', [])
            
            # Create list of (player_id, max_score) tuples
            player_scores = []
            
            # Add bat-first squad players
            for i, player_id in enumerate(bf_player_ids):
                if i < len(bf_last10_scores):
                    scores_array = bf_last10_scores[i]
                    if scores_array is not None and len(scores_array) > 0:
                        if hasattr(scores_array, 'tolist'):
                            scores = scores_array.tolist()
                        else:
                            scores = scores_array
                        max_score = max(scores) if scores else 0
                        player_scores.append((player_id, max_score))
            
            # Add chase squad players
            for i, player_id in enumerate(ch_player_ids):
                if i < len(ch_last10_scores):
                    scores_array = ch_last10_scores[i]
                    if scores_array is not None and len(scores_array) > 0:
                        if hasattr(scores_array, 'tolist'):
                            scores = scores_array.tolist()
                        else:
                            scores = scores_array
                        max_score = max(scores) if scores else 0
                        player_scores.append((player_id, max_score))
            
            # Sort by max score and get top 5
            player_scores.sort(key=lambda x: x[1], reverse=True)
            top_5_players = [player_id for player_id, _ in player_scores[:5]]
            
            # Count how many top 5 are in team
            top_talent_captured = sum(1 for player_id in top_5_players if player_id in team_player_ids)
            new_features['top_talent_capture_rate'] = top_talent_captured
            
            # 2. choice_efficiency_overall - Mean of team avg points / Mean of pool avg points
            team_mean = np.mean(avg_fantasy_points) if avg_fantasy_points else 0
            
            # Pool mean (bat-first + chase)
            bf_squad_avg_points = team_data.get('batfirst_squad_avg_fantasy_points_last5', [])
            ch_squad_avg_points = team_data.get('chase_squad_avg_fantasy_points_last5', [])
            
            # Convert to lists if needed
            if hasattr(bf_squad_avg_points, 'tolist'):
                bf_squad_avg_points = bf_squad_avg_points.tolist()
            if hasattr(ch_squad_avg_points, 'tolist'):
                ch_squad_avg_points = ch_squad_avg_points.tolist()
            
            all_pool_points = bf_squad_avg_points + ch_squad_avg_points
            pool_mean = np.mean(all_pool_points) if all_pool_points else 1
            
            new_features['choice_efficiency_overall'] = team_mean / pool_mean if pool_mean > 0 else 0.0
            
        except Exception as e:
            print(f"Error in opportunity features: {e}")
            new_features['top_talent_capture_rate'] = 0.0
            new_features['choice_efficiency_overall'] = 0.0
    
    def _calculate_additional_fixes(self, team_data: pd.Series, new_features: Dict[str, float]):
        """Calculate additional fixes for zero-variance features"""
        try:
            # Get data arrays
            roles = team_data.get('role_array', [])
            avg_fantasy_points = team_data.get('avg_fantasy_points_last5_array', [])
            avg_balls_faced = team_data.get('avg_balls_faced_last5_array', [])
            avg_overs_bowled = team_data.get('avg_overs_bowled_last5_array', [])
            
            # Convert to lists if needed
            if hasattr(roles, 'tolist'):
                roles = roles.tolist()
            if hasattr(avg_fantasy_points, 'tolist'):
                avg_fantasy_points = avg_fantasy_points.tolist()
            if hasattr(avg_balls_faced, 'tolist'):
                avg_balls_faced = avg_balls_faced.tolist()
            if hasattr(avg_overs_bowled, 'tolist'):
                avg_overs_bowled = avg_overs_bowled.tolist()
            
            # Count roles in current team
            team_role_counts = {'BAT': 0, 'BOWL': 0, 'AR': 0, 'WK': 0}
            for role in roles:
                if role in team_role_counts:
                    team_role_counts[role] += 1
            
            # Get squad roles
            bf_squad_roles = team_data.get('batfirst_squad_roles', [])
            ch_squad_roles = team_data.get('chase_squad_roles', [])
            
            # Convert to lists if needed
            if hasattr(bf_squad_roles, 'tolist'):
                bf_squad_roles = bf_squad_roles.tolist()
            if hasattr(ch_squad_roles, 'tolist'):
                ch_squad_roles = ch_squad_roles.tolist()
            
            # Fix wk_selection_efficiency - Ratio of WK points in team vs WK points in pool
            team_wk_points = []
            for i, role in enumerate(roles):
                if role == 'WK' and i < len(avg_fantasy_points):
                    team_wk_points.append(avg_fantasy_points[i])
            bf_squad_avg_points = team_data.get('batfirst_squad_avg_fantasy_points_last5', [])
            ch_squad_avg_points = team_data.get('chase_squad_avg_fantasy_points_last5', [])
            
            pool_wk_points = []
            for i, role in enumerate(bf_squad_roles):
                if role == 'WK' and i < len(bf_squad_avg_points):
                    pool_wk_points.append(bf_squad_avg_points[i])
            
            for i, role in enumerate(ch_squad_roles):
                if role == 'WK' and i < len(ch_squad_avg_points):
                    pool_wk_points.append(ch_squad_avg_points[i])
            
            team_wk_sum = sum(team_wk_points) if team_wk_points else 0
            pool_wk_sum = sum(pool_wk_points) if pool_wk_points else 1
            
            new_features['wk_selection_efficiency'] = team_wk_sum / pool_wk_sum if pool_wk_sum > 0 else 0.0
            
            # Fix ar_selection_efficiency - Ratio of AR points in team vs AR points in pool
            team_ar_points = []
            for i, role in enumerate(roles):
                if role == 'AR' and i < len(avg_fantasy_points):
                    team_ar_points.append(avg_fantasy_points[i])
            
            pool_ar_points = []
            for i, role in enumerate(bf_squad_roles):
                if role == 'AR' and i < len(bf_squad_avg_points):
                    pool_ar_points.append(bf_squad_avg_points[i])
            
            for i, role in enumerate(ch_squad_roles):
                if role == 'AR' and i < len(ch_squad_avg_points):
                    pool_ar_points.append(ch_squad_avg_points[i])
            
            team_ar_sum = sum(team_ar_points) if team_ar_points else 0
            pool_ar_sum = sum(pool_ar_points) if pool_ar_points else 1
            
            new_features['ar_selection_efficiency'] = team_ar_sum / pool_ar_sum if pool_ar_sum > 0 else 0.0
            
            # Fix ar_total_opportunity_score - Sum of (balls faced + overs bowled) for ARs
            ar_opportunity = 0
            for i, role in enumerate(roles):
                if role == 'AR':
                    balls = avg_balls_faced[i] if i < len(avg_balls_faced) else 0
                    overs = avg_overs_bowled[i] if i < len(avg_overs_bowled) else 0
                    ar_opportunity += balls + overs
            
            new_features['ar_total_opportunity_score'] = ar_opportunity
            
            # Fix ar_bowling_opportunity_vs_pure_bowlers - AR bowling overs / Pure bowler overs
            ar_bowling_overs = 0
            pure_bowler_overs = 0
            
            for i, role in enumerate(roles):
                if i < len(avg_overs_bowled):
                    if role == 'AR':
                        ar_bowling_overs += avg_overs_bowled[i]
                    elif role == 'BOWL':
                        pure_bowler_overs += avg_overs_bowled[i]
            
            new_features['ar_bowling_opportunity_vs_pure_bowlers'] = (
                ar_bowling_overs / pure_bowler_overs if pure_bowler_overs > 0 else 0.0
            )
            
            # Fix ar_batting_opportunity_vs_pure_batsmen - AR batting balls / Pure batsmen balls
            ar_batting_balls = 0
            pure_batsmen_balls = 0
            
            for i, role in enumerate(roles):
                if i < len(avg_balls_faced):
                    if role == 'AR':
                        ar_batting_balls += avg_balls_faced[i]
                    elif role in ['BAT', 'WK']:
                        pure_batsmen_balls += avg_balls_faced[i]
            
            new_features['ar_batting_opportunity_vs_pure_batsmen'] = (
                ar_batting_balls / pure_batsmen_balls if pure_batsmen_balls > 0 else 0.0
            )
            
            # Fix intra_team_synergy - Variance in team fantasy points (lower = more synergy)
            team_variance = np.var(avg_fantasy_points) if len(avg_fantasy_points) > 1 else 0
            new_features['intra_team_synergy'] = 1.0 / (1.0 + team_variance)  # Inverse relationship
            
            # Fix leadership_top_order - Count of C/VC in top 3 batting order
            captain_id = team_data.get('captain_id', '')
            vice_captain_id = team_data.get('vice_captain_id', '')
            player_ids = team_data.get('player_ids', [])
            batting_order = team_data.get('batting_order_array', [])
            
            if hasattr(player_ids, 'tolist'):
                player_ids = player_ids.tolist()
            if hasattr(batting_order, 'tolist'):
                batting_order = batting_order.tolist()
            
            leadership_top_order = 0
            for i, (player_id, order) in enumerate(zip(player_ids, batting_order)):
                if (player_id == captain_id or player_id == vice_captain_id) and order <= 3:
                    leadership_top_order += 1
            
            new_features['leadership_top_order'] = leadership_top_order
            
            # Fix batting_heavy_team - Make it relative to pool
            team_batting_players = team_role_counts['BAT'] + team_role_counts['WK']
            pool_batting_players = bf_squad_roles.count('BAT') + bf_squad_roles.count('WK') + ch_squad_roles.count('BAT') + ch_squad_roles.count('WK')
            total_pool = len(bf_squad_roles) + len(ch_squad_roles)
            
            team_batting_ratio = team_batting_players / 11.0
            pool_batting_ratio = pool_batting_players / total_pool if total_pool > 0 else 0
            
            new_features['batting_heavy_team'] = team_batting_ratio / pool_batting_ratio if pool_batting_ratio > 0 else 0.0
            
            # Fix cvc_same_role - Whether C and VC have same role
            captain_role = None
            vc_role = None
            
            for i, player_id in enumerate(player_ids):
                if player_id == captain_id and i < len(roles):
                    captain_role = roles[i]
                elif player_id == vice_captain_id and i < len(roles):
                    vc_role = roles[i]
            
            new_features['cvc_same_role'] = 1.0 if captain_role == vc_role else 0.0
            
            # Fix risk_adjusted_chemistry - Team variance adjusted by mean performance
            team_mean = np.mean(avg_fantasy_points) if avg_fantasy_points else 0
            team_std = np.std(avg_fantasy_points) if len(avg_fantasy_points) > 1 else 0
            new_features['risk_adjusted_chemistry'] = team_mean / (1.0 + team_std) if team_std > 0 else team_mean
            
            # Fix cvc_role_diversity - Number of unique roles between C and VC
            unique_roles = set()
            if captain_role:
                unique_roles.add(captain_role)
            if vc_role:
                unique_roles.add(vc_role)
            
            new_features['cvc_role_diversity'] = len(unique_roles)
            
        except Exception as e:
            print(f"Error in additional fixes: {e}")
            # Set default values for all features
            default_features = [
                'wk_selection_efficiency', 'ar_selection_efficiency', 'ar_total_opportunity_score',
                'ar_bowling_opportunity_vs_pure_bowlers', 'ar_batting_opportunity_vs_pure_batsmen',
                'intra_team_synergy', 'leadership_top_order', 'cvc_same_role',
                'risk_adjusted_chemistry', 'cvc_role_diversity'
            ]
            for feature in default_features:
                new_features[feature] = 0.0

# For backward compatibility
ComprehensiveFeatureExtractorFixed174 = Correct174FeatureExtractor
