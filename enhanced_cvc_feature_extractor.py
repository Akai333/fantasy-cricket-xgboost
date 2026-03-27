#!/usr/bin/env python3
"""
ENHANCED CAPTAIN/VICE-CAPTAIN FEATURE EXTRACTOR
Comprehensive C/VC feature extraction with:
- Multiplier impact simulation (2x captain, 1.5x VC)
- Role-aware expected fantasy points calculation  
- Squad-based choice quality assessment
- Ownership bias analysis with logarithmic field expectation
- Pitch descriptor encoding (26 characteristics)
- Field psychology and contrarian analysis
"""

import pandas as pd
import numpy as np
import math
import ast
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class EnhancedCVCFeatureExtractor:
    """
    PRODUCTION-READY C/VC feature extractor addressing model blindness to:
    - Captain/VC multiplier impact (currently zero importance)
    - Choice quality within available squad constraints
    - Field bias and contrarian opportunities
    - Pitch conditions affecting performance
    """
    
    def __init__(self):
        """Initialize with comprehensive mappings and thresholds"""
        self.role_mapping = {'BAT': 1, 'BOWL': 2, 'AR': 3, 'WK': 4}
        
        # Pitch descriptor categories (26 total from pitch modal)
        self.pitch_descriptors = [
            'green', 'grassy', 'damp', 'wet', 'moist', 'sticky', 'sticky_wicket',  # Grass/Moisture (7)
            'dry', 'dusty', 'cracks', 'crumbling', 'rough',  # Dry/Cracked (5)
            'flat', 'hard', 'true', 'even_bounce', 'dead', 'batting_paradise', 'road',  # Batting-Friendly (7)
            'live', 'lively', 'seam', 'swing',  # Pace-Related (4)
            'turn', 'turning_track', 'variable_bounce'  # Spin-Related (3)
        ]
        
        print("🚀 Enhanced C/VC Feature Extractor initialized")
        print(f"✅ {len(self.pitch_descriptors)} pitch descriptors mapped")
        print("🎯 Focus: Multiplier impact, choice quality, field bias, pitch conditions")
    
    def safe_array_extract(self, array_data, default_value=0.0, target_length=11):
        """Safely extract numerical values from arrays with robust error handling"""
        try:
            if array_data is None:
                return [default_value] * target_length
            
            # Convert to list if needed
            if hasattr(array_data, 'tolist'):
                array_data = array_data.tolist()
            elif isinstance(array_data, str):
                array_data = ast.literal_eval(array_data)
            
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
    
    def safe_divide(self, numerator, denominator, default=0.0):
        """Safe division with NaN and zero handling, rounded to eliminate floating-point precision errors"""
        try:
            if denominator is None or np.isnan(denominator) or denominator == 0:
                return default
            if numerator is None or np.isnan(numerator):
                return default
            
            result = float(numerator) / float(denominator)
            # Round to 8 decimal places to eliminate floating-point precision artifacts
            result = round(result, 8) if not np.isnan(result) else default
            return result
        except:
            return default
    
    def safe_log(self, value, base=math.e, default=0.0):
        """Safe logarithm calculation for ownership analysis"""
        try:
            if value is None or np.isnan(value) or value <= 0:
                return default
            if base == math.e:
                return math.log(value)
            else:
                return math.log(value) / math.log(base)
        except:
            return default
    
    def calculate_role_aware_expected_fp(self, player_data: Dict) -> float:
        """
        Calculate expected fantasy points based on role-specific usage patterns
        BAT/WK: fp_per_ball × avg_balls_faced
        BOWL: fp_per_over × avg_overs_bowled  
        AR: weighted combination of both
        """
        try:
            role = player_data.get('role', 'BAT')
            avg_fp = player_data.get('avg_fantasy_points_last5', 0.0)
            avg_balls = player_data.get('avg_balls_faced_last5', 0.0)
            avg_overs = player_data.get('avg_overs_bowled_last5', 0.0)
            
            if role in ['BAT', 'WK']:
                fp_per_ball = self.safe_divide(avg_fp, avg_balls, 0.0)
                return fp_per_ball * avg_balls
                
            elif role == 'BOWL':
                fp_per_over = self.safe_divide(avg_fp, avg_overs, 0.0)
                return fp_per_over * avg_overs
                
            elif role == 'AR':
                # Weighted combination for all-rounders
                fp_bat = self.safe_divide(avg_fp, avg_balls, 0.0) * avg_balls * 0.6
                fp_bowl = self.safe_divide(avg_fp, avg_overs, 0.0) * avg_overs * 0.4
                return fp_bat + fp_bowl
            
            return avg_fp  # Fallback
            
        except:
            return player_data.get('avg_fantasy_points_last5', 0.0)
    
    def extract_enhanced_cvc_features(self, team_data: pd.Series) -> Dict[str, float]:
        """
        Extract comprehensive C/VC features addressing current model blindness
        
        CRITICAL AREAS:
        1. Multiplier Impact Simulation
        2. Squad-Based Choice Quality  
        3. Ownership Bias Analysis (with logarithms)
        4. Pitch Condition Encoding
        5. Field Psychology Assessment
        """
        features = {}
        
        try:
            # === DATA EXTRACTION ===
            captain_id = team_data.get('captain_id', '')
            vc_id = team_data.get('vice_captain_id', '')
            
            # Selected team arrays - handle different data types correctly
            player_ids_raw = team_data.get('player_ids', [])
            roles_raw = team_data.get('roles', [])
            
            # Convert to lists if needed
            if hasattr(player_ids_raw, 'tolist'):
                player_ids = player_ids_raw.tolist()
            else:
                player_ids = list(player_ids_raw)
                
            if hasattr(roles_raw, 'tolist'):
                roles = roles_raw.tolist()
            else:
                roles = list(roles_raw)
            
            # Numerical arrays
            avg_fp_array = self.safe_array_extract(team_data.get('avg_fantasy_points_last5_array', []))
            avg_balls_array = self.safe_array_extract(team_data.get('avg_balls_faced_last5_array', []))
            avg_overs_array = self.safe_array_extract(team_data.get('avg_overs_bowled_last5_array', []))
            
            # Ready for production - debug output removed
            last10_scores = team_data.get('last10_fantasy_scores_array', [])
            
            # OWNERSHIP FEATURES REMOVED - We don't want these features
            
            # Pitch descriptors  
            pitch_descriptors = self.safe_array_extract(team_data.get('pitch_descriptors', []), 0.0, 26)
            
            # Squad context data
            bf_squad_fp = self.safe_array_extract(team_data.get('batfirst_squad_avg_fantasy_points_last5', []))
            bf_squad_balls = self.safe_array_extract(team_data.get('batfirst_squad_avg_balls_faced_last5', []))
            bf_squad_overs = self.safe_array_extract(team_data.get('batfirst_squad_avg_overs_bowled_last5', []))
            bf_squad_roles = self.safe_array_extract(team_data.get('batfirst_squad_roles', []), '', len(bf_squad_fp))
            
            ch_squad_fp = self.safe_array_extract(team_data.get('chase_squad_avg_fantasy_points_last5', []))
            ch_squad_balls = self.safe_array_extract(team_data.get('chase_squad_avg_balls_faced_last5', []))
            ch_squad_overs = self.safe_array_extract(team_data.get('chase_squad_avg_overs_bowled_last5', []))
            ch_squad_roles = self.safe_array_extract(team_data.get('chase_squad_roles', []), '', len(ch_squad_fp))
            
            # === 1. CAPTAIN ANALYSIS ===
            features.update(self._extract_captain_features(
                captain_id, player_ids, roles, avg_fp_array, avg_balls_array, avg_overs_array,
                last10_scores, bf_squad_fp, bf_squad_balls, 
                bf_squad_overs, bf_squad_roles, ch_squad_fp, ch_squad_balls, ch_squad_overs, ch_squad_roles
            ))
            
            # === 2. VICE-CAPTAIN ANALYSIS ===
            features.update(self._extract_vice_captain_features(
                vc_id, player_ids, roles, avg_fp_array, avg_balls_array, avg_overs_array,
                last10_scores, bf_squad_fp, bf_squad_balls,
                bf_squad_overs, bf_squad_roles, ch_squad_fp, ch_squad_balls, ch_squad_overs, ch_squad_roles
            ))
            
            # === 3. COMBINED C/VC IMPACT ===
            features.update(self._extract_combined_cvc_features(
                captain_id, vc_id, player_ids, roles, avg_fp_array, avg_balls_array, avg_overs_array
            ))
            
            # === 4. PITCH CONDITION FEATURES ===
            features.update(self._extract_pitch_features(pitch_descriptors))
            
            return features
            
        except Exception as e:
            print(f"❌ Error in enhanced C/VC feature extraction: {e}")
            return self._get_default_cvc_features()
    
    def _extract_captain_features(self, captain_id, player_ids, roles, avg_fp_array, 
                                 avg_balls_array, avg_overs_array, last10_scores, 
                                 bf_squad_fp, bf_squad_balls, bf_squad_overs, bf_squad_roles, 
                                 ch_squad_fp, ch_squad_balls, ch_squad_overs, ch_squad_roles) -> Dict[str, float]:
        """Extract comprehensive captain-specific features"""
        features = {}
        
        try:
            # Find captain in team
            captain_idx = -1
            for i, pid in enumerate(player_ids):
                if str(pid) == str(captain_id):
                    captain_idx = i
                    break
            
            if captain_idx >= 0:
                # Captain basic stats
                cap_role = roles[captain_idx] if captain_idx < len(roles) else 'BAT'
                cap_avg_fp = avg_fp_array[captain_idx] if captain_idx < len(avg_fp_array) else 0.0
                cap_balls = avg_balls_array[captain_idx] if captain_idx < len(avg_balls_array) else 0.0
                cap_overs = avg_overs_array[captain_idx] if captain_idx < len(avg_overs_array) else 0.0
                
                # Production ready - debug removed
                
                # Captain expected FP (role-aware)
                cap_data = {
                    'role': cap_role,
                    'avg_fantasy_points_last5': cap_avg_fp,
                    'avg_balls_faced_last5': cap_balls,
                    'avg_overs_bowled_last5': cap_overs
                }
                cap_expected_fp = self.calculate_role_aware_expected_fp(cap_data)
                
                # Captain max score from last 10
                cap_max_fp = 0.0
                cap_std_fp = 0.0
                if captain_idx < len(last10_scores):
                    # Handle both list and numpy array data types
                    cap_scores_raw = last10_scores[captain_idx]
                    if isinstance(cap_scores_raw, (list, np.ndarray)) and len(cap_scores_raw) > 0:
                        cap_scores = np.array(cap_scores_raw).flatten()  # Ensure it's a 1D array
                        cap_scores = cap_scores[~np.isnan(cap_scores)]  # Remove NaN values
                        if len(cap_scores) > 0:
                            cap_max_fp = float(np.max(cap_scores))
                            cap_std_fp = float(np.std(cap_scores)) if len(cap_scores) > 1 else 0.0
                
                # Basic captain features
                features['captain_avg_fp'] = cap_avg_fp
                features['captain_max_fp'] = cap_max_fp
                features['captain_std_fp'] = cap_std_fp
                features['captain_expected_fp'] = cap_expected_fp
                # Calculate captain_fp_per_ball based on role
                if cap_role in ['BAT', 'WK']:
                    features['captain_fp_per_ball'] = self.safe_divide(cap_avg_fp, cap_balls + 0.1)
                elif cap_role == 'BOWL':
                    features['captain_fp_per_ball'] = self.safe_divide(cap_avg_fp, (cap_overs + 0.1) * 6 + 0.1)
                elif cap_role == 'AR':
                    features['captain_fp_per_ball'] = self.safe_divide(cap_avg_fp, cap_balls + (cap_overs + 0.1) * 6 + 0.1)
                else:
                    features['captain_fp_per_ball'] = 0.0
                
                # Captain choice quality vs squad
                # FIXED: Safe array combination to prevent broadcasting errors
                combined_squad_fp = []
                combined_squad_balls = []
                combined_squad_overs = []
                combined_squad_roles = []
                
                # Safely combine arrays with proper type checking
                if isinstance(bf_squad_fp, (list, np.ndarray)) and isinstance(ch_squad_fp, (list, np.ndarray)):
                    combined_squad_fp = list(bf_squad_fp) + list(ch_squad_fp)
                elif isinstance(bf_squad_fp, (list, np.ndarray)):
                    combined_squad_fp = list(bf_squad_fp)
                elif isinstance(ch_squad_fp, (list, np.ndarray)):
                    combined_squad_fp = list(ch_squad_fp)
                
                if isinstance(bf_squad_balls, (list, np.ndarray)) and isinstance(ch_squad_balls, (list, np.ndarray)):
                    combined_squad_balls = list(bf_squad_balls) + list(ch_squad_balls)
                elif isinstance(bf_squad_balls, (list, np.ndarray)):
                    combined_squad_balls = list(bf_squad_balls)
                elif isinstance(ch_squad_balls, (list, np.ndarray)):
                    combined_squad_balls = list(ch_squad_balls)
                
                if isinstance(bf_squad_overs, (list, np.ndarray)) and isinstance(ch_squad_overs, (list, np.ndarray)):
                    combined_squad_overs = list(bf_squad_overs) + list(ch_squad_overs)
                elif isinstance(bf_squad_overs, (list, np.ndarray)):
                    combined_squad_overs = list(bf_squad_overs)
                elif isinstance(ch_squad_overs, (list, np.ndarray)):
                    combined_squad_overs = list(ch_squad_overs)
                
                if isinstance(bf_squad_roles, (list, np.ndarray)) and isinstance(ch_squad_roles, (list, np.ndarray)):
                    combined_squad_roles = list(bf_squad_roles) + list(ch_squad_roles)
                elif isinstance(bf_squad_roles, (list, np.ndarray)):
                    combined_squad_roles = list(bf_squad_roles)
                elif isinstance(ch_squad_roles, (list, np.ndarray)):
                    combined_squad_roles = list(ch_squad_roles)
                
                if combined_squad_fp:
                    # Calculate expected FP for all squad players
                    squad_expected_fps = []
                    for i in range(len(combined_squad_fp)):
                        squad_data = {
                            'role': combined_squad_roles[i] if i < len(combined_squad_roles) else 'BAT',
                            'avg_fantasy_points_last5': combined_squad_fp[i],
                            'avg_balls_faced_last5': combined_squad_balls[i] if i < len(combined_squad_balls) else 0.0,
                            'avg_overs_bowled_last5': combined_squad_overs[i] if i < len(combined_squad_overs) else 0.0
                        }
                        squad_expected_fps.append(self.calculate_role_aware_expected_fp(squad_data))
                    
                    # Captain ranking in squad
                    sorted_squad_fps = sorted(squad_expected_fps, reverse=True)
                    cap_rank = len([x for x in sorted_squad_fps if x > cap_expected_fp]) + 1
                    features['captain_rank_in_squad'] = cap_rank / len(sorted_squad_fps) if sorted_squad_fps else 1.0
                    
                    # Captain gap from best in squad
                    best_squad_fp = max(squad_expected_fps) if squad_expected_fps else cap_expected_fp
                    features['captain_gap_from_best'] = 2.0 * (best_squad_fp - cap_expected_fp)  # 2x multiplier impact
                    
                    # Captain boost efficiency
                    features['captain_boost_ratio'] = self.safe_divide(2.0 * cap_expected_fp, sum(squad_expected_fps))
                else:
                    features['captain_rank_in_squad'] = 0.5
                    features['captain_gap_from_best'] = 0.0
                    features['captain_boost_ratio'] = 0.0
                    
            else:
                # Captain not found in team - use defaults
                features.update(self._get_default_captain_features())
                
        except Exception as e:
            print(f"⚠️ Error in captain feature extraction: {e}")
            features.update(self._get_default_captain_features())
        
        return features
    
    def _extract_vice_captain_features(self, vc_id, player_ids, roles, avg_fp_array,
                                      avg_balls_array, avg_overs_array, last10_scores, 
                                      bf_squad_fp, bf_squad_balls, bf_squad_overs, bf_squad_roles, 
                                      ch_squad_fp, ch_squad_balls, ch_squad_overs, ch_squad_roles) -> Dict[str, float]:
        """Extract comprehensive vice-captain-specific features"""
        features = {}
        
        try:
            # Find VC in team
            vc_idx = -1
            for i, pid in enumerate(player_ids):
                if str(pid) == str(vc_id):
                    vc_idx = i
                    break
            
            if vc_idx >= 0:
                # VC basic stats
                vc_role = roles[vc_idx] if vc_idx < len(roles) else 'BAT'
                vc_avg_fp = avg_fp_array[vc_idx] if vc_idx < len(avg_fp_array) else 0.0
                vc_balls = avg_balls_array[vc_idx] if vc_idx < len(avg_balls_array) else 0.0
                vc_overs = avg_overs_array[vc_idx] if vc_idx < len(avg_overs_array) else 0.0
                
                # VC expected FP (role-aware)
                vc_data = {
                    'role': vc_role,
                    'avg_fantasy_points_last5': vc_avg_fp,
                    'avg_balls_faced_last5': vc_balls,
                    'avg_overs_bowled_last5': vc_overs
                }
                vc_expected_fp = self.calculate_role_aware_expected_fp(vc_data)
                
                # VC max score from last 10
                vc_max_fp = 0.0
                vc_std_fp = 0.0
                if vc_idx < len(last10_scores):
                    # Handle both list and numpy array data types
                    vc_scores_raw = last10_scores[vc_idx]
                    if isinstance(vc_scores_raw, (list, np.ndarray)) and len(vc_scores_raw) > 0:
                        vc_scores = np.array(vc_scores_raw).flatten()  # Ensure it's a 1D array
                        vc_scores = vc_scores[~np.isnan(vc_scores)]  # Remove NaN values
                        if len(vc_scores) > 0:
                            vc_max_fp = float(np.max(vc_scores))
                            vc_std_fp = float(np.std(vc_scores)) if len(vc_scores) > 1 else 0.0
                
                # Basic VC features
                features['vc_avg_fp'] = vc_avg_fp
                features['vc_max_fp'] = vc_max_fp
                features['vc_std_fp'] = vc_std_fp
                features['vc_expected_fp'] = vc_expected_fp
                # Calculate vc_fp_per_ball based on role
                if vc_role in ['BAT', 'WK']:
                    features['vc_fp_per_ball'] = self.safe_divide(vc_avg_fp, vc_balls + 0.1)
                elif vc_role == 'BOWL':
                    features['vc_fp_per_ball'] = self.safe_divide(vc_avg_fp, (vc_overs + 0.1) * 6 + 0.1)
                elif vc_role == 'AR':
                    features['vc_fp_per_ball'] = self.safe_divide(vc_avg_fp, vc_balls + (vc_overs + 0.1) * 6 + 0.1)
                else:
                    features['vc_fp_per_ball'] = 0.0
                
                # VC choice quality vs squad (similar to captain logic)
                # FIXED: Safe array combination to prevent broadcasting errors
                combined_squad_fp = []
                combined_squad_balls = []
                combined_squad_overs = []
                combined_squad_roles = []
                
                # Safely combine arrays with proper type checking
                if isinstance(bf_squad_fp, (list, np.ndarray)) and isinstance(ch_squad_fp, (list, np.ndarray)):
                    combined_squad_fp = list(bf_squad_fp) + list(ch_squad_fp)
                elif isinstance(bf_squad_fp, (list, np.ndarray)):
                    combined_squad_fp = list(bf_squad_fp)
                elif isinstance(ch_squad_fp, (list, np.ndarray)):
                    combined_squad_fp = list(ch_squad_fp)
                
                if isinstance(bf_squad_balls, (list, np.ndarray)) and isinstance(ch_squad_balls, (list, np.ndarray)):
                    combined_squad_balls = list(bf_squad_balls) + list(ch_squad_balls)
                elif isinstance(bf_squad_balls, (list, np.ndarray)):
                    combined_squad_balls = list(bf_squad_balls)
                elif isinstance(ch_squad_balls, (list, np.ndarray)):
                    combined_squad_balls = list(ch_squad_balls)
                
                if isinstance(bf_squad_overs, (list, np.ndarray)) and isinstance(ch_squad_overs, (list, np.ndarray)):
                    combined_squad_overs = list(bf_squad_overs) + list(ch_squad_overs)
                elif isinstance(bf_squad_overs, (list, np.ndarray)):
                    combined_squad_overs = list(bf_squad_overs)
                elif isinstance(ch_squad_overs, (list, np.ndarray)):
                    combined_squad_overs = list(ch_squad_overs)
                
                if isinstance(bf_squad_roles, (list, np.ndarray)) and isinstance(ch_squad_roles, (list, np.ndarray)):
                    combined_squad_roles = list(bf_squad_roles) + list(ch_squad_roles)
                elif isinstance(bf_squad_roles, (list, np.ndarray)):
                    combined_squad_roles = list(bf_squad_roles)
                elif isinstance(ch_squad_roles, (list, np.ndarray)):
                    combined_squad_roles = list(ch_squad_roles)
                
                if combined_squad_fp:
                    # Calculate expected FP for all squad players
                    squad_expected_fps = []
                    for i in range(len(combined_squad_fp)):
                        squad_data = {
                            'role': combined_squad_roles[i] if i < len(combined_squad_roles) else 'BAT',
                            'avg_fantasy_points_last5': combined_squad_fp[i],
                            'avg_balls_faced_last5': combined_squad_balls[i] if i < len(combined_squad_balls) else 0.0,
                            'avg_overs_bowled_last5': combined_squad_overs[i] if i < len(combined_squad_overs) else 0.0
                        }
                        squad_expected_fps.append(self.calculate_role_aware_expected_fp(squad_data))
                    
                    # VC ranking in squad
                    sorted_squad_fps = sorted(squad_expected_fps, reverse=True)
                    vc_rank = len([x for x in sorted_squad_fps if x > vc_expected_fp]) + 1
                    features['vc_rank_in_squad'] = vc_rank / len(sorted_squad_fps) if sorted_squad_fps else 1.0
                    
                    # VC gap from best in squad
                    best_squad_fp = max(squad_expected_fps) if squad_expected_fps else vc_expected_fp
                    features['vc_gap_from_best'] = 1.5 * (best_squad_fp - vc_expected_fp)  # 1.5x multiplier impact
                    
                    # VC boost efficiency
                    features['vc_boost_ratio'] = self.safe_divide(1.5 * vc_expected_fp, sum(squad_expected_fps))
                else:
                    features['vc_rank_in_squad'] = 0.5
                    features['vc_gap_from_best'] = 0.0
                    features['vc_boost_ratio'] = 0.0
                    
            else:
                # VC not found in team - use defaults
                features.update(self._get_default_vc_features())
                
        except Exception as e:
            print(f"⚠️ Error in VC feature extraction: {e}")
            features.update(self._get_default_vc_features())
        
        return features
    
    def _extract_combined_cvc_features(self, captain_id, vc_id, player_ids, roles, 
                                      avg_fp_array, avg_balls_array, avg_overs_array) -> Dict[str, float]:
        """Extract combined C/VC impact and interaction features"""
        features = {}
        
        try:
            # Find captain and VC indices
            cap_idx = -1
            vc_idx = -1
            for i, pid in enumerate(player_ids):
                if str(pid) == str(captain_id):
                    cap_idx = i
                elif str(pid) == str(vc_id):
                    vc_idx = i
            
            if cap_idx >= 0 and vc_idx >= 0:
                # Calculate expected FPs for both
                cap_data = {
                    'role': roles[cap_idx] if cap_idx < len(roles) else 'BAT',
                    'avg_fantasy_points_last5': avg_fp_array[cap_idx] if cap_idx < len(avg_fp_array) else 0.0,
                    'avg_balls_faced_last5': avg_balls_array[cap_idx] if cap_idx < len(avg_balls_array) else 0.0,
                    'avg_overs_bowled_last5': avg_overs_array[cap_idx] if cap_idx < len(avg_overs_array) else 0.0
                }
                cap_expected_fp = self.calculate_role_aware_expected_fp(cap_data)
                
                vc_data = {
                    'role': roles[vc_idx] if vc_idx < len(roles) else 'BAT',
                    'avg_fantasy_points_last5': avg_fp_array[vc_idx] if vc_idx < len(avg_fp_array) else 0.0,
                    'avg_balls_faced_last5': avg_balls_array[vc_idx] if vc_idx < len(avg_balls_array) else 0.0,
                    'avg_overs_bowled_last5': avg_overs_array[vc_idx] if vc_idx < len(avg_overs_array) else 0.0
                }
                vc_expected_fp = self.calculate_role_aware_expected_fp(vc_data)
                
                # Calculate team total expected FP
                team_total_expected = 0.0
                for i in range(len(avg_fp_array)):
                    player_data = {
                        'role': roles[i] if i < len(roles) else 'BAT',
                        'avg_fantasy_points_last5': avg_fp_array[i],
                        'avg_balls_faced_last5': avg_balls_array[i] if i < len(avg_balls_array) else 0.0,
                        'avg_overs_bowled_last5': avg_overs_array[i] if i < len(avg_overs_array) else 0.0
                    }
                    team_total_expected += self.calculate_role_aware_expected_fp(player_data)
                
                # Combined C/VC impact features
                cvc_boost_total = 2.0 * cap_expected_fp + 1.5 * vc_expected_fp
                features['cvc_boost_ratio'] = self.safe_divide(cvc_boost_total, team_total_expected + cvc_boost_total)
                features['cvc_ceiling_boost'] = cvc_boost_total  # Absolute boost
                features['cvc_boost_dominance'] = self.safe_divide(cvc_boost_total, team_total_expected)
                
                # C/VC role and structural features
                cap_role = roles[cap_idx] if cap_idx < len(roles) else 'BAT'
                vc_role = roles[vc_idx] if vc_idx < len(roles) else 'BAT'
                features['cvc_same_role'] = float(cap_role == vc_role)
                features['cvc_role_diversity'] = float(cap_role != vc_role)
                
                # Role combination encoding (for one-hot later)
                role_combo = f"{cap_role}_{vc_role}"
                combo_encoding = {
                    'BAT_BAT': 1, 'BAT_AR': 2, 'BAT_BOWL': 3, 'BAT_WK': 4,
                    'AR_BAT': 5, 'AR_AR': 6, 'AR_BOWL': 7, 'AR_WK': 8,
                    'BOWL_BAT': 9, 'BOWL_AR': 10, 'BOWL_BOWL': 11, 'BOWL_WK': 12,
                    'WK_BAT': 13, 'WK_AR': 14, 'WK_BOWL': 15, 'WK_WK': 16
                }
                features['cvc_role_combo_encoded'] = combo_encoding.get(role_combo, 0)
                
                # Ownership features removed - we don't want these
                
            else:
                features.update(self._get_default_combined_cvc_features())
                
        except Exception as e:
            print(f"⚠️ Error in combined C/VC feature extraction: {e}")
            features.update(self._get_default_combined_cvc_features())
        
        return features
    
    def _extract_pitch_features(self, pitch_descriptors: List[float]) -> Dict[str, float]:
        """
        Extract pitch condition features from 26-descriptor encoding
        Maps to batting-friendly, bowling-friendly, and neutral conditions
        """
        features = {}
        
        try:
            # Ensure we have 26 descriptors
            if len(pitch_descriptors) < 26:
                pitch_descriptors.extend([0.0] * (26 - len(pitch_descriptors)))
            
            # Group descriptors by category
            grass_moisture = sum(pitch_descriptors[0:7])    # green, grassy, damp, wet, moist, sticky, sticky_wicket
            dry_cracked = sum(pitch_descriptors[7:12])      # dry, dusty, cracks, crumbling, rough
            batting_friendly = sum(pitch_descriptors[12:19])  # flat, hard, true, even_bounce, dead, batting_paradise, road
            pace_related = sum(pitch_descriptors[19:23])    # live, lively, seam, swing
            spin_related = sum(pitch_descriptors[23:26])    # turn, turning_track, variable_bounce
            
            # Pitch category features
            features['pitch_grass_moisture_score'] = grass_moisture
            features['pitch_dry_cracked_score'] = dry_cracked
            features['pitch_batting_friendly_score'] = batting_friendly
            features['pitch_pace_bowling_score'] = pace_related
            features['pitch_spin_bowling_score'] = spin_related
            
            # Derived pitch insights
            features['pitch_total_descriptors'] = sum(pitch_descriptors)
            features['pitch_batting_advantage'] = batting_friendly - (pace_related + spin_related)
            features['pitch_bowling_advantage'] = (pace_related + spin_related) - batting_friendly
            features['pitch_moisture_vs_dry'] = grass_moisture - dry_cracked
            features['pitch_pace_vs_spin'] = pace_related - spin_related
            
            # Pitch condition flags
            features['pitch_is_batting_paradise'] = float(batting_friendly > 2.0)
            features['pitch_is_bowling_friendly'] = float((pace_related + spin_related) > 2.0)
            features['pitch_is_neutral'] = float(abs(features['pitch_batting_advantage']) < 1.0)
            
        except Exception as e:
            print(f"⚠️ Error in pitch feature extraction: {e}")
            # Default pitch features (neutral conditions)
            pitch_feature_names = [
                'pitch_grass_moisture_score', 'pitch_dry_cracked_score', 'pitch_batting_friendly_score',
                'pitch_pace_bowling_score', 'pitch_spin_bowling_score', 'pitch_total_descriptors',
                'pitch_batting_advantage', 'pitch_bowling_advantage', 'pitch_moisture_vs_dry',
                'pitch_pace_vs_spin', 'pitch_is_batting_paradise', 'pitch_is_bowling_friendly', 'pitch_is_neutral'
            ]
            features = {name: 0.0 for name in pitch_feature_names}
            features['pitch_is_neutral'] = 1.0  # Default to neutral
        
        return features
    
    def _extract_field_bias_features(self, ownership_array, cown_array, vcown_array, 
                                    captain_id, vc_id, player_ids) -> Dict[str, float]:
        """
        Extract field bias and contrarian analysis using logarithmic field expectation
        CRITICAL: This is where elite teams diverge from field psychology
        """
        features = {}
        
        try:
            # Find captain and VC indices
            cap_idx = -1
            vc_idx = -1
            for i, pid in enumerate(player_ids):
                if str(pid) == str(captain_id):
                    cap_idx = i
                elif str(pid) == str(vc_id):
                    vc_idx = i
            
            if cap_idx >= 0 and vc_idx >= 0:
                # Raw ownership values
                cap_ownership = ownership_array[cap_idx] if cap_idx < len(ownership_array) else 0.5
                cap_cown = cown_array[cap_idx] if cap_idx < len(cown_array) else 0.09
                vc_ownership = ownership_array[vc_idx] if vc_idx < len(ownership_array) else 0.5
                vc_vcown = vcown_array[vc_idx] if vc_idx < len(vcown_array) else 0.09
                
                # === LOGARITHMIC FIELD EXPECTATION ANALYSIS ===
                # Log-scaled ownership indicates field strength of belief
                features['captain_log_ownership'] = self.safe_log(cap_ownership, math.e, -1.0)
                features['captain_log_cown'] = self.safe_log(cap_cown, math.e, -2.5)  # log(0.09) ≈ -2.4
                features['vc_log_ownership'] = self.safe_log(vc_ownership, math.e, -1.0)
                features['vc_log_vcown'] = self.safe_log(vc_vcown, math.e, -2.5)
                
                # Contrarian opportunity signals (lower = more contrarian)
                features['captain_contrarian_score'] = -features['captain_log_cown']  # Higher when cown is low
                features['vc_contrarian_score'] = -features['vc_log_vcown']
                
                # Field consensus strength
                features['captain_field_consensus'] = cap_cown * cap_ownership  # High when both high
                features['vc_field_consensus'] = vc_vcown * vc_ownership
                features['cvc_combined_field_consensus'] = features['captain_field_consensus'] + features['vc_field_consensus']
                
                # Elite vs field divergence signals
                # Elite teams often pick players with high expected_fp but low field ownership
                team_avg_ownership = np.mean(ownership_array) if ownership_array else 0.5
                features['captain_ownership_vs_field'] = cap_ownership - team_avg_ownership
                features['vc_ownership_vs_field'] = vc_ownership - team_avg_ownership
                
                # Multiplier-weighted field sentiment
                total_multiplier_ownership = 2.0 * cap_cown + 1.5 * vc_vcown
                features['cvc_multiplier_field_weight'] = total_multiplier_ownership
                features['cvc_anti_field_score'] = self.safe_divide(1.0, total_multiplier_ownership, 5.0)
                
                # Cross-ownership analysis (ownership vs captain ownership patterns)
                features['captain_ownership_cown_ratio'] = self.safe_divide(cap_ownership, cap_cown, 1.0)
                features['vc_ownership_vcown_ratio'] = self.safe_divide(vc_ownership, vc_vcown, 1.0)
                
                # Field psychology indicators  
                features['captain_underowned_but_captained'] = float(cap_ownership < 0.3 and cap_cown > 0.15)
                features['vc_underowned_but_vc'] = float(vc_ownership < 0.3 and vc_vcown > 0.15)
                features['cvc_field_mispricing_score'] = features['captain_underowned_but_captained'] + features['vc_underowned_but_vc']
                
            else:
                features.update(self._get_default_field_bias_features())
                
        except Exception as e:
            print(f"⚠️ Error in field bias feature extraction: {e}")
            features.update(self._get_default_field_bias_features())
        
        return features
    
    def _get_default_cvc_features(self) -> Dict[str, float]:
        """Return default feature set when extraction fails"""
        defaults = {}
        defaults.update(self._get_default_captain_features())
        defaults.update(self._get_default_vc_features())
        defaults.update(self._get_default_combined_cvc_features())
        defaults.update(self._get_default_pitch_features())
        defaults.update(self._get_default_field_bias_features())
        return defaults
    
    def _get_default_captain_features(self) -> Dict[str, float]:
        return {
            'captain_avg_fp': 0.0, 'captain_max_fp': 0.0, 'captain_std_fp': 0.0,
            'captain_expected_fp': 0.0, 'captain_fp_per_ball': 0.0, 'captain_fp_per_over': 0.0,
            'captain_ownership': 0.5, 'captain_cown': 0.09,
            'captain_rank_in_squad': 0.5, 'captain_gap_from_best': 0.0, 'captain_boost_ratio': 0.0
        }
    
    def _get_default_vc_features(self) -> Dict[str, float]:
        return {
            'vc_avg_fp': 0.0, 'vc_max_fp': 0.0, 'vc_std_fp': 0.0,
            'vc_expected_fp': 0.0, 'vc_fp_per_ball': 0.0, 'vc_fp_per_over': 0.0,
            'vc_ownership': 0.5, 'vc_vcown': 0.09,
            'vc_rank_in_squad': 0.5, 'vc_gap_from_best': 0.0, 'vc_boost_ratio': 0.0
        }
    
    def _get_default_combined_cvc_features(self) -> Dict[str, float]:
        return {
            'cvc_boost_ratio': 0.0, 'cvc_ceiling_boost': 0.0, 'cvc_boost_dominance': 0.0,
            'cvc_same_role': 0.0, 'cvc_role_diversity': 0.0, 'cvc_role_combo_encoded': 0,
            'cvc_weighted_ownership': 0.27, 'cvc_uniqueness_score': 3.7
        }
    
    def _get_default_pitch_features(self) -> Dict[str, float]:
        return {
            'pitch_grass_moisture_score': 0.0, 'pitch_dry_cracked_score': 0.0, 'pitch_batting_friendly_score': 0.0,
            'pitch_pace_bowling_score': 0.0, 'pitch_spin_bowling_score': 0.0, 'pitch_total_descriptors': 0.0,
            'pitch_batting_advantage': 0.0, 'pitch_bowling_advantage': 0.0, 'pitch_moisture_vs_dry': 0.0,
            'pitch_pace_vs_spin': 0.0, 'pitch_is_batting_paradise': 0.0, 'pitch_is_bowling_friendly': 0.0, 'pitch_is_neutral': 1.0
        }
    
    def _get_default_field_bias_features(self) -> Dict[str, float]:
        return {
            'captain_log_ownership': -1.0, 'captain_log_cown': -2.5, 'vc_log_ownership': -1.0, 'vc_log_vcown': -2.5,
            'captain_contrarian_score': 2.5, 'vc_contrarian_score': 2.5, 'captain_field_consensus': 0.045,
            'vc_field_consensus': 0.045, 'cvc_combined_field_consensus': 0.09, 'captain_ownership_vs_field': 0.0,
            'vc_ownership_vs_field': 0.0, 'cvc_multiplier_field_weight': 0.315, 'cvc_anti_field_score': 3.17,
            'captain_ownership_cown_ratio': 5.56, 'vc_ownership_vcown_ratio': 5.56, 'captain_underowned_but_captained': 0.0,
            'vc_underowned_but_vc': 0.0, 'cvc_field_mispricing_score': 0.0
        }

# Example usage and integration
if __name__ == "__main__":
    extractor = EnhancedCVCFeatureExtractor()
    print("🎯 Enhanced C/VC Feature Extractor ready for integration")
    print("Features count by category:")
    print("  Captain: 11 features")
    print("  Vice-Captain: 11 features") 
    print("  Combined C/VC: 8 features")
    print("  Pitch Conditions: 13 features")
    print("  Field Bias Analysis: 18 features")
    print("  TOTAL: 61 new C/VC-related features") 