#!/usr/bin/env python3
"""
STRATEGIC TEAM FEATURES EXTRACTOR
Implementation of the 6 transformative features discussed in the conversation:

1. Team Avg Score - Sum of avg FP with C/VC multipliers (2x, 1.5x)
2. Team Max Score - Sum of max FP with C/VC multipliers (the "ceiling")
3. CVC Contribution Avg - % of team score from C/VC on average day
4. CVC Contribution Max - % of team score from C/VC on best day
5. Innings Balance Avg - How well team scores in both innings (avg)
6. Innings Balance Max - How well team scores in both innings (max)

These features give the model the foundational understanding it was missing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class StrategicTeamFeaturesExtractor:
    """
    STRATEGIC TEAM FEATURES EXTRACTOR
    
    Implements the 6 game-changing features identified in the conversation:
    - Direct team score prediction (avg and max)
    - C/VC contribution balance analysis  
    - Innings balance for robust team construction
    
    These features transform abstract player stats into concrete team performance metrics.
    """
    
    def __init__(self):
        """Initialize the strategic features extractor"""
        logger.info("🎯 Strategic Team Features Extractor initialized")
        logger.info("📊 Target: 6 transformative features for elite team discovery")
        logger.info("🔍 Focus: Team scores, C/VC balance, innings distribution")
        
    def safe_array_extract(self, array_data, default_value=0.0, target_length=11):
        """Safely extract numerical values from arrays with robust error handling"""
        try:
            if array_data is None:
                return [default_value] * target_length
            
            # Convert to list if needed
            if hasattr(array_data, 'tolist'):
                array_data = array_data.tolist()
            elif isinstance(array_data, str):
                import ast
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
        """Safe division with NaN and zero handling"""
        try:
            if denominator is None or np.isnan(denominator) or denominator == 0:
                return default
            if numerator is None or np.isnan(numerator):
                return default
            
            result = float(numerator) / float(denominator)
            return result if not np.isnan(result) else default
        except:
            return default
    
    def get_player_max_fp(self, last10_scores, player_idx):
        """
        Get maximum fantasy points for a specific player from their last 10 scores
        Handles both list and numpy array data types
        """
        try:
            if player_idx >= len(last10_scores):
                return 0.0
            
            player_scores_raw = last10_scores[player_idx]
            if isinstance(player_scores_raw, (list, np.ndarray)) and len(player_scores_raw) > 0:
                player_scores = np.array(player_scores_raw).flatten()
                player_scores = player_scores[~np.isnan(player_scores)]  # Remove NaN values
                if len(player_scores) > 0:
                    return float(np.max(player_scores))
            
            return 0.0
        except:
            return 0.0
    
    def extract_strategic_features(self, team_data: pd.Series) -> Dict[str, float]:
        """
        Extract all 6 strategic team features
        
        Returns the foundational features that were missing from the model:
        - Team performance prediction (avg/max)
        - C/VC balance analysis
        - Innings scoring distribution
        """
        features = {}
        
        try:
            # === DATA EXTRACTION ===
            captain_id = team_data.get('captain_id', '')
            vc_id = team_data.get('vice_captain_id', '')
            
            # Player data arrays
            player_ids_raw = team_data.get('player_ids', [])
            if hasattr(player_ids_raw, 'tolist'):
                player_ids = player_ids_raw.tolist()
            else:
                player_ids = list(player_ids_raw)
            
            roles_raw = team_data.get('roles', [])
            if hasattr(roles_raw, 'tolist'):
                roles = roles_raw.tolist()
            else:
                roles = list(roles_raw)
            
            # Statistical arrays
            avg_fp_array = self.safe_array_extract(team_data.get('avg_fantasy_points_last5_array', []))
            avg_balls_array = self.safe_array_extract(team_data.get('avg_balls_faced_last5_array', []))
            avg_overs_array = self.safe_array_extract(team_data.get('avg_overs_bowled_last5_array', []))
            last10_scores = team_data.get('last10_fantasy_scores_array', [])
            
            # Team context for innings balance
            toss_decision = team_data.get('toss_decision', 'bat')
            
            # === FIND CAPTAIN AND VC INDICES ===
            captain_idx = -1
            vc_idx = -1
            for i, pid in enumerate(player_ids):
                if str(pid) == str(captain_id):
                    captain_idx = i
                elif str(pid) == str(vc_id):
                    vc_idx = i
            
            # === FEATURE 1 & 2: TEAM AVG SCORE & TEAM MAX SCORE ===
            team_avg_score, team_max_score = self._calculate_team_scores(
                avg_fp_array, last10_scores, captain_idx, vc_idx
            )
            
            features['team_avg_score'] = team_avg_score
            features['team_max_score'] = team_max_score
            
            # === FEATURE 3 & 4: CVC CONTRIBUTION AVG & MAX ===
            cvc_contrib_avg, cvc_contrib_max = self._calculate_cvc_contribution(
                avg_fp_array, last10_scores, captain_idx, vc_idx, team_avg_score, team_max_score
            )
            
            features['cvc_contribution_avg'] = cvc_contrib_avg
            features['cvc_contribution_max'] = cvc_contrib_max
            
            # === FEATURE 5 & 6: INNINGS BALANCE AVG & MAX ===
            innings_balance_avg, innings_balance_max = self._calculate_innings_balance(
                player_ids, roles, avg_fp_array, last10_scores, avg_balls_array, avg_overs_array,
                captain_idx, vc_idx, team_data
            )
            
            features['innings_balance_avg'] = innings_balance_avg
            features['innings_balance_max'] = innings_balance_max
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error in strategic feature extraction: {e}")
            return self._get_default_strategic_features()
    
    def _calculate_team_scores(self, avg_fp_array, last10_scores, captain_idx, vc_idx):
        """
        Calculate Team Avg Score and Team Max Score
        
        Team Avg Score = sum of avg FP with C/VC multipliers (2x captain, 1.5x VC)
        Team Max Score = sum of max FP with C/VC multipliers (the ceiling)
        """
        try:
            # Calculate base team scores (sum of all 11 players)
            team_avg_base = sum(avg_fp_array)
            
            # Calculate team max score (sum of each player's maximum from last 10)
            team_max_base = 0.0
            for i in range(len(avg_fp_array)):
                player_max = self.get_player_max_fp(last10_scores, i)
                team_max_base += player_max
            
            # Add C/VC multiplier bonuses
            captain_avg_bonus = 0.0
            captain_max_bonus = 0.0
            vc_avg_bonus = 0.0
            vc_max_bonus = 0.0
            
            if captain_idx >= 0 and captain_idx < len(avg_fp_array):
                captain_avg_fp = avg_fp_array[captain_idx]
                captain_max_fp = self.get_player_max_fp(last10_scores, captain_idx)
                captain_avg_bonus = captain_avg_fp * 1.0  # 2x total - 1x already counted = 1x bonus
                captain_max_bonus = captain_max_fp * 1.0
            
            if vc_idx >= 0 and vc_idx < len(avg_fp_array):
                vc_avg_fp = avg_fp_array[vc_idx]
                vc_max_fp = self.get_player_max_fp(last10_scores, vc_idx)
                vc_avg_bonus = vc_avg_fp * 0.5  # 1.5x total - 1x already counted = 0.5x bonus
                vc_max_bonus = vc_max_fp * 0.5
            
            # Final team scores
            team_avg_score = team_avg_base + captain_avg_bonus + vc_avg_bonus
            team_max_score = team_max_base + captain_max_bonus + vc_max_bonus
            
            return team_avg_score, team_max_score
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating team scores: {e}")
            return 0.0, 0.0
    
    def _calculate_cvc_contribution(self, avg_fp_array, last10_scores, captain_idx, vc_idx, 
                                   team_avg_score, team_max_score):
        """
        Calculate CVC Contribution Avg and Max
        
        Measures what percentage of team's total score comes from C/VC multipliers
        Ideal range: 45-60% according to the conversation
        """
        try:
            # Calculate C/VC scores with multipliers
            captain_avg_total = 0.0
            captain_max_total = 0.0
            vc_avg_total = 0.0
            vc_max_total = 0.0
            
            if captain_idx >= 0 and captain_idx < len(avg_fp_array):
                captain_avg_fp = avg_fp_array[captain_idx]
                captain_max_fp = self.get_player_max_fp(last10_scores, captain_idx)
                captain_avg_total = captain_avg_fp * 2.0  # Full 2x multiplier
                captain_max_total = captain_max_fp * 2.0
            
            if vc_idx >= 0 and vc_idx < len(avg_fp_array):
                vc_avg_fp = avg_fp_array[vc_idx]
                vc_max_fp = self.get_player_max_fp(last10_scores, vc_idx)
                vc_avg_total = vc_avg_fp * 1.5  # Full 1.5x multiplier
                vc_max_total = vc_max_fp * 1.5
            
            # Calculate contribution percentages
            cvc_avg_score = captain_avg_total + vc_avg_total
            cvc_max_score = captain_max_total + vc_max_total
            
            cvc_contrib_avg = self.safe_divide(cvc_avg_score, team_avg_score, 0.0)
            cvc_contrib_max = self.safe_divide(cvc_max_score, team_max_score, 0.0)
            
            return cvc_contrib_avg, cvc_contrib_max
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating CVC contribution: {e}")
            return 0.0, 0.0
    
    def _calculate_innings_balance(self, player_ids, roles, avg_fp_array, last10_scores,
                                  avg_balls_array, avg_overs_array, captain_idx, vc_idx, team_data):
        """
        Calculate Innings Balance Avg and Max
        
        Measures how well the team scores in both innings (1st and 2nd)
        Uses data-driven AR contribution ratios based on avg balls faced vs avg overs bowled
        """
        try:
            # Determine team affiliations (simplified - assume half from each squad)
            # In practice, this would use actual squad data from team_data
            batfirst_team = "Squad1"  # This would be extracted from actual data
            chase_team = "Squad2"     # This would be extracted from actual data
            
            innings1_avg_score = 0.0
            innings1_max_score = 0.0
            innings2_avg_score = 0.0
            innings2_max_score = 0.0
            
            for i in range(len(player_ids)):
                if i >= len(roles) or i >= len(avg_fp_array):
                    continue
                
                role = roles[i]
                player_avg_fp = avg_fp_array[i]
                player_max_fp = self.get_player_max_fp(last10_scores, i)
                
                # Apply multipliers if this is captain or VC
                multiplier = 1.0
                if i == captain_idx:
                    multiplier = 2.0
                elif i == vc_idx:
                    multiplier = 1.5
                
                adjusted_avg_fp = player_avg_fp * multiplier
                adjusted_max_fp = player_max_fp * multiplier
                
                # Determine team affiliation (simplified approach)
                # In practice, this would use actual squad membership data
                is_batfirst_team = (i < 6)  # Simplified assumption
                
                # Distribute scores based on role and team
                if role in ['BAT', 'WK']:
                    if is_batfirst_team:
                        # Batfirst batsmen score in 1st innings
                        innings1_avg_score += adjusted_avg_fp
                        innings1_max_score += adjusted_max_fp
                    else:
                        # Chase batsmen score in 2nd innings
                        innings2_avg_score += adjusted_avg_fp
                        innings2_max_score += adjusted_max_fp
                        
                elif role == 'BOWL':
                    if is_batfirst_team:
                        # Batfirst bowlers score in 2nd innings
                        innings2_avg_score += adjusted_avg_fp
                        innings2_max_score += adjusted_max_fp
                    else:
                        # Chase bowlers score in 1st innings
                        innings1_avg_score += adjusted_avg_fp
                        innings1_max_score += adjusted_max_fp
                        
                elif role == 'AR':
                    # All-rounders: data-driven split based on usage
                    avg_balls = avg_balls_array[i] if i < len(avg_balls_array) else 0.0
                    avg_overs = avg_overs_array[i] if i < len(avg_overs_array) else 0.0
                    avg_balls_bowled = avg_overs * 6.0
                    
                    total_opportunity = avg_balls + avg_balls_bowled
                    
                    if total_opportunity > 0:
                        batting_ratio = avg_balls / total_opportunity
                        bowling_ratio = avg_balls_bowled / total_opportunity
                    else:
                        # Fallback to 50/50 split
                        batting_ratio = 0.5
                        bowling_ratio = 0.5
                    
                    batting_avg_points = adjusted_avg_fp * batting_ratio
                    batting_max_points = adjusted_max_fp * batting_ratio
                    bowling_avg_points = adjusted_avg_fp * bowling_ratio
                    bowling_max_points = adjusted_max_fp * bowling_ratio
                    
                    if is_batfirst_team:
                        # Batting in 1st innings, bowling in 2nd innings
                        innings1_avg_score += batting_avg_points
                        innings1_max_score += batting_max_points
                        innings2_avg_score += bowling_avg_points
                        innings2_max_score += bowling_max_points
                    else:
                        # Bowling in 1st innings, batting in 2nd innings
                        innings1_avg_score += bowling_avg_points
                        innings1_max_score += bowling_max_points
                        innings2_avg_score += batting_avg_points
                        innings2_max_score += batting_max_points
            
            # Calculate balance ratios (min/total gives balance score)
            total_avg_score = innings1_avg_score + innings2_avg_score
            total_max_score = innings1_max_score + innings2_max_score
            
            innings_balance_avg = self.safe_divide(
                min(innings1_avg_score, innings2_avg_score), 
                total_avg_score + 1e-6, 
                0.0
            )
            
            innings_balance_max = self.safe_divide(
                min(innings1_max_score, innings2_max_score), 
                total_max_score + 1e-6, 
                0.0
            )
            
            return innings_balance_avg, innings_balance_max
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating innings balance: {e}")
            return 0.0, 0.0
    
    def _get_default_strategic_features(self) -> Dict[str, float]:
        """Return default feature values when extraction fails"""
        return {
            'team_avg_score': 0.0,
            'team_max_score': 0.0,
            'cvc_contribution_avg': 0.0,
            'cvc_contribution_max': 0.0,
            'innings_balance_avg': 0.0,
            'innings_balance_max': 0.0
        }

# Example usage
if __name__ == "__main__":
    extractor = StrategicTeamFeaturesExtractor()
    print("🎯 Strategic Team Features Extractor ready for integration")
    print("Features implemented:")
    print("  1. Team Avg Score - Expected team performance")
    print("  2. Team Max Score - Team ceiling potential") 
    print("  3. CVC Contribution Avg - Captain/VC balance (avg)")
    print("  4. CVC Contribution Max - Captain/VC balance (max)")
    print("  5. Innings Balance Avg - Scoring distribution (avg)")
    print("  6. Innings Balance Max - Scoring distribution (max)")
    print("  TOTAL: 6 transformative strategic features")



