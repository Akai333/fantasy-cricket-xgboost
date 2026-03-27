#!/usr/bin/env python3
"""
Real Model Interface for MGAG
Loads actual trained XGBoost models and applies them to teams
"""

import json
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import feature extractors  
from R1_global_ranker_feature_extractor import R1GlobalRankerFeatureExtractor


class RealModelInterface:
    """Interface to load and use actual trained XGBoost models"""
    
    def __init__(self, models_base_path: str = "Trained model"):
        """
        Initialize model interface with paths to trained models
        
        Args:
            models_base_path: Base path to trained models directory
        """
        self.models_base_path = Path(models_base_path)
        
        # Model storage
        self.stage1_model = None
        self.stage2_model = None
        self.stage1_metadata = None
        self.stage2_metadata = None
        
        # Feature extractor (the actual one used for training)
        self.feature_extractor = R1GlobalRankerFeatureExtractor()
        
        # Performance tracking
        self.stage1_call_count = 0
        self.stage2_call_count = 0
        self.total_inference_time = 0
        self.feature_extraction_time = 0
        
    def load_models(self, league: str = "cpl"):
        """
        Load Stage 1 Filter and Stage 2 league-specific ranker
        
        Args:
            league: League identifier (e.g., "cpl", "t20blast")
        """
        print(f"📊 Loading real trained models for {league.upper()}...")
        
        # Load Stage 1 Global Filter
        self._load_stage1_filter()
        
        # Load Stage 2 League-specific Ranker
        self._load_stage2_ranker(league)
        
        print(f"   ✅ Models loaded successfully")
    
    def _load_stage1_filter(self):
        """Load Stage 1 Global Filter model"""
        filter_dir = self.models_base_path / "Filter"
        
        # Find model files (latest timestamp)
        model_files = list(filter_dir.glob("S1_global_filter_model_*.json"))
        metadata_files = list(filter_dir.glob("S1_global_filter_metadata_*.json"))
        
        if not model_files or not metadata_files:
            raise FileNotFoundError(f"Stage 1 model files not found in {filter_dir}")
        
        # Use the latest file (sort by timestamp)
        model_file = sorted(model_files)[-1]
        metadata_file = sorted(metadata_files)[-1]
        
        print(f"   📁 Loading Stage 1 Filter: {model_file.name}")
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.stage1_metadata = json.load(f)
        
        # Load XGBoost model
        self.stage1_model = xgb.Booster()
        self.stage1_model.load_model(str(model_file))
        
        print(f"      AUC: {self.stage1_metadata['performance']['auc_score']:.3f}")
        print(f"      Optimal Threshold: {self.stage1_metadata['performance']['optimal_threshold']:.4f}")
        print(f"      Recall @ 90%: {self.stage1_metadata['performance']['optimal_recall']:.1%}")
    
    def _load_stage2_ranker(self, league: str):
        """Load Stage 2 league-specific ranker"""
        league_mapping = {
            "cpl": "CPL Ranker",
            "t20blast": "T20 Blast Ranker"
        }
        
        ranker_dir_name = league_mapping.get(league.lower(), "CPL Ranker")
        ranker_dir = self.models_base_path / ranker_dir_name
        
        if not ranker_dir.exists():
            print(f"   ⚠️  {ranker_dir_name} not found, falling back to CPL Ranker")
            ranker_dir = self.models_base_path / "CPL Ranker"
        
        # Find model files
        model_files = list(ranker_dir.glob("R1_*_ranker_model_*.json"))
        metadata_files = list(ranker_dir.glob("R1_*_ranker_metadata_*.json"))
        
        if not model_files or not metadata_files:
            raise FileNotFoundError(f"Stage 2 model files not found in {ranker_dir}")
        
        # Use the latest file
        model_file = sorted(model_files)[-1]
        metadata_file = sorted(metadata_files)[-1]
        
        print(f"   📁 Loading Stage 2 Ranker: {model_file.name}")
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.stage2_metadata = json.load(f)
        
        # Load XGBoost model
        self.stage2_model = xgb.Booster()
        self.stage2_model.load_model(str(model_file))
        
        print(f"      Precision@10: {self.stage2_metadata['performance']['avg_precision_at_10']:.1%}")
        print(f"      NDCG@10: {self.stage2_metadata['performance']['avg_ndcg_at_10']:.3f}")
    
    def apply_stage1_filter(self, teams: List[Dict], 
                          squad_context: Dict,
                          filter_threshold: Optional[float] = None) -> List[Dict]:
        """
        Apply Stage 1 filter to teams using real model
        
        Args:
            teams: List of teams to filter
            squad_context: Squad and match context for feature extraction
            filter_threshold: Custom threshold (uses optimal if None)
            
        Returns:
            List of teams that pass Stage 1 filter
        """
        if not teams or not self.stage1_model:
            return []
        
        start_time = time.time()
        self.stage1_call_count += 1
        
        # Use optimal threshold from metadata if not provided
        if filter_threshold is None:
            filter_threshold = self.stage1_metadata['performance']['optimal_threshold']
        
        print(f"   🔍 Stage 1 Filter: Processing {len(teams):,} teams...")
        
        # Extract features for all teams
        feature_start = time.time()
        features_df = self._extract_features_batch(teams, squad_context)
        feature_time = time.time() - feature_start
        self.feature_extraction_time += feature_time
        
        print(f"      Features extracted: {features_df.shape[1]} features ({feature_time:.2f}s)")
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features_df)
        
        # Get predictions
        predictions = self.stage1_model.predict(dmatrix)
        
        # Filter teams based on threshold
        filtered_teams = []
        for i, (team, prediction) in enumerate(zip(teams, predictions)):
            if prediction >= filter_threshold:
                team['stage1_score'] = float(prediction)
                filtered_teams.append(team)
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        pass_rate = len(filtered_teams) / len(teams) * 100
        print(f"   🔍 Stage 1 Filter: {len(teams):,} → {len(filtered_teams):,} teams "
              f"({pass_rate:.1f}% pass rate, {inference_time:.1f}s)")
        
        return filtered_teams
    
    def apply_stage2_ranker(self, teams: List[Dict], 
                          squad_context: Dict) -> List[Dict]:
        """
        Apply Stage 2 ranker to teams using real model
        
        Args:
            teams: List of teams to rank
            squad_context: Squad and match context for feature extraction
            
        Returns:
            List of teams with P_elite scores, sorted by rank
        """
        if not teams or not self.stage2_model:
            return []
        
        start_time = time.time()
        self.stage2_call_count += 1
        
        print(f"   🎯 Stage 2 Ranker: Processing {len(teams):,} teams...")
        
        # Extract features for all teams (reuse if already extracted)
        feature_start = time.time()
        features_df = self._extract_features_batch(teams, squad_context)
        feature_time = time.time() - feature_start
        self.feature_extraction_time += feature_time
        
        print(f"      Features extracted: {features_df.shape[1]} features ({feature_time:.2f}s)")
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features_df)
        
        # Get ranking scores
        ranking_scores = self.stage2_model.predict(dmatrix)
        
        # Add scores to teams and sort
        scored_teams = []
        for team, score in zip(teams, ranking_scores):
            team['p_elite'] = float(score)
            team['stage2_score'] = float(score)
            scored_teams.append(team)
        
        # Sort by ranking score (higher is better for elite probability)
        scored_teams.sort(key=lambda x: x['p_elite'], reverse=True)
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        # Calculate stats
        avg_p_elite = np.mean([t['p_elite'] for t in scored_teams])
        high_confidence = sum(1 for t in scored_teams if t['p_elite'] >= 0.4)
        
        print(f"   🎯 Stage 2 Ranker: {len(scored_teams):,} teams ranked "
              f"(avg score: {avg_p_elite:.3f}, high confidence: {high_confidence}, {inference_time:.1f}s)")
        
        return scored_teams
    
    def _extract_features_batch(self, teams: List[Dict], 
                              squad_context: Dict) -> pd.DataFrame:
        """
        Extract features for a batch of teams
        
        Args:
            teams: List of teams
            squad_context: Squad and match context
            
        Returns:
            DataFrame with features for all teams
        """
        try:
            # Convert teams to DataFrame format expected by feature extractor
            teams_data = []
            
            for i, team in enumerate(teams):
                # Create team record matching the parquet schema
                team_record = {
                    # Core identifiers
                    'team_uuid': f"mgag_team_{i}",
                    'match_id': squad_context.get('match_id', 'mgag_match'),
                    
                    # Player selection
                    'player_ids': team.get('player_ids', []),
                    'captain_id': team.get('captain_id'),
                    'vice_captain_id': team.get('vice_captain_id'),
                    'roles': self._create_roles_array(team.get('player_ids', []), squad_context),
                    'team_ids': team.get('team_ids', []),
                    
                    # Match context
                    'venue': squad_context.get('match_context', {}).get('venue', ''),
                    'league': squad_context.get('match_context', {}).get('league', 'cpl'),
                    'batting_first_team': squad_context.get('match_context', {}).get('batting_first_team', ''),
                    'chasing_team': squad_context.get('match_context', {}).get('chasing_team', ''),
                    'toss_winner': squad_context.get('match_context', {}).get('toss_winner', ''),
                    'toss_decision': squad_context.get('match_context', {}).get('toss_decision', ''),
                    'gender': squad_context.get('match_context', {}).get('gender', 'male'),
                    'series': squad_context.get('match_context', {}).get('series', 'CPL 2024'),
                    'date': squad_context.get('match_context', {}).get('date', '2024-01-01'),
                    
                    # Squad data (flattened for feature extraction)
                    'batfirst_squad': self._flatten_squad_data(squad_context.get('batfirst_players', {})),
                    'chase_squad': self._flatten_squad_data(squad_context.get('chase_players', {})),
                    
                    # Default values for other fields
                    'contextual_template': team.get('contextual_template', '1-3-4-3'),
                    'ownership': [0.5] * 11,  # Default ownership
                    'cown': [0.09] * 11,      # Default captain ownership
                    'vcown': [0.09] * 11,     # Default VC ownership
                    
                    # Required array fields for feature extraction
                    'avg_fantasy_points_last5_array': self._create_player_stats_array(
                        team.get('player_ids', []), squad_context, 'avg_fantasy_points_last5'
                    ),
                    'avg_balls_faced_last5_array': self._create_player_stats_array(
                        team.get('player_ids', []), squad_context, 'avg_balls_faced_last5'
                    ),
                    'avg_overs_bowled_last5_array': self._create_player_stats_array(
                        team.get('player_ids', []), squad_context, 'avg_overs_bowled_last5'
                    ),
                    'last10_fantasy_scores_array': self._create_last10_scores_array(
                        team.get('player_ids', []), squad_context
                    ),
                    'batting_style_array': self._create_player_attribute_array(
                        team.get('player_ids', []), squad_context, 'batting_style'
                    ),
                    'bowling_style_array': self._create_player_attribute_array(
                        team.get('player_ids', []), squad_context, 'bowling_style'
                    ),
                    'bowling_phases_array': self._create_bowling_phases_array(
                        team.get('player_ids', []), squad_context
                    ),
                    'batting_order_array': list(range(1, 12)),  # Simple 1-11 order
                    
                    # Squad-level arrays
                    'batfirst_squad_avg_fantasy_points_last5': self._create_squad_stats_array(
                        squad_context.get('batfirst_players', {}), 'avg_fantasy_points_last5'
                    ),
                    'chase_squad_avg_fantasy_points_last5': self._create_squad_stats_array(
                        squad_context.get('chase_players', {}), 'avg_fantasy_points_last5'
                    )
                }
                teams_data.append(team_record)
            
            # Convert to DataFrame
            teams_df = pd.DataFrame(teams_data)
            
            # Extract features for each team using the R1 extractor
            all_features = []
            
            print(f"      Extracting features with R1GlobalRankerFeatureExtractor...")
            
            for idx, team_row in teams_df.iterrows():
                try:
                    features = self.feature_extractor.extract_all_features(team_row)
                    all_features.append(features)
                except Exception as e:
                    print(f"      Warning: Feature extraction failed for team {idx}: {e}")
                    # Create dummy features for this team
                    dummy_features = {f'feature_{i}': 0.0 for i in range(255)}
                    all_features.append(dummy_features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(all_features)
            
            # Handle feature count alignment
            expected_features = self.stage1_metadata.get('feature_count', 242)
            
            if features_df.shape[1] != expected_features:
                print(f"      ⚠️  Feature count mismatch: got {features_df.shape[1]}, "
                      f"expected {expected_features}")
                
                # Handle feature mismatch by padding or truncating
                if features_df.shape[1] < expected_features:
                    # Pad with zeros
                    missing_cols = expected_features - features_df.shape[1]
                    for i in range(missing_cols):
                        features_df[f'missing_feature_{i}'] = 0.0
                elif features_df.shape[1] > expected_features:
                    # Truncate to expected size
                    features_df = features_df.iloc[:, :expected_features]
            
            return features_df
            
        except Exception as e:
            print(f"      ❌ Feature extraction failed: {e}")
            import traceback
            print(f"      Traceback: {traceback.format_exc()}")
            
            # Create dummy features as fallback
            expected_features = self.stage1_metadata.get('feature_count', 242)
            dummy_features = pd.DataFrame(
                np.zeros((len(teams), expected_features)),
                columns=[f'feature_{i}' for i in range(expected_features)]
            )
            return dummy_features
    
    def _flatten_squad_data(self, squad_players: Dict) -> List[Dict]:
        """
        Flatten squad data to format expected by feature extractor
        
        Args:
            squad_players: Dictionary of player data
            
        Returns:
            List of player dictionaries
        """
        flattened = []
        
        for player_id, player_data in squad_players.items():
            player_record = {
                'player_id': player_id,
                'name': player_data.get('name', ''),
                'role': player_data.get('role', 'BAT'),
                'team': player_data.get('team', ''),
                'batting_style': player_data.get('batting_style', ''),
                'bowling_style': player_data.get('bowling_style', ''),
                'avg_fantasy_points_last5': player_data.get('avg_fantasy_points_last5', 0.0),
                'avg_balls_faced_last5': player_data.get('avg_balls_faced_last5', 0.0),
                'avg_overs_bowled_last5': player_data.get('avg_overs_bowled_last5', 0.0),
                'ownership': player_data.get('ownership', 0.5)
            }
            flattened.append(player_record)
        
        return flattened
    
    def _create_player_stats_array(self, player_ids: List[str], 
                                  squad_context: Dict, 
                                  stat_field: str) -> List[float]:
        """
        Create array of player statistics for a team
        
        Args:
            player_ids: List of player IDs in the team
            squad_context: Squad context with player data
            stat_field: Field to extract (e.g., 'avg_fantasy_points_last5')
            
        Returns:
            List of stat values for the team players
        """
        stats_array = []
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        for player_id in player_ids:
            if player_id in all_players:
                stat_value = all_players[player_id].get(stat_field, 0.0)
                stats_array.append(float(stat_value))
            else:
                stats_array.append(0.0)
        
        # Ensure exactly 11 values
        while len(stats_array) < 11:
            stats_array.append(0.0)
        
        return stats_array[:11]
    
    def _create_player_attribute_array(self, player_ids: List[str],
                                     squad_context: Dict,
                                     attribute_field: str) -> List[str]:
        """
        Create array of player attributes for a team
        
        Args:
            player_ids: List of player IDs in the team
            squad_context: Squad context with player data
            attribute_field: Field to extract (e.g., 'batting_style')
            
        Returns:
            List of attribute values for the team players
        """
        attr_array = []
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        for player_id in player_ids:
            if player_id in all_players:
                attr_value = all_players[player_id].get(attribute_field, '')
                attr_array.append(str(attr_value))
            else:
                attr_array.append('')
        
        # Ensure exactly 11 values
        while len(attr_array) < 11:
            attr_array.append('')
        
        return attr_array[:11]
    
    def _create_last10_scores_array(self, player_ids: List[str],
                                  squad_context: Dict) -> List[List[float]]:
        """
        Create array of last 10 fantasy scores for team players
        
        Args:
            player_ids: List of player IDs in the team
            squad_context: Squad context with player data
            
        Returns:
            List of lists containing last 10 scores for each player
        """
        scores_array = []
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        for player_id in player_ids:
            if player_id in all_players:
                # Create dummy last 10 scores based on avg_fantasy_points_last5
                avg_points = all_players[player_id].get('avg_fantasy_points_last5', 30.0)
                # Generate 10 scores with some variance around the average
                last10 = [
                    max(0, avg_points + np.random.normal(0, avg_points * 0.3))
                    for _ in range(10)
                ]
                scores_array.append(last10)
            else:
                scores_array.append([0.0] * 10)
        
        # Ensure exactly 11 player score arrays
        while len(scores_array) < 11:
            scores_array.append([0.0] * 10)
        
        return scores_array[:11]
    
    def _create_roles_array(self, player_ids: List[str], squad_context: Dict) -> List[str]:
        """
        Create roles array for team players
        
        Args:
            player_ids: List of player IDs in the team
            squad_context: Squad context with player data
            
        Returns:
            List of roles for each player
        """
        roles_array = []
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        for player_id in player_ids:
            if player_id in all_players:
                role = all_players[player_id].get('role', 'BAT')
                roles_array.append(role)
            else:
                roles_array.append('BAT')  # Default to BAT if player not found
        
        # Ensure exactly 11 player roles
        while len(roles_array) < 11:
            roles_array.append('BAT')
        
        return roles_array[:11]

    def _create_bowling_phases_array(self, player_ids: List[str],
                                   squad_context: Dict) -> List[List[float]]:
        """
        Create bowling phases array for team players
        
        Args:
            player_ids: List of player IDs in the team
            squad_context: Squad context with player data
            
        Returns:
            List of bowling phase arrays for each player
        """
        phases_array = []
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        for player_id in player_ids:
            if player_id in all_players:
                player_role = all_players[player_id].get('role', 'BAT')
                if player_role in ['BOWL', 'AR']:
                    # Create realistic bowling phases [powerplay, middle, death]
                    avg_overs = all_players[player_id].get('avg_overs_bowled_last5', 2.0)
                    phases = [avg_overs * 0.3, avg_overs * 0.4, avg_overs * 0.3]
                else:
                    phases = [0.0, 0.0, 0.0]
                phases_array.append(phases)
            else:
                phases_array.append([0.0, 0.0, 0.0])
        
        # Ensure exactly 11 player phase arrays
        while len(phases_array) < 11:
            phases_array.append([0.0, 0.0, 0.0])
        
        return phases_array[:11]
    
    def _create_squad_stats_array(self, squad_players: Dict, stat_field: str) -> List[float]:
        """
        Create array of statistics for all squad players
        
        Args:
            squad_players: Dictionary of squad player data
            stat_field: Field to extract
            
        Returns:
            List of stat values for squad players
        """
        stats_array = []
        
        for player_id, player_data in squad_players.items():
            stat_value = player_data.get(stat_field, 0.0)
            stats_array.append(float(stat_value))
        
        return stats_array
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        return {
            'stage1_calls': self.stage1_call_count,
            'stage2_calls': self.stage2_call_count,
            'total_inference_time': self.total_inference_time,
            'feature_extraction_time': self.feature_extraction_time,
            'avg_stage1_time': self.total_inference_time / max(1, self.stage1_call_count),
            'avg_stage2_time': self.total_inference_time / max(1, self.stage2_call_count),
            'stage1_metadata': self.stage1_metadata,
            'stage2_metadata': self.stage2_metadata
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'stage1_loaded': self.stage1_model is not None,
            'stage2_loaded': self.stage2_model is not None,
            'stage1_auc': self.stage1_metadata['performance']['auc_score'] if self.stage1_metadata else None,
            'stage2_precision_at_10': self.stage2_metadata['performance']['avg_precision_at_10'] if self.stage2_metadata else None,
            'stage1_threshold': self.stage1_metadata['performance']['optimal_threshold'] if self.stage1_metadata else None,
            'league_focus': self.stage2_metadata.get('league_focus', 'Unknown') if self.stage2_metadata else None
        }
