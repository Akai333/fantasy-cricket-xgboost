#!/usr/bin/env python3
"""
Refined Venue Interaction Feature Extractor

Based on expert analysis, this version keeps only the "gold standard" features
and removes potentially problematic ones that could introduce noise or target leakage.

KEPT (Gold Standard):
- Category 1: Player Form vs Venue Context (8 features) ✅
- Category 2: Bowling Style Venue Synergy (6 features) ✅  
- Selected Category 4: captain/vc_venue_leverage (2 features) ✅

REMOVED (Potentially Problematic):
- Category 3: Vague "fit" features ❌
- Category 4: "Potential" features (target leakage risk) ❌
- Category 5: Abstract "momentum/adaptability" features ❌

Total Refined Features: 16 high-quality venue interaction features
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RefinedVenueInteractionFeatureExtractor:
    """
    Refined venue interaction feature extractor focusing on high-quality,
    interpretable features with clear mathematical definitions.
    """
    
    def __init__(self, venue_data_dir: str = "./"):
        """
        Initialize the refined venue feature extractor.
        
        Args:
            venue_data_dir: Directory containing venue JSON files
        """
        self.venue_data_dir = Path(venue_data_dir)
        self.venue_data = {}
        self.spin_styles = {
            'Right arm Offbreak', 'Left arm Orthodox', 'Right arm Legbreak', 
            'Left arm Chinaman', 'Right arm Googly', 'Slow Left arm Orthodox'
        }
        self.pace_styles = {
            'Right arm Fast', 'Left arm Fast', 'Right arm Medium fast', 
            'Left arm Medium fast', 'Right arm Medium', 'Left arm Medium'
        }
        
        logger.info("🏟️  Refined Venue Interaction Feature Extractor initialized")
        logger.info(f"📂 Venue data directory: {venue_data_dir}")
        
        self._load_venue_data()
    
    def _load_venue_data(self):
        """Load venue context data from JSON files."""
        try:
            # Load Caribbean Premier League venue data
            venue_file = self.venue_data_dir / "caribbean_premier_league_venue.json"
            if venue_file.exists():
                with open(venue_file, 'r') as f:
                    cpl_data = json.load(f)
                    # Extract the venues section from the JSON
                    venues_data = cpl_data.get('venues', {})
                    self.venue_data['Caribbean Premier League'] = venues_data
                    logger.info(f"✅ Loaded venue data for Caribbean Premier League: {len(venues_data)} venues")
            else:
                logger.warning(f"⚠️ Venue file not found: {venue_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading venue data: {e}")
            self.venue_data = {}
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return float(numerator / denominator)
    
    def _get_venue_stats(self, venue: str, league: str = 'Caribbean Premier League') -> Dict[str, float]:
        """Get venue statistics for a specific venue and league."""
        if league not in self.venue_data:
            return {}
        
        league_venues = self.venue_data[league]
        for venue_name, stats in league_venues.items():
            if venue_name.lower() in venue.lower() or venue.lower() in venue_name.lower():
                return stats
        
        return {}
    
    def _is_spinner(self, bowling_style: str) -> bool:
        """Check if a bowling style is spin bowling."""
        return bowling_style in self.spin_styles
    
    def _is_pacer(self, bowling_style: str) -> bool:
        """Check if a bowling style is pace bowling.""" 
        return bowling_style in self.pace_styles
    
    def extract_refined_venue_features(self, team_data: pd.Series) -> Dict[str, float]:
        """
        Extract refined venue interaction features for a team.
        
        Args:
            team_data: Team data containing player arrays and venue information
            
        Returns:
            Dictionary of refined venue interaction features
        """
        features = {}
        
        try:
            # Get venue information
            venue = team_data.get('venue', '')
            venue_stats = self._get_venue_stats(venue)
            
            if not venue_stats:
                logger.warning(f"⚠️ No venue stats found for: {venue}")
                return self._get_default_features()
            
            # Extract player data arrays
            avg_fantasy_points = np.array(team_data.get('avg_fantasy_points_last5_array', []))
            bowling_styles = team_data.get('bowling_style_array', [])
            roles = np.array(team_data.get('role_array', []))
            captain_idx = team_data.get('captain_index', 0)
            vc_idx = team_data.get('vc_index', 1)
            
            # Validate data
            if len(avg_fantasy_points) == 0:
                return self._get_default_features()
            
            # === CATEGORY 1: PLAYER FORM VS VENUE CONTEXT (8 features) ===
            # These are the "gold standard" features - clear mathematical definitions
            
            features.update(self._extract_player_form_vs_venue_features(
                avg_fantasy_points, roles, venue_stats
            ))
            
            # === CATEGORY 2: BOWLING STYLE VENUE SYNERGY (6 features) ===
            # Perfect multiplicative interactions - high value only when both conditions met
            
            features.update(self._extract_bowling_style_synergy_features(
                avg_fantasy_points, bowling_styles, roles, venue_stats
            ))
            
            # === CATEGORY 4: STRATEGIC VENUE LEVERAGE (2 features) ===
            # Keep only captain/vc leverage - specific and measurable
            
            features.update(self._extract_strategic_leverage_features(
                avg_fantasy_points, captain_idx, vc_idx, venue_stats
            ))
            
        except Exception as e:
            logger.error(f"❌ Error extracting venue features: {e}")
            return self._get_default_features()
        
        return features
    
    def _extract_player_form_vs_venue_features(self, fantasy_points: np.ndarray, 
                                             roles: np.ndarray, venue_stats: Dict) -> Dict[str, float]:
        """
        Extract Player Form vs Venue Context features.
        
        These use subtraction (player_form - venue_average) to create 
        "value over baseline" metrics. The model sees player form relative 
        to historical venue difficulty.
        """
        features = {}
        
        # Venue baselines
        venue_batting_avg = venue_stats.get('avg_bf_batting_points', 0)
        venue_bowling_avg = venue_stats.get('avg_ch_bowling_points', 0)
        
        # Identify batsmen and bowlers
        batsman_mask = np.isin(roles, ['BAT', 'WK', 'AR'])
        bowler_mask = np.isin(roles, ['BOWL', 'AR'])
        
        if np.any(batsman_mask):
            batsman_points = fantasy_points[batsman_mask]
            
            # Batsman form vs venue mean/max
            features['batsman_form_vs_venue_mean'] = float(np.mean(batsman_points) - venue_batting_avg)
            features['batsman_form_vs_venue_max'] = float(np.max(batsman_points) - venue_batting_avg)
            
            # Count of batsmen outperforming venue average
            features['batsman_form_advantage_count'] = float(np.sum(batsman_points > venue_batting_avg))
            
            # Best batsman's advantage over venue ceiling
            venue_batting_max = venue_stats.get('avg_bf_batting_points', 0) + venue_stats.get('std_bf_batting_points', 0)
            features['best_batsman_vs_venue_ceiling'] = float(np.max(batsman_points) - venue_batting_max)
        else:
            features.update({
                'batsman_form_vs_venue_mean': 0.0,
                'batsman_form_vs_venue_max': 0.0,
                'batsman_form_advantage_count': 0.0,
                'best_batsman_vs_venue_ceiling': 0.0
            })
        
        if np.any(bowler_mask):
            bowler_points = fantasy_points[bowler_mask]
            
            # Bowler form vs venue mean/max
            features['bowler_form_vs_venue_mean'] = float(np.mean(bowler_points) - venue_bowling_avg)
            features['bowler_form_vs_venue_max'] = float(np.max(bowler_points) - venue_bowling_avg)
            
            # Count of bowlers outperforming venue average
            features['bowler_form_advantage_count'] = float(np.sum(bowler_points > venue_bowling_avg))
            
            # Best bowler's advantage over venue ceiling
            venue_bowling_max = venue_stats.get('avg_ch_bowling_points', 0) + venue_stats.get('std_ch_bowling_points', 0)
            features['best_bowler_vs_venue_ceiling'] = float(np.max(bowler_points) - venue_bowling_max)
        else:
            features.update({
                'bowler_form_vs_venue_mean': 0.0,
                'bowler_form_vs_venue_max': 0.0,
                'bowler_form_advantage_count': 0.0,
                'best_bowler_vs_venue_ceiling': 0.0
            })
        
        return features
    
    def _extract_bowling_style_synergy_features(self, fantasy_points: np.ndarray,
                                              bowling_styles: List[str], roles: np.ndarray,
                                              venue_stats: Dict) -> Dict[str, float]:
        """
        Extract Bowling Style Venue Synergy features.
        
        These are multiplicative interactions - high value only when:
        (1) Team has good spinners/pacers, AND (2) Venue is friendly to that style.
        """
        features = {}
        
        # Venue bowling style preferences
        venue_spin_dominance = venue_stats.get('avg_spin_dominance', 0.5)
        venue_pace_dominance = venue_stats.get('avg_pace_dominance', 0.5)
        
        # Separate bowlers by style
        spinner_points = []
        pacer_points = []
        
        for i, role in enumerate(roles):
            if role in ['BOWL', 'AR'] and i < len(bowling_styles) and i < len(fantasy_points):
                style = bowling_styles[i]
                points = fantasy_points[i]
                
                if self._is_spinner(style):
                    spinner_points.append(points)
                elif self._is_pacer(style):
                    pacer_points.append(points)
        
        # Spinner venue synergy
        if spinner_points:
            # Multiplicative interaction: spinner quality × venue spin friendliness
            spinner_quality = np.mean(spinner_points)
            features['spinner_venue_synergy_total'] = float(spinner_quality * venue_spin_dominance * len(spinner_points))
            features['spinner_venue_synergy_avg'] = float(spinner_quality * venue_spin_dominance)
            features['spinner_venue_synergy_max'] = float(np.max(spinner_points) * venue_spin_dominance)
        else:
            features.update({
                'spinner_venue_synergy_total': 0.0,
                'spinner_venue_synergy_avg': 0.0,
                'spinner_venue_synergy_max': 0.0
            })
        
        # Pacer venue synergy  
        if pacer_points:
            # Multiplicative interaction: pacer quality × venue pace friendliness
            pacer_quality = np.mean(pacer_points)
            features['pacer_venue_synergy_total'] = float(pacer_quality * venue_pace_dominance * len(pacer_points))
            features['pacer_venue_synergy_avg'] = float(pacer_quality * venue_pace_dominance)
            features['pacer_venue_synergy_max'] = float(np.max(pacer_points) * venue_pace_dominance)
        else:
            features.update({
                'pacer_venue_synergy_total': 0.0,
                'pacer_venue_synergy_avg': 0.0,
                'pacer_venue_synergy_max': 0.0
            })
        
        return features
    
    def _extract_strategic_leverage_features(self, fantasy_points: np.ndarray,
                                           captain_idx: int, vc_idx: int,
                                           venue_stats: Dict) -> Dict[str, float]:
        """
        Extract Strategic Venue Leverage features.
        
        These isolate the most important players and ask specific questions
        about their venue-specific performance potential.
        """
        features = {}
        
        # Venue averages for comparison
        venue_batting_avg = venue_stats.get('avg_bf_batting_points', 0)
        
        # Captain venue leverage
        if captain_idx < len(fantasy_points):
            captain_points = fantasy_points[captain_idx]
            # Captain's form advantage over venue average (with captaincy multiplier effect)
            features['captain_venue_leverage'] = float((captain_points - venue_batting_avg) * 2.0)  # 2x multiplier for captaincy
        else:
            features['captain_venue_leverage'] = 0.0
        
        # Vice-captain venue leverage
        if vc_idx < len(fantasy_points):
            vc_points = fantasy_points[vc_idx]
            # VC's form advantage over venue average (with VC multiplier effect)
            features['vc_venue_leverage'] = float((vc_points - venue_batting_avg) * 1.5)  # 1.5x multiplier for VC
        else:
            features['vc_venue_leverage'] = 0.0
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when venue data is unavailable."""
        return {
            # Player Form vs Venue Context (8 features)
            'batsman_form_vs_venue_mean': 0.0,
            'batsman_form_vs_venue_max': 0.0,
            'batsman_form_advantage_count': 0.0,
            'best_batsman_vs_venue_ceiling': 0.0,
            'bowler_form_vs_venue_mean': 0.0,
            'bowler_form_vs_venue_max': 0.0,
            'bowler_form_advantage_count': 0.0,
            'best_bowler_vs_venue_ceiling': 0.0,
            
            # Bowling Style Venue Synergy (6 features)
            'spinner_venue_synergy_total': 0.0,
            'spinner_venue_synergy_avg': 0.0,
            'spinner_venue_synergy_max': 0.0,
            'pacer_venue_synergy_total': 0.0,
            'pacer_venue_synergy_avg': 0.0,
            'pacer_venue_synergy_max': 0.0,
            
            # Strategic Venue Leverage (2 features)
            'captain_venue_leverage': 0.0,
            'vc_venue_leverage': 0.0
        }
    
    def get_feature_count(self) -> int:
        """Return the total number of features extracted."""
        return 16  # 8 + 6 + 2
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return descriptions of all refined venue features."""
        return {
            # Category 1: Player Form vs Venue Context (GOLD STANDARD)
            'batsman_form_vs_venue_mean': 'Average batsman form minus venue batting average (value over baseline)',
            'batsman_form_vs_venue_max': 'Best batsman form minus venue batting average',
            'batsman_form_advantage_count': 'Number of batsmen outperforming venue average',
            'best_batsman_vs_venue_ceiling': 'Best batsman form minus venue batting ceiling',
            'bowler_form_vs_venue_mean': 'Average bowler form minus venue bowling average (value over baseline)',
            'bowler_form_vs_venue_max': 'Best bowler form minus venue bowling average',
            'bowler_form_advantage_count': 'Number of bowlers outperforming venue average',
            'best_bowler_vs_venue_ceiling': 'Best bowler form minus venue bowling ceiling',
            
            # Category 2: Bowling Style Venue Synergy (GOLD STANDARD)
            'spinner_venue_synergy_total': 'Total spinner quality × venue spin friendliness × spinner count',
            'spinner_venue_synergy_avg': 'Average spinner quality × venue spin friendliness',
            'spinner_venue_synergy_max': 'Best spinner quality × venue spin friendliness',
            'pacer_venue_synergy_total': 'Total pacer quality × venue pace friendliness × pacer count',
            'pacer_venue_synergy_avg': 'Average pacer quality × venue pace friendliness',
            'pacer_venue_synergy_max': 'Best pacer quality × venue pace friendliness',
            
            # Category 4: Strategic Venue Leverage (SELECTED HIGH-QUALITY)
            'captain_venue_leverage': 'Captain form advantage over venue average × 2.0 (captaincy multiplier)',
            'vc_venue_leverage': 'Vice-captain form advantage over venue average × 1.5 (VC multiplier)'
        }


if __name__ == "__main__":
    # Test the refined extractor
    print("🧪 TESTING REFINED VENUE INTERACTION FEATURE EXTRACTOR")
    print("=" * 60)
    
    extractor = RefinedVenueInteractionFeatureExtractor("./")
    
    # Show feature count and descriptions
    print(f"📊 Total Refined Features: {extractor.get_feature_count()}")
    print(f"📋 Feature Descriptions:")
    for feature, description in extractor.get_feature_descriptions().items():
        print(f"  • {feature}: {description}")
    
    print(f"\n✅ Refined venue feature extractor ready for production!")
    print(f"🎯 Focus: High-quality, interpretable features with clear mathematical definitions")
    print(f"❌ Removed: Vague 'fit', dangerous 'potential', and abstract 'momentum' features")
