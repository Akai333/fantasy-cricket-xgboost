#!/usr/bin/env python3
"""
🎯 MGAG LIVE ORCHESTRATOR
Complete MGAG-R3 pipeline for live matches from Live_Matches folder

FEATURES:
- Reads from Live_Matches/[match_folder]/ structure
- Auto-converts SimSim format to MGAG format
- Outputs results to same match folder
- Handles multiple live matches
- Full R3 Elite Discovery integration
"""

import pandas as pd
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Add mgag_r3_integration to sys.path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'mgag_r3_integration'))
sys.path.append(str(current_dir / 'dependencies'))

from simsim_to_mgag_converter import SimSimToMGAGConverter

# Import R3 components individually to avoid circular imports
try:
    from mgag_r3_integration.mgag_r3_model_interface import MGAGR3ModelInterface
    from mgag_r3_integration.mgag_round_robin_generator import MGAGRoundRobinGenerator
    # Skip problematic imports for now
    MGAGR3ModelInterface_available = True
except ImportError as e:
    logger.warning(f"⚠️ R3 imports failed: {e}")
    MGAGR3ModelInterface_available = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MGAGLiveOrchestrator:
    """
    Complete MGAG-R3 pipeline for live matches.
    Processes matches from Live_Matches folder and outputs results to same folder.
    """
    
    def __init__(self):
        self.mgag_dir = Path(__file__).parent
        self.live_matches_dir = self.mgag_dir / "Live_Matches"
        
        # Initialize components
        self.converter = SimSimToMGAGConverter()
        
        # Initialize R3 components if available
        if MGAGR3ModelInterface_available:
            try:
                self.model_interface = MGAGR3ModelInterface()
                self.team_generator = None  # Will be initialized when we have squad data
                self.r3_available = True
            except Exception as e:
                logger.warning(f"⚠️ R3 initialization failed: {e}")
                self.r3_available = False
                self.model_interface = None
                self.team_generator = None
        else:
            self.r3_available = False
            self.model_interface = None
            self.team_generator = None
        
        logger.info("🚀 MGAG Live Orchestrator initialized")
        logger.info("   ✅ SimSim to MGAG converter ready")
        logger.info("   ✅ R3 Elite Discovery Model loaded")
        logger.info("   ✅ Round-Robin team generator ready")
        logger.info("   ✅ Elite validation ready")
    
    def build_squad_context(self, match_data: Dict, eligible_players_df: pd.DataFrame) -> Dict[str, Any]:
        """Build squad context for team generator initialization"""
        try:
            batting_first_team = match_data.get('batting_first_team', '')
            chasing_team = match_data.get('chasing_team', '')
            
            # Split players by team
            batfirst_players = eligible_players_df[eligible_players_df['team'] == batting_first_team]
            chase_players = eligible_players_df[eligible_players_df['team'] == chasing_team]
            
            squad_context = {
                'batfirst_squad_player_ids': batfirst_players['player_id'].tolist(),
                'batfirst_squad_roles': batfirst_players['role'].tolist(),
                'batfirst_squad_avg_fantasy_points_last5': batfirst_players['avg_fantasy_points_last5'].tolist(),
                'chase_squad_player_ids': chase_players['player_id'].tolist(),
                'chase_squad_roles': chase_players['role'].tolist(),
                'chase_squad_avg_fantasy_points_last5': chase_players['avg_fantasy_points_last5'].tolist(),
                'batting_first_team': batting_first_team,
                'chasing_team': chasing_team
            }
            
            logger.info(f"✅ Squad context built: {len(batfirst_players)} vs {len(chase_players)} players")
            return squad_context
            
        except Exception as e:
            logger.error(f"❌ Error building squad context: {e}")
            return {}
    
    def enrich_teams_with_context(self, generated_teams: List[Dict], match_data: Dict, eligible_players_df: pd.DataFrame) -> List[Dict]:
        """Enrich generated teams with match_context and squad_context needed by R3 feature extractor"""
        try:
            # Build comprehensive match context
            match_context = {
                'cricsheet_match_id': match_data.get('cricsheet_match_id', ''),
                'date': match_data.get('date', ''),
                'venue': match_data.get('venue', ''),
                'league': 'Caribbean Premier League',
                'batting_first_team': match_data.get('batting_first_team', ''),
                'chasing_team': match_data.get('chasing_team', ''),
                'toss_winner': match_data.get('toss_winner', ''),
                'toss_decision': match_data.get('toss_decision', ''),
                'pitch': match_data.get('pitch', '')
            }
            
            # Build squad context arrays
            batting_first_team = match_data.get('batting_first_team', '')
            chasing_team = match_data.get('chasing_team', '')
            
            # Split players by team
            batfirst_players = eligible_players_df[eligible_players_df['team'] == batting_first_team]
            chase_players = eligible_players_df[eligible_players_df['team'] == chasing_team]
            
            # Build comprehensive squad context
            squad_context = {
                # Full squad player arrays (needed for feature extraction)
                'batfirst_squad_player_ids': batfirst_players['player_id'].tolist(),
                'batfirst_squad_roles': batfirst_players['role'].tolist(),
                'batfirst_squad_avg_fantasy_points_last5': batfirst_players['avg_fantasy_points_last5'].tolist(),
                'batfirst_squad_avg_balls_faced_last5': batfirst_players['avg_balls_faced_last5'].tolist(),
                'batfirst_squad_avg_overs_bowled_last5': batfirst_players['avg_overs_bowled_last5'].tolist(),
                'batfirst_squad_last10_fantasy_scores': batfirst_players['last10_fantasy_scores'].tolist(),
                'batfirst_squad_bowling_phases': batfirst_players['bowling_phases'].tolist(),
                
                'chase_squad_player_ids': chase_players['player_id'].tolist(),
                'chase_squad_roles': chase_players['role'].tolist(),
                'chase_squad_avg_fantasy_points_last5': chase_players['avg_fantasy_points_last5'].tolist(),
                'chase_squad_avg_balls_faced_last5': chase_players['avg_balls_faced_last5'].tolist(),
                'chase_squad_avg_overs_bowled_last5': chase_players['avg_overs_bowled_last5'].tolist(),
                'chase_squad_last10_fantasy_scores': chase_players['last10_fantasy_scores'].tolist(),
                'chase_squad_bowling_phases': chase_players['bowling_phases'].tolist(),
                
                # Team-level data arrays (for selected players in team)
                'roles': [],
                'avg_fantasy_points_last5_array': [],
                'last10_fantasy_scores_array': [],
                'avg_balls_faced_last5_array': [],
                'avg_overs_bowled_last5_array': [],
                'batting_order_array': [],
                'batting_style_array': [],
                'bowling_style_array': [],
                'bowling_phases_array': []
            }
            
            # Create player lookup for quick access
            player_lookup = eligible_players_df.set_index('player_id').to_dict('index')
            
            # Enrich each team
            enriched_teams = []
            for team in generated_teams:
                enriched_team = team.copy()
                
                # Add match and squad context
                enriched_team['match_context'] = match_context
                enriched_team['squad_context'] = squad_context.copy()
                
                # Build team-specific arrays based on selected players
                player_ids = team['player_ids']
                team_roles = []
                team_avg_fp = []
                team_last10_scores = []
                team_avg_balls = []
                team_avg_overs = []
                team_batting_order = []
                team_batting_style = []
                team_bowling_style = []
                team_bowling_phases = []
                
                for player_id in player_ids:
                    if player_id in player_lookup:
                        player_data = player_lookup[player_id]
                        team_roles.append(player_data.get('role', 'BAT'))
                        team_avg_fp.append(player_data.get('avg_fantasy_points_last5', 0.0))
                        team_last10_scores.append(player_data.get('last10_fantasy_scores', [0]*10))
                        team_avg_balls.append(player_data.get('avg_balls_faced_last5', 0.0))
                        team_avg_overs.append(player_data.get('avg_overs_bowled_last5', 0.0))
                        team_batting_order.append(player_data.get('batting_order', 0))
                        team_batting_style.append(player_data.get('batting_style', 'Right'))
                        team_bowling_style.append(player_data.get('bowling_style', 'None'))
                        team_bowling_phases.append(player_data.get('bowling_phases', [0,0,0]))
                    else:
                        # Default values for missing players
                        team_roles.append('BAT')
                        team_avg_fp.append(0.0)
                        team_last10_scores.append([0]*10)
                        team_avg_balls.append(0.0)
                        team_avg_overs.append(0.0)
                        team_batting_order.append(0)
                        team_batting_style.append('Right')
                        team_bowling_style.append('None')
                        team_bowling_phases.append([0,0,0])
                
                # Update squad context with team-specific arrays
                enriched_team['squad_context']['roles'] = team_roles
                enriched_team['squad_context']['avg_fantasy_points_last5_array'] = team_avg_fp
                enriched_team['squad_context']['last10_fantasy_scores_array'] = team_last10_scores
                enriched_team['squad_context']['avg_balls_faced_last5_array'] = team_avg_balls
                enriched_team['squad_context']['avg_overs_bowled_last5_array'] = team_avg_overs
                enriched_team['squad_context']['batting_order_array'] = team_batting_order
                enriched_team['squad_context']['batting_style_array'] = team_batting_style
                enriched_team['squad_context']['bowling_style_array'] = team_bowling_style
                enriched_team['squad_context']['bowling_phases_array'] = team_bowling_phases
                
                enriched_teams.append(enriched_team)
            
            logger.info(f"✅ Enriched {len(enriched_teams):,} teams with context")
            return enriched_teams
            
        except Exception as e:
            logger.error(f"❌ Error enriching teams with context: {e}")
            return generated_teams  # Return original teams as fallback
    
    def process_live_match(self, match_folder_name: str, num_teams: int = 50000) -> bool:
        """
        Process a single live match from folder name
        
        Args:
            match_folder_name: Name of folder in Live_Matches (e.g., 'CPL_TeamA_vs_TeamB_20240910')
            num_teams: Number of teams to generate
        
        Returns:
            True if processing successful, False otherwise
        """
        start_time = time.time()
        match_folder = self.live_matches_dir / match_folder_name
        
        if not match_folder.exists():
            logger.error(f"❌ Match folder not found: {match_folder}")
            return False
        
        logger.info(f"\n🎯 PROCESSING LIVE MATCH: {match_folder_name}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Convert SimSim format to MGAG format
            logger.info("🔄 Step 1: Converting SimSim CSV to MGAG format...")
            conversion_success = self.converter.convert_live_match(match_folder_name)
            
            if not conversion_success:
                logger.error("❌ CSV format conversion failed")
                return False
            
            # Step 2: Load converted data
            logger.info("🔄 Step 2: Loading converted match data...")
            match_data, eligible_players_df = self.load_match_data(match_folder)
            
            if match_data is None or eligible_players_df is None:
                logger.error("❌ Failed to load match data")
                return False
            
            # Step 3: Generate teams using R3 Round-Robin strategy
            logger.info(f"🔄 Step 3: Generating {num_teams:,} teams using R3 strategy...")
            
            # Initialize team generator with squad context
            if self.team_generator is None:
                squad_context = self.build_squad_context(match_data, eligible_players_df)
                self.team_generator = MGAGRoundRobinGenerator(squad_context)
            
            generated_teams = self.team_generator.generate_teams_systematic(num_teams)
            
            if not generated_teams:
                logger.error("❌ Team generation failed")
                return False
            
            logger.info(f"✅ Generated {len(generated_teams):,} unique teams")
            
            # Step 4: Enrich teams with context and predict quality scores
            logger.info("🔄 Step 4: Enriching teams with context...")
            enriched_teams = self.enrich_teams_with_context(generated_teams, match_data, eligible_players_df)
            
            logger.info("🔄 Step 4: Predicting team quality scores...")
            predicted_qualities = self.model_interface.predict_batch_quality(enriched_teams)
            
            # Add quality scores to teams
            for i, team in enumerate(generated_teams):
                team['predicted_quality'] = predicted_qualities[i]
            
            logger.info(f"✅ Quality scores predicted for {len(generated_teams):,} teams")
            
            # Step 5: Rank and select elite teams
            logger.info("🔄 Step 5: Ranking and selecting elite teams...")
            ranked_teams = sorted(generated_teams, key=lambda x: x['predicted_quality'], reverse=True)
            
            # Select top teams at different thresholds
            top_10 = ranked_teams[:10]
            top_50 = ranked_teams[:50]
            top_100 = ranked_teams[:100]
            elite_teams = [team for team in ranked_teams if team['predicted_quality'] >= 0.8]
            
            logger.info(f"✅ Elite team selection completed:")
            logger.info(f"   Best quality score: {top_10[0]['predicted_quality']:.4f}")
            logger.info(f"   Elite teams (≥0.8): {len(elite_teams)}")
            logger.info(f"   Elite percentage: {(len(elite_teams)/len(ranked_teams)*100):.2f}%")
            
            # Step 6: Save results to match folder
            logger.info("🔄 Step 6: Saving results...")
            results_saved = self.save_results(
                match_folder, match_data, ranked_teams, 
                top_10, top_50, top_100, elite_teams,
                num_teams, start_time
            )
            
            if not results_saved:
                logger.warning("⚠️ Results saving had issues")
            
            total_time = time.time() - start_time
            logger.info(f"✅ LIVE MATCH PROCESSING COMPLETED in {total_time:.1f} seconds")
            logger.info(f"📁 Results saved in: {match_folder}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Live match processing error: {e}")
            return False
    
    def load_match_data(self, match_folder: Path) -> tuple[Optional[Dict], Optional[pd.DataFrame]]:
        """Load match data from converted CSV files"""
        try:
            matches_file = match_folder / "matches.csv"
            players_file = match_folder / "players.csv"
            
            if not matches_file.exists() or not players_file.exists():
                logger.error("❌ Converted CSV files not found")
                return None, None
            
            # Load match metadata
            matches_df = pd.read_csv(matches_file)
            if len(matches_df) == 0:
                logger.error("❌ Empty matches.csv")
                return None, None
            
            match_data = matches_df.iloc[0].to_dict()
            
            # Load player data
            players_df = pd.read_csv(players_file)
            if len(players_df) == 0:
                logger.error("❌ Empty players.csv")
                return None, None
            
            # Prepare eligible players data for R3 team generator
            eligible_players_df = self.prepare_eligible_players_data(players_df)
            
            logger.info(f"✅ Match data loaded:")
            logger.info(f"   Match ID: {match_data.get('cricsheet_match_id')}")
            logger.info(f"   Teams: {match_data.get('batting_first_team')} vs {match_data.get('chasing_team')}")
            logger.info(f"   Players: {len(eligible_players_df)}")
            
            return match_data, eligible_players_df
            
        except Exception as e:
            logger.error(f"❌ Data loading error: {e}")
            return None, None
    
    def prepare_eligible_players_data(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare player data in format expected by R3 team generator"""
        try:
            # Convert columns to expected format
            eligible_players_df = players_df.copy()
            
            # Parse complex list columns
            def safe_eval_list(x):
                if pd.isna(x) or x == '' or x == '[]':
                    return []
                try:
                    if isinstance(x, str):
                        # Handle different formats
                        if x.startswith('[') and x.endswith(']'):
                            return eval(x)
                        else:
                            # Comma separated values
                            return [float(val.strip()) for val in x.split(',') if val.strip()]
                    else:
                        return x if isinstance(x, list) else []
                except:
                    return []
            
            # Parse last10_fantasy_scores
            eligible_players_df['last10_fantasy_scores_list'] = eligible_players_df['last10_fantasy_scores'].apply(safe_eval_list)
            
            # Parse bowling_phases
            eligible_players_df['bowling_phases_list'] = eligible_players_df['bowling_phases'].apply(safe_eval_list)
            
            # Ensure required columns exist with defaults
            if 'player_name' not in eligible_players_df.columns and 'name' in eligible_players_df.columns:
                eligible_players_df['player_name'] = eligible_players_df['name']
            
            # Fill missing values
            eligible_players_df['batting_order'] = eligible_players_df['batting_order'].fillna(0)
            eligible_players_df['avg_fantasy_points_last5'] = eligible_players_df['avg_fantasy_points_last5'].fillna(0.0)
            eligible_players_df['avg_balls_faced_last5'] = eligible_players_df['avg_balls_faced_last5'].fillna(0.0)
            eligible_players_df['avg_overs_bowled_last5'] = eligible_players_df['avg_overs_bowled_last5'].fillna(0.0)
            
            logger.info(f"✅ Player data prepared for R3 team generator")
            return eligible_players_df
            
        except Exception as e:
            logger.error(f"❌ Player data preparation error: {e}")
            return players_df
    
    def save_results(self, match_folder: Path, match_data: Dict, ranked_teams: List[Dict],
                    top_10: List[Dict], top_50: List[Dict], top_100: List[Dict], 
                    elite_teams: List[Dict], num_teams: int, start_time: float) -> bool:
        """Save results to match folder"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            match_id = match_data.get('cricsheet_match_id', 'unknown')
            
            # Create comprehensive results
            results = {
                'metadata': {
                    'match_id': match_id,
                    'match_folder': match_folder.name,
                    'processing_timestamp': timestamp,
                    'processing_time_seconds': time.time() - start_time,
                    'teams_requested': num_teams,
                    'teams_generated': len(ranked_teams),
                    'model_version': 'R3_Elite_Discovery_v1.0',
                    'system_version': 'MGAG-R3_Live_v1.0'
                },
                'match_context': {
                    'batting_first_team': match_data.get('batting_first_team'),
                    'chasing_team': match_data.get('chasing_team'),
                    'venue': match_data.get('venue'),
                    'date': match_data.get('date'),
                    'toss_winner': match_data.get('toss_winner'),
                    'toss_decision': match_data.get('toss_decision'),
                    'pitch': match_data.get('pitch')
                },
                'quality_statistics': {
                    'best_quality': top_10[0]['predicted_quality'] if top_10 else 0.0,
                    'top_10_avg': sum(team['predicted_quality'] for team in top_10) / len(top_10) if top_10 else 0.0,
                    'elite_count': len(elite_teams),
                    'elite_percentage': (len(elite_teams) / len(ranked_teams) * 100) if ranked_teams else 0.0,
                    'quality_distribution': {
                        'min': min(team['predicted_quality'] for team in ranked_teams) if ranked_teams else 0.0,
                        'max': max(team['predicted_quality'] for team in ranked_teams) if ranked_teams else 0.0,
                        'mean': sum(team['predicted_quality'] for team in ranked_teams) / len(ranked_teams) if ranked_teams else 0.0
                    }
                },
                'top_teams': {
                    'top_10': self.format_teams_for_output(top_10),
                    'top_50': self.format_teams_for_output(top_50),
                    'top_100': self.format_teams_for_output(top_100)
                },
                'elite_teams': self.format_teams_for_output(elite_teams),
                'generation_summary': {
                    'unique_teams': len(ranked_teams),
                    'generation_rate': len(ranked_teams) / (time.time() - start_time),
                    'average_quality': sum(team['predicted_quality'] for team in ranked_teams) / len(ranked_teams) if ranked_teams else 0.0
                }
            }
            
            # Save comprehensive results
            results_file = match_folder / f"mgag_r3_results_{match_id}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save top teams as CSV with player names for Dream11
            if top_100:
                top_teams_data = []
                
                # Create player ID to name mapping
                player_id_to_name = {}
                players_file = match_folder / "players.csv"
                if players_file.exists():
                    players_df = pd.read_csv(players_file)
                    player_id_to_name = dict(zip(players_df['player_id'], players_df['player_name']))
                
                for i, team in enumerate(top_100):
                    # Convert player IDs to names
                    player_names = []
                    for player_id in team['player_ids']:
                        player_name = player_id_to_name.get(player_id, player_id)
                        player_names.append(player_name)
                    
                    captain_name = player_id_to_name.get(team['captain_id'], team['captain_id'])
                    vc_name = player_id_to_name.get(team['vice_captain_id'], team['vice_captain_id'])
                    
                    team_row = {
                        'rank': i + 1,
                        'quality_score': f"{team['predicted_quality']:.4f}",
                        'captain': captain_name,
                        'vice_captain': vc_name,
                        'player_1': player_names[0] if len(player_names) > 0 else '',
                        'player_2': player_names[1] if len(player_names) > 1 else '',
                        'player_3': player_names[2] if len(player_names) > 2 else '',
                        'player_4': player_names[3] if len(player_names) > 3 else '',
                        'player_5': player_names[4] if len(player_names) > 4 else '',
                        'player_6': player_names[5] if len(player_names) > 5 else '',
                        'player_7': player_names[6] if len(player_names) > 6 else '',
                        'player_8': player_names[7] if len(player_names) > 7 else '',
                        'player_9': player_names[8] if len(player_names) > 8 else '',
                        'player_10': player_names[9] if len(player_names) > 9 else '',
                        'player_11': player_names[10] if len(player_names) > 10 else '',
                        'all_players': ' | '.join(player_names)
                    }
                    top_teams_data.append(team_row)
                
                # Save top 100 teams
                top_100_df = pd.DataFrame(top_teams_data)
                csv_file_100 = match_folder / f"top_100_teams_{match_id}_{timestamp}.csv"
                top_100_df.to_csv(csv_file_100, index=False)
                
                # Save top 20 teams (separate file for convenience)
                top_20_df = pd.DataFrame(top_teams_data[:20])
                csv_file_20 = match_folder / f"top_20_teams_{match_id}_{timestamp}.csv"
                top_20_df.to_csv(csv_file_20, index=False)
                
                logger.info(f"✅ Top 100 teams CSV saved: {csv_file_100}")
                logger.info(f"✅ Top 20 teams CSV saved: {csv_file_20}")
                logger.info(f"   Teams ready for Dream11 with player names!")
            
            # Save elite teams separately if any found
            if elite_teams:
                elite_file = match_folder / f"elite_teams_{match_id}_{timestamp}.json"
                with open(elite_file, 'w') as f:
                    json.dump({
                        'elite_threshold': 0.8,
                        'elite_count': len(elite_teams),
                        'elite_teams': self.format_teams_for_output(elite_teams)
                    }, f, indent=2)
                
                logger.info(f"✅ Elite teams saved: {elite_file}")
            
            logger.info(f"✅ Results saved: {results_file}")
            logger.info(f"   Best team quality: {results['quality_statistics']['best_quality']:.4f}")
            logger.info(f"   Elite teams found: {results['quality_statistics']['elite_count']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Results saving error: {e}")
            return False
    
    def format_teams_for_output(self, teams: List[Dict]) -> List[Dict]:
        """Format teams for JSON output (remove internal fields)"""
        formatted_teams = []
        
        for team in teams:
            formatted_team = {
                'player_ids': team.get('player_ids', []),
                'captain_id': team.get('captain_id', ''),
                'vice_captain_id': team.get('vice_captain_id', ''),
                'predicted_quality': team.get('predicted_quality', 0.0),
                'contextual_template': team.get('contextual_template', '')
            }
            formatted_teams.append(formatted_team)
        
        return formatted_teams
    
    def list_available_live_matches(self) -> List[str]:
        """List all available live matches"""
        return self.converter.list_available_matches()
    
    def process_all_live_matches(self, num_teams: int = 50000) -> Dict[str, bool]:
        """Process all available live matches"""
        available_matches = self.list_available_live_matches()
        
        if not available_matches:
            logger.warning("⚠️ No live matches found")
            return {}
        
        logger.info(f"🔄 Processing {len(available_matches)} live matches...")
        
        results = {}
        for match_folder_name in available_matches:
            logger.info(f"\n{'='*20} PROCESSING {match_folder_name} {'='*20}")
            success = self.process_live_match(match_folder_name, num_teams)
            results[match_folder_name] = success
        
        # Summary
        successful = sum(results.values())
        logger.info(f"\n📊 PROCESSING SUMMARY:")
        logger.info(f"   Total matches: {len(available_matches)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {len(available_matches) - successful}")
        
        return results


def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MGAG Live Match Orchestrator')
    parser.add_argument('--match', type=str, help='Specific match folder name to process')
    parser.add_argument('--all', action='store_true', help='Process all available live matches')
    parser.add_argument('--list', action='store_true', help='List available live matches')
    parser.add_argument('--teams', type=int, default=50000, help='Number of teams to generate (default: 50000)')
    
    args = parser.parse_args()
    
    orchestrator = MGAGLiveOrchestrator()
    
    if args.list:
        matches = orchestrator.list_available_live_matches()
        print(f"\n📋 Available Live Matches ({len(matches)}):")
        for i, match in enumerate(matches, 1):
            print(f"   {i}. {match}")
    
    elif args.all:
        results = orchestrator.process_all_live_matches(args.teams)
        
    elif args.match:
        success = orchestrator.process_live_match(args.match, args.teams)
        if success:
            print(f"✅ Successfully processed: {args.match}")
        else:
            print(f"❌ Failed to process: {args.match}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
