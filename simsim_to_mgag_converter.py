#!/usr/bin/env python3
"""
🔄 SIMSIM TO MGAG CONVERTER
Converts SimSim CSV format to MGAG-R3 compatible format for live matches

FEATURES:
- Converts match_context.csv to matches.csv format
- Converts squads_combined.csv to players.csv format  
- Handles Live_Matches folder structure
- Provides data validation and error checking
- Ensures MGAG-R3 compatibility
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimSimToMGAGConverter:
    """Converts SimSim CSV format to MGAG-R3 compatible format"""
    
    def __init__(self):
        self.mgag_dir = Path(__file__).parent
        self.live_matches_dir = self.mgag_dir / "Live_Matches"
        self.database_join_dir = self.mgag_dir / "database_join"
        
        # Ensure database_join directory exists
        self.database_join_dir.mkdir(exist_ok=True)
        
        logger.info("🔄 SimSim to MGAG Converter initialized")
    
    def convert_live_match(self, match_folder_name: str) -> bool:
        """
        Convert a specific live match from various formats to MGAG format
        
        Args:
            match_folder_name: Name of the folder in Live_Matches (e.g., 'CPL_TeamA_vs_TeamB_20240910')
        
        Returns:
            True if conversion successful, False otherwise
        """
        match_folder = self.live_matches_dir / match_folder_name
        
        if not match_folder.exists():
            logger.error(f"❌ Match folder not found: {match_folder}")
            return False
        
        logger.info(f"🎯 Converting live match: {match_folder_name}")
        
        # Check for different CSV formats (flexible)
        context_file = match_folder / "match_context.csv"
        squad_file = None
        
        # Try different squad file names
        for possible_squad_file in ["squads_combined.csv", "squads.csv", "players.csv"]:
            potential_file = match_folder / possible_squad_file
            if potential_file.exists():
                squad_file = potential_file
                break
        
        if not context_file.exists():
            logger.error(f"❌ match_context.csv not found in {match_folder}")
            return False
        
        if not squad_file:
            logger.error(f"❌ No squad file found in {match_folder} (tried: squads_combined.csv, squads.csv, players.csv)")
            return False
        
        logger.info(f"📋 Found context file: {context_file.name}")
        logger.info(f"📋 Found squad file: {squad_file.name}")
        
        try:
            # Step 1: Convert match context
            match_success = self.convert_match_context(context_file, match_folder)
            if not match_success:
                logger.error("❌ Match context conversion failed")
                return False
            
            # Step 2: Convert squad data
            squad_success = self.convert_squad_data(squad_file, match_folder)
            if not squad_success:
                logger.error("❌ Squad data conversion failed")
                return False
            
            # Step 3: Validate converted data
            validation_success = self.validate_converted_data(match_folder)
            if not validation_success:
                logger.warning("⚠️ Data validation found issues, but conversion completed")
            
            logger.info(f"✅ Live match conversion completed: {match_folder_name}")
            logger.info(f"📁 MGAG-compatible files saved in: {match_folder}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Conversion error: {e}")
            return False
    
    def convert_match_context(self, context_file: Path, output_folder: Path) -> bool:
        """Convert match_context.csv to MGAG matches.csv format (handles multiple formats)"""
        try:
            # Read context format
            context_df = pd.read_csv(context_file)
            
            # Detect format and extract data accordingly
            if 'field' in context_df.columns and 'value' in context_df.columns:
                # SimSim field-value format
                context_dict = {}
                for _, row in context_df.iterrows():
                    context_dict[row['field']] = row['value']
                
                cricsheet_match_id = context_dict.get('match_id', f"live_match_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                series_name = context_dict.get('series_name', 'Caribbean Premier League')
                venue = context_dict.get('venue', 'Unknown Venue')
                date = context_dict.get('date', pd.Timestamp.now().strftime('%Y-%m-%d'))
                team_a = context_dict.get('team_a', 'Team A')
                team_b = context_dict.get('team_b', 'Team B')
                toss_winner = context_dict.get('toss_winner', team_a)
                toss_decision = context_dict.get('toss_decision', 'bat')
                pitch_conditions = context_dict.get('pitch_conditions', 'good')
                
            else:
                # Direct column format (current Live_Matches format)
                row = context_df.iloc[0]  # Assume single row
                
                cricsheet_match_id = row.get('match_id', f"live_match_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                series_name = row.get('series', row.get('league', 'Caribbean Premier League'))
                venue = row.get('venue', 'Unknown Venue')
                date = row.get('date', pd.Timestamp.now().strftime('%Y-%m-%d'))
                team_a = row.get('batting_first_team', 'Team A')
                team_b = row.get('chasing_team', 'Team B')
                toss_winner = row.get('toss_winner', team_a)
                toss_decision = row.get('toss_decision', 'bat')
                pitch_conditions = row.get('pitch_conditions', row.get('pitch', 'good'))
            
            # Determine batting order based on toss
            if toss_decision.lower() in ['bat', 'batting']:
                batting_first_team = toss_winner
                chasing_team = team_b if toss_winner == team_a else team_a
            else:  # field/bowling
                batting_first_team = team_b if toss_winner == team_a else team_a
                chasing_team = toss_winner
            
            # Create MGAG matches.csv format
            matches_data = [{
                'cricsheet_match_id': cricsheet_match_id,
                'date': date,
                'venue': venue,
                'league': series_name,
                'gender': 'male',  # Default
                'batting_first_team': batting_first_team,
                'chasing_team': chasing_team,
                'toss_winner': toss_winner,
                'toss_decision': toss_decision.lower(),
                'pitch': pitch_conditions
            }]
            
            matches_df = pd.DataFrame(matches_data)
            
            # Save to output folder (for this specific match)
            output_file = output_folder / "matches.csv"
            matches_df.to_csv(output_file, index=False)
            
            # Also save to central database_join for MGAG compatibility
            central_matches_file = self.database_join_dir / "matches.csv"
            matches_df.to_csv(central_matches_file, index=False)
            
            logger.info(f"✅ Match context converted: {output_file}")
            logger.info(f"   Match ID: {cricsheet_match_id}")
            logger.info(f"   Teams: {batting_first_team} vs {chasing_team}")
            logger.info(f"   Toss: {toss_winner} chose to {toss_decision}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Match context conversion error: {e}")
            return False
    
    def convert_squad_data(self, squad_file: Path, output_folder: Path) -> bool:
        """Convert squads data to MGAG players.csv format (handles multiple formats)"""
        try:
            # Read squad format
            squad_df = pd.read_csv(squad_file)
            
            # Get match ID from the converted matches.csv
            matches_file = output_folder / "matches.csv"
            if matches_file.exists():
                matches_df = pd.read_csv(matches_file)
                cricsheet_match_id = matches_df.iloc[0]['cricsheet_match_id']
            else:
                cricsheet_match_id = f"live_match_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert each player row
            players_data = []
            for _, row in squad_df.iterrows():
                
                # Handle different column names and formats
                player_name = row.get('player_name', row.get('name', 'Unknown Player'))
                player_id = row.get('player_id', f"player_{player_name.replace(' ', '_').lower()}")
                team = row.get('team', 'Unknown Team')
                role = self.map_role(row.get('role', 'BAT'))
                
                # Handle batting/bowling styles with different possible column names
                batting_style = row.get('batting_style', row.get('batting_hand', 'Right')).replace(' Handed Bat', '').replace(' Hand', '')
                bowling_style = row.get('bowling_style', 'None')
                if pd.isna(bowling_style) or bowling_style == '':
                    bowling_style = 'None'
                
                # Handle statistics with different column names
                avg_fantasy = float(row.get('avg_fantasy_pts', row.get('avg_fantasy_points_last5', 0.0)))
                avg_balls = float(row.get('avg_balls_faced', row.get('avg_balls_faced_last5', 0.0)))
                avg_overs = float(row.get('avg_overs_bowled', row.get('avg_overs_bowled_last5', 0.0)))
                
                # Parse last_10_scores (handle different formats)
                last_10_scores = self.parse_last_10_scores(row.get('last_10_scores', row.get('last10_fantasy_scores', '')))
                
                # Parse bowling phases (handle different formats)
                bowling_phases = self.parse_bowling_phases(row.get('bowl_phases', row.get('bowling_phases', '0.0,0.0,0.0')))
                
                # Handle ownership data
                ownership = float(row.get('ownership_pct', row.get('ownership', 1.0)))
                captain_pct = float(row.get('captain_pct', row.get('cown', 1.0)))
                vc_pct = float(row.get('vc_pct', row.get('vcown', 1.0)))
                
                # Create MGAG player record
                player_record = {
                    'cricsheet_match_id': cricsheet_match_id,
                    'player_id': player_id,
                    'player_name': player_name,
                    'team': team,
                    'role': role,
                    'batting_style': batting_style,
                    'bowling_style': bowling_style,
                    'batting_order': int(row.get('batting_order', 0)) if str(row.get('batting_order', '')).strip() != '' else 0,
                    'avg_fantasy_points_last5': avg_fantasy,
                    'avg_balls_faced_last5': avg_balls,
                    'avg_overs_bowled_last5': avg_overs,
                    'last10_fantasy_scores': str(last_10_scores),
                    'bowling_phases': str(bowling_phases),
                    'fantasy_points': 0.0,  # Will be filled after match
                    'ownership_pct': ownership,
                    'captain_pct': captain_pct,
                    'vc_pct': vc_pct
                }
                
                players_data.append(player_record)
            
            players_df = pd.DataFrame(players_data)
            
            # Save to output folder (for this specific match)
            output_file = output_folder / "players.csv"
            players_df.to_csv(output_file, index=False)
            
            # Also save to central database_join for MGAG compatibility
            central_players_file = self.database_join_dir / "players.csv"
            players_df.to_csv(central_players_file, index=False)
            
            logger.info(f"✅ Squad data converted: {output_file}")
            logger.info(f"   Total players: {len(players_df)}")
            logger.info(f"   Teams: {players_df['team'].unique().tolist()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Squad data conversion error: {e}")
            return False
    
    def parse_last_10_scores(self, scores_str: str) -> List[float]:
        """Parse last 10 scores from various formats"""
        if not scores_str or scores_str.strip() == '':
            return [0.0] * 10
        
        try:
            # Handle different separators
            if '|' in scores_str:
                scores = scores_str.split('|')
            elif ',' in scores_str:
                scores = scores_str.split(',')
            else:
                # Single score or space separated
                scores = scores_str.split()
            
            # Convert to floats and pad/trim to 10
            float_scores = []
            for score in scores:
                try:
                    float_scores.append(float(score.strip()))
                except ValueError:
                    float_scores.append(0.0)
            
            # Ensure exactly 10 scores
            if len(float_scores) < 10:
                float_scores.extend([0.0] * (10 - len(float_scores)))
            else:
                float_scores = float_scores[:10]
            
            return float_scores
            
        except Exception:
            return [0.0] * 10
    
    def parse_bowling_phases(self, phases_str: str) -> List[float]:
        """Parse bowling phases from comma-separated format"""
        if not phases_str or phases_str.strip() == '':
            return [0.0, 0.0, 0.0]
        
        try:
            phases = [float(x.strip()) for x in phases_str.split(',')]
            
            # Ensure exactly 3 phases
            if len(phases) < 3:
                phases.extend([0.0] * (3 - len(phases)))
            else:
                phases = phases[:3]
            
            # Normalize to sum to 1.0 if they don't already
            phase_sum = sum(phases)
            if phase_sum > 0:
                phases = [p / phase_sum for p in phases]
            
            return phases
            
        except Exception:
            return [0.0, 0.0, 0.0]
    
    def map_role(self, role: str) -> str:
        """Map role to MGAG standard format"""
        role_mapping = {
            'WK': 'WK',
            'BAT': 'BAT', 
            'AR': 'AR',
            'BOWL': 'BOWL',
            'Wicketkeeper': 'WK',
            'Batsman': 'BAT',
            'All-rounder': 'AR', 
            'Allrounder': 'AR',
            'Bowler': 'BOWL'
        }
        
        return role_mapping.get(role, 'BAT')  # Default to BAT
    
    def validate_converted_data(self, match_folder: Path) -> bool:
        """Validate the converted data for MGAG compatibility"""
        try:
            matches_file = match_folder / "matches.csv"
            players_file = match_folder / "players.csv"
            
            if not matches_file.exists() or not players_file.exists():
                logger.error("❌ Converted files not found")
                return False
            
            # Validate matches.csv
            matches_df = pd.read_csv(matches_file)
            if len(matches_df) != 1:
                logger.error("❌ matches.csv should contain exactly 1 match")
                return False
            
            # Validate players.csv
            players_df = pd.read_csv(players_file)
            
            # Check player count (should be exactly 22 for cricket)
            if len(players_df) < 20 or len(players_df) > 25:
                logger.warning(f"⚠️ Unusual player count: {len(players_df)} (expected ~22)")
            
            # Check teams
            teams = players_df['team'].unique()
            if len(teams) != 2:
                logger.error(f"❌ Should have exactly 2 teams, found: {len(teams)}")
                return False
            
            # Check roles
            required_roles = ['WK', 'BAT', 'AR', 'BOWL']
            available_roles = players_df['role'].unique()
            missing_roles = [role for role in required_roles if role not in available_roles]
            if missing_roles:
                logger.warning(f"⚠️ Missing roles: {missing_roles}")
            
            # Check data completeness
            required_fields = ['player_id', 'player_name', 'team', 'role']
            for field in required_fields:
                if players_df[field].isna().any():
                    logger.warning(f"⚠️ Missing values in {field}")
            
            logger.info("✅ Data validation completed")
            logger.info(f"   Teams: {teams.tolist()}")
            logger.info(f"   Players per team: {players_df['team'].value_counts().tolist()}")
            logger.info(f"   Available roles: {available_roles.tolist()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False
    
    def list_available_matches(self) -> List[str]:
        """List all available live matches in Live_Matches folder"""
        if not self.live_matches_dir.exists():
            return []
        
        matches = []
        for folder in self.live_matches_dir.iterdir():
            if folder.is_dir():
                # Check if it has match context file
                context_file = folder / "match_context.csv"
                
                # Check for any squad file format
                squad_file = None
                for possible_squad_file in ["squads_combined.csv", "squads.csv", "players.csv"]:
                    potential_file = folder / possible_squad_file
                    if potential_file.exists():
                        squad_file = potential_file
                        break
                
                if context_file.exists() and squad_file:
                    matches.append(folder.name)
        
        return matches
    
    def convert_all_live_matches(self) -> Dict[str, bool]:
        """Convert all available live matches"""
        available_matches = self.list_available_matches()
        
        if not available_matches:
            logger.warning("⚠️ No live matches found in Live_Matches folder")
            return {}
        
        results = {}
        logger.info(f"🔄 Converting {len(available_matches)} live matches...")
        
        for match_folder_name in available_matches:
            logger.info(f"\n📋 Processing: {match_folder_name}")
            success = self.convert_live_match(match_folder_name)
            results[match_folder_name] = success
        
        # Summary
        successful = sum(results.values())
        logger.info(f"\n📊 Conversion Summary:")
        logger.info(f"   Total matches: {len(available_matches)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {len(available_matches) - successful}")
        
        return results


def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SimSim CSV format to MGAG-R3 format')
    parser.add_argument('--match', type=str, help='Specific match folder name to convert')
    parser.add_argument('--all', action='store_true', help='Convert all available live matches')
    parser.add_argument('--list', action='store_true', help='List available live matches')
    
    args = parser.parse_args()
    
    converter = SimSimToMGAGConverter()
    
    if args.list:
        matches = converter.list_available_matches()
        print(f"\n📋 Available Live Matches ({len(matches)}):")
        for i, match in enumerate(matches, 1):
            print(f"   {i}. {match}")
    
    elif args.all:
        results = converter.convert_all_live_matches()
        
    elif args.match:
        success = converter.convert_live_match(args.match)
        if success:
            print(f"✅ Successfully converted: {args.match}")
        else:
            print(f"❌ Failed to convert: {args.match}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
