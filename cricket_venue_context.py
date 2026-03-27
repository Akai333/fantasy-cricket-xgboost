#!/usr/bin/env python3
"""
Cricket Venue Context Generator
Creates venue-specific contextual features for fantasy cricket modeling.

This script analyzes historical match data to generate venue "DNA" profiles
that capture how different grounds affect fantasy point distribution.

Key Venue Features:
1. Batting Context: Average fantasy points for batting-first team
2. Bowling Context: Average fantasy points for chasing team bowlers  
3. Chase Advantage: Fantasy point differential between chase and bat-first teams
4. Volatility: Standard deviations of all above (for GPP vs Cash game context)

Output: League-specific JSON files (e.g., CPL_venue.json, IPL_venue.json)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
import argparse

class VenueContextGenerator:
    def __init__(self, data_directory: str = None):
        """Initialize with path to directory containing matches.csv and players.csv"""
        if data_directory:
            self.data_dir = Path(data_directory)
        else:
            # Default: from Scripts folder, go up to database_join folder
            self.data_dir = Path(__file__).parent.parent
        self.matches_path = self.data_dir / "matches.csv"
        self.players_path = self.data_dir / "players.csv"
        
        # Bowling style classification
        self.spin_styles = {
            'Right-arm offbreak', 'Right-arm legbreak', 
            'Left-arm orthodox', 'Left-arm wrist-spin'
        }
        self.pace_styles = {
            'Right-arm medium', 'Right-arm fast-medium', 'Right-arm fast',
            'Left-arm fast-medium', 'Left-arm medium', 'Left-arm fast'
        }
        
        # Load data
        self.matches_df = None
        self.players_df = None
        self.load_data()
        
    def load_data(self):
        """Load matches and players data"""
        try:
            if not self.matches_path.exists():
                raise FileNotFoundError(f"matches.csv not found at {self.matches_path}")
            if not self.players_path.exists():
                raise FileNotFoundError(f"players.csv not found at {self.players_path}")
                
            print(f"📂 Loading matches from: {self.matches_path}")
            self.matches_df = pd.read_csv(self.matches_path)
            print(f"✅ Loaded {len(self.matches_df)} matches")
            
            print(f"📂 Loading players from: {self.players_path}")
            self.players_df = pd.read_csv(self.players_path)
            print(f"✅ Loaded {len(self.players_df)} player records")
            
            # Data validation
            self.validate_data()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def validate_data(self):
        """Validate that required columns exist"""
        required_match_cols = ['cricsheet_match_id', 'venue', 'league', 'batting_first_team', 'chasing_team']
        required_player_cols = ['cricsheet_match_id', 'team', 'fantasy_points', 'role', 'overs_bowled']
        
        missing_match_cols = [col for col in required_match_cols if col not in self.matches_df.columns]
        missing_player_cols = [col for col in required_player_cols if col not in self.players_df.columns]
        
        if missing_match_cols:
            raise ValueError(f"Missing required columns in matches.csv: {missing_match_cols}")
        if missing_player_cols:
            raise ValueError(f"Missing required columns in players.csv: {missing_player_cols}")
            
        print("✅ Data validation passed")
    
    def is_spinner(self, bowling_style: str) -> bool:
        """Check if a bowling style is spin"""
        return bowling_style in self.spin_styles
    
    def is_pacer(self, bowling_style: str) -> bool:
        """Check if a bowling style is pace"""
        return bowling_style in self.pace_styles
    
    def split_ar_points(self, fantasy_points: float, balls_faced: int, overs_bowled: float) -> tuple:
        """Split all-rounder fantasy points into batting and bowling components"""
        if balls_faced == 0 and overs_bowled == 0:
            return 0.0, 0.0
        
        # Convert overs to balls for calculation
        balls_bowled = overs_bowled * 6
        total_involvement = balls_faced + balls_bowled
        
        if total_involvement == 0:
            return 0.0, 0.0
        
        # Calculate proportions
        batting_proportion = balls_faced / total_involvement
        bowling_proportion = balls_bowled / total_involvement
        
        # Split fantasy points
        ar_batting_points = fantasy_points * batting_proportion
        ar_bowling_points = fantasy_points * bowling_proportion
        
        return ar_batting_points, ar_bowling_points
    
    def calculate_venue_stats(self, league: Optional[str] = None, min_matches: int = 1) -> Dict:
        """
        Calculate comprehensive venue statistics from historical data
        
        Args:
            league: Filter by specific league (e.g., 'CPL', 'IPL'). If None, processes all leagues.
            min_matches: Minimum matches required at a venue to include it
            
        Returns:
            Dictionary with venue statistics
        """
        print(f"🏟️  Calculating venue statistics...")
        if league:
            print(f"🎯 Filtering for league: {league}")
        
        # Filter matches by league if specified
        matches_filtered = self.matches_df.copy()
        if league:
            matches_filtered = matches_filtered[matches_filtered['league'].str.contains(league, case=False, na=False)]
            print(f"📊 Found {len(matches_filtered)} {league} matches")
        
        # Group matches by venue
        venue_stats = {}
        
        for venue in matches_filtered['venue'].unique():
            if pd.isna(venue) or venue.strip() == '':
                continue
                
            venue_matches = matches_filtered[matches_filtered['venue'] == venue]
            
            if len(venue_matches) < min_matches:
                print(f"⚠️  Processing {venue} with only {len(venue_matches)} matches (limited data)")
            else:
                print(f"🏟️  Processing {venue}: {len(venue_matches)} matches")
            
            # Calculate venue statistics
            venue_data = self.calculate_single_venue_stats(venue, venue_matches)
            
            if venue_data:
                venue_stats[venue] = venue_data
                
        print(f"✅ Processed {len(venue_stats)} venues with sufficient data")
        return venue_stats
    
    def calculate_single_venue_stats(self, venue: str, venue_matches: pd.DataFrame) -> Dict:
        """Calculate statistics for a single venue"""
        try:
            match_ids = venue_matches['cricsheet_match_id'].tolist()
            venue_players = self.players_df[self.players_df['cricsheet_match_id'].isin(match_ids)]
            
            if len(venue_players) == 0:
                print(f"⚠️  No player data found for {venue}")
                return {}
            
            # Initialize containers for aggregation
            bf_batting_points = []  # Batting-first team batting points
            ch_bowling_points = []  # Chasing team bowling points  
            bf_total_points = []    # Batting-first team total points
            ch_total_points = []    # Chasing team total points
            
            # New features: Bat Dominance and Spin Dominance
            total_batting_points = []    # BAT + WK + AR_batting points (both teams)
            total_bowling_points = []    # BOWL + AR_bowling points (both teams)
            total_fantasy_points = []    # All fantasy points (both teams)
            spinner_bowling_points = []  # Points from spinners only
            pacer_bowling_points = []    # Points from pacers only
            
            # Process each match at this venue
            for _, match in venue_matches.iterrows():
                match_id = match['cricsheet_match_id']
                bf_team = match['batting_first_team']
                ch_team = match['chasing_team']
                
                if pd.isna(bf_team) or pd.isna(ch_team):
                    continue
                
                match_players = venue_players[venue_players['cricsheet_match_id'] == match_id]
                
                if len(match_players) == 0:
                    continue
                
                # 1. BATTING-FIRST TEAM BATTING POINTS
                bf_batsmen = match_players[
                    (match_players['team'] == bf_team) & 
                    (match_players['role'].isin(['BAT', 'WK', 'AR']))
                ]
                bf_batting_fp = bf_batsmen['fantasy_points'].sum()
                bf_batting_points.append(bf_batting_fp)
                
                # 2. CHASING TEAM BOWLING POINTS
                ch_bowlers = match_players[
                    (match_players['team'] == ch_team) & 
                    (match_players['overs_bowled'] > 0)
                ]
                ch_bowling_fp = ch_bowlers['fantasy_points'].sum()
                ch_bowling_points.append(ch_bowling_fp)
                
                # 3. TOTAL TEAM POINTS (for chase advantage calculation)
                bf_total = match_players[match_players['team'] == bf_team]['fantasy_points'].sum()
                ch_total = match_players[match_players['team'] == ch_team]['fantasy_points'].sum()
                bf_total_points.append(bf_total)
                ch_total_points.append(ch_total)
                
                # 4. BAT DOMINANCE AND SPIN DOMINANCE CALCULATIONS
                match_total_batting = 0  # BAT + WK + AR_batting for both teams
                match_total_bowling = 0  # BOWL + AR_bowling for both teams
                match_total_fantasy = 0  # All fantasy points for both teams
                match_spinner_points = 0  # Points from spinners only
                match_pacer_points = 0    # Points from pacers only
                
                # Process all players in this match
                for _, player in match_players.iterrows():
                    fantasy_pts = player['fantasy_points']
                    role = player['role']
                    balls_faced = player.get('balls_faced', 0)
                    overs_bowled = player.get('overs_bowled', 0.0)
                    bowling_style = player.get('bowling_style', '')
                    
                    match_total_fantasy += fantasy_pts
                    
                    if role == 'BAT' or role == 'WK':
                        # Pure batting points
                        match_total_batting += fantasy_pts
                    elif role == 'BOWL':
                        # Pure bowling points
                        match_total_bowling += fantasy_pts
                        
                        # Categorize by bowling style
                        if self.is_spinner(bowling_style):
                            match_spinner_points += fantasy_pts
                        elif self.is_pacer(bowling_style):
                            match_pacer_points += fantasy_pts
                    elif role == 'AR':
                        # Split AR points between batting and bowling
                        ar_batting, ar_bowling = self.split_ar_points(fantasy_pts, balls_faced, overs_bowled)
                        match_total_batting += ar_batting
                        match_total_bowling += ar_bowling
                        
                        # AR bowling points categorized by style
                        if self.is_spinner(bowling_style):
                            match_spinner_points += ar_bowling
                        elif self.is_pacer(bowling_style):
                            match_pacer_points += ar_bowling
                
                # Store match-level aggregates
                total_batting_points.append(match_total_batting)
                total_bowling_points.append(match_total_bowling)
                total_fantasy_points.append(match_total_fantasy)
                spinner_bowling_points.append(match_spinner_points)
                pacer_bowling_points.append(match_pacer_points)
                

                
            # Calculate aggregate statistics
            if not bf_batting_points or not ch_bowling_points:
                print(f"⚠️  Insufficient data for {venue}")
                return {}
            
            # Core venue features
            avg_bf_batting_points = np.mean(bf_batting_points)
            std_bf_batting_points = np.std(bf_batting_points)
            
            avg_ch_bowling_points = np.mean(ch_bowling_points)
            std_ch_bowling_points = np.std(ch_bowling_points)
            
            # Chase advantage (positive = chase team scores more on average)
            chase_advantages = [ch - bf for ch, bf in zip(ch_total_points, bf_total_points)]
            avg_chase_advantage = np.mean(chase_advantages)
            std_chase_advantage = np.std(chase_advantages)
            
            # Chase win rate proxy (percentage of matches where chase team scored more)
            chase_wins = sum(1 for adv in chase_advantages if adv > 0)
            chase_win_rate = chase_wins / len(chase_advantages)
            
            # Calculate Bat Dominance and Spin Dominance
            avg_bat_dominance = np.mean([batting / fantasy if fantasy > 0 else 0 
                                        for batting, fantasy in zip(total_batting_points, total_fantasy_points)])
            std_bat_dominance = np.std([batting / fantasy if fantasy > 0 else 0 
                                       for batting, fantasy in zip(total_batting_points, total_fantasy_points)])
            
            avg_spin_dominance = np.mean([spinner / bowling if bowling > 0 else 0 
                                         for spinner, bowling in zip(spinner_bowling_points, total_bowling_points)])
            std_spin_dominance = np.std([spinner / bowling if bowling > 0 else 0 
                                        for spinner, bowling in zip(spinner_bowling_points, total_bowling_points)])
            
            avg_pace_dominance = np.mean([pacer / bowling if bowling > 0 else 0 
                                         for pacer, bowling in zip(pacer_bowling_points, total_bowling_points)])
            

            
            # Venue type classification
            venue_type = self.classify_venue_type(avg_bf_batting_points, avg_ch_bowling_points, chase_win_rate)
            
            venue_stats = {
                'venue_name': venue,
                'total_matches': len(venue_matches),
                'data_quality': len(bf_batting_points) / len(venue_matches),  # Ratio of matches with complete data
                
                # Core Features: Batting Context
                'avg_bf_batting_points': round(avg_bf_batting_points, 2),
                'std_bf_batting_points': round(std_bf_batting_points, 2),
                
                # Core Features: Bowling Context  
                'avg_ch_bowling_points': round(avg_ch_bowling_points, 2),
                'std_ch_bowling_points': round(std_ch_bowling_points, 2),
                
                # Core Features: Chase Context
                'avg_chase_advantage': round(avg_chase_advantage, 2),
                'std_chase_advantage': round(std_chase_advantage, 2),
                'chase_win_rate': round(chase_win_rate, 3),
                
                # New Features: Bat and Spin Dominance
                'bat_dominance': round(avg_bat_dominance, 3),
                'bat_dominance_std': round(std_bat_dominance, 3),
                'spin_dominance': round(avg_spin_dominance, 3),
                'spin_dominance_std': round(std_spin_dominance, 3),
                'pace_dominance': round(avg_pace_dominance, 3),
                
                # Derived Features
                'venue_type': venue_type,
                'batting_volatility': round(std_bf_batting_points / avg_bf_batting_points, 3) if avg_bf_batting_points > 0 else 0,
                'bowling_volatility': round(std_ch_bowling_points / avg_ch_bowling_points, 3) if avg_ch_bowling_points > 0 else 0,
                
                # Raw data for debugging
                'sample_bf_batting_points': bf_batting_points[:5],
                'sample_ch_bowling_points': ch_bowling_points[:5],
                'sample_chase_advantages': chase_advantages[:5]
            }
            
            print(f"  ✅ {venue}: {venue_stats['total_matches']} matches")
            print(f"     Fantasy: BF Bat: {venue_stats['avg_bf_batting_points']:.1f}±{venue_stats['std_bf_batting_points']:.1f}, "
                  f"CH Bowl: {venue_stats['avg_ch_bowling_points']:.1f}±{venue_stats['std_ch_bowling_points']:.1f}, "
                  f"Chase Rate: {venue_stats['chase_win_rate']:.1%}")
            print(f"     Dominance: Bat: {venue_stats['bat_dominance']:.1%}, "
                  f"Spin: {venue_stats['spin_dominance']:.1%}, "
                  f"Pace: {venue_stats['pace_dominance']:.1%}")
            
            return venue_stats
            
        except Exception as e:
            print(f"❌ Error calculating stats for {venue}: {e}")
            return {}
    
    def classify_venue_type(self, avg_bf_batting: float, avg_ch_bowling: float, chase_rate: float) -> str:
        """Classify venue type based on statistical patterns"""
        
        # High batting points threshold (adjust based on your data)
        high_batting_threshold = 300  # Adjust based on your typical scores
        high_bowling_threshold = 200  # Adjust based on your typical bowling points
        
        if avg_bf_batting > high_batting_threshold:
            return "batting_paradise"
        elif avg_ch_bowling > high_bowling_threshold:
            return "bowling_friendly" 
        elif chase_rate > 0.6:
            return "chase_friendly"
        elif chase_rate < 0.4:
            return "defend_friendly"
        else:
            return "balanced"
    
    def save_venue_data(self, venue_stats: Dict, league: str, output_dir: str = None) -> str:
        """Save venue statistics to JSON file"""
        if output_dir is None:
            output_dir = str(self.data_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        safe_league = league.replace(" ", "_").replace("/", "_").lower()
        filename = f"{safe_league}_venue.json"
        filepath = os.path.join(output_dir, filename)
        
        # Add metadata
        output_data = {
            'metadata': {
                'league': league,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_venues': len(venue_stats),
                'script_version': '1.0',
                'description': 'Venue contextual features for fantasy cricket modeling',
                'features_included': [
                    'avg_bf_batting_points', 'std_bf_batting_points',
                    'avg_ch_bowling_points', 'std_ch_bowling_points', 
                    'avg_chase_advantage', 'std_chase_advantage',
                    'chase_win_rate', 'venue_type', 'bat_dominance',
                    'spin_dominance', 'pace_dominance'
                ]
            },
            'venues': venue_stats
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved venue data to: {filepath}")
        return filepath
    
    def generate_league_report(self, venue_stats: Dict, league: str) -> str:
        """Generate a summary report of venue characteristics"""
        report = []
        report.append(f"📊 VENUE ANALYSIS REPORT: {league}")
        report.append("=" * 60)
        
        if not venue_stats:
            report.append("❌ No venue data available")
            return "\n".join(report)
        
        # Overall statistics
        all_batting = [v['avg_bf_batting_points'] for v in venue_stats.values()]
        all_bowling = [v['avg_ch_bowling_points'] for v in venue_stats.values()]
        all_chase_rates = [v['chase_win_rate'] for v in venue_stats.values()]
        
        report.append(f"Total Venues Analyzed: {len(venue_stats)}")
        report.append(f"Average BF Batting Points: {np.mean(all_batting):.1f} (±{np.std(all_batting):.1f})")
        report.append(f"Average CH Bowling Points: {np.mean(all_bowling):.1f} (±{np.std(all_bowling):.1f})")
        report.append(f"Average Chase Win Rate: {np.mean(all_chase_rates):.1%}")
        report.append("")
        
        # Venue type distribution
        venue_types = {}
        for venue_data in venue_stats.values():
            vtype = venue_data['venue_type']
            venue_types[vtype] = venue_types.get(vtype, 0) + 1
        
        report.append("📈 VENUE TYPE DISTRIBUTION:")
        for vtype, count in sorted(venue_types.items()):
            percentage = count / len(venue_stats) * 100
            report.append(f"  {vtype}: {count} venues ({percentage:.1f}%)")
        report.append("")
        
        # Top venues by category
        report.append("🏆 TOP VENUES BY CATEGORY:")
        
        # Highest batting venues
        batting_venues = sorted(venue_stats.items(), key=lambda x: x[1]['avg_bf_batting_points'], reverse=True)[:3]
        report.append("  🏏 Highest Batting Points:")
        for venue, data in batting_venues:
            report.append(f"    {venue}: {data['avg_bf_batting_points']:.1f} points")
        
        # Highest bowling venues  
        bowling_venues = sorted(venue_stats.items(), key=lambda x: x[1]['avg_ch_bowling_points'], reverse=True)[:3]
        report.append("  🎳 Highest Bowling Points:")
        for venue, data in bowling_venues:
            report.append(f"    {venue}: {data['avg_ch_bowling_points']:.1f} points")
        
        # Most chase-friendly venues
        chase_venues = sorted(venue_stats.items(), key=lambda x: x[1]['chase_win_rate'], reverse=True)[:3]
        report.append("  🎯 Most Chase-Friendly:")
        for venue, data in chase_venues:
            report.append(f"    {venue}: {data['chase_win_rate']:.1%} chase success")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Generate venue contextual features for cricket modeling')
    parser.add_argument('--league', type=str, default=None,
                       help='Specific league to process (e.g., "CPL", "IPL"). If not provided, processes all leagues separately.')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing matches.csv and players.csv (default: database_join/)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for JSON files (default: same as data-dir)')
    parser.add_argument('--min-matches', type=int, default=1,
                       help='Minimum matches required at a venue to include it (default: 1)')
    parser.add_argument('--report', action='store_true',
                       help='Generate and display summary report')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = VenueContextGenerator(args.data_dir)
        
        # Determine leagues to process
        if args.league:
            leagues_to_process = [args.league]
        else:
            # Get unique leagues from data
            leagues_to_process = generator.matches_df['league'].unique()
            leagues_to_process = [league for league in leagues_to_process if pd.notna(league)]
            print(f"🏆 Found {len(leagues_to_process)} leagues: {list(leagues_to_process)}")
        
        # Process each league
        for league in leagues_to_process:
            print(f"\n🚀 Processing league: {league}")
            print("-" * 50)
            
            # Calculate venue statistics
            venue_stats = generator.calculate_venue_stats(league, args.min_matches)
            
            if not venue_stats:
                print(f"⚠️  No venue data generated for {league}")
                continue
            
            # Save to file
            output_file = generator.save_venue_data(venue_stats, league, args.output_dir)
            
            # Generate report if requested
            if args.report:
                report = generator.generate_league_report(venue_stats, league)
                print(f"\n{report}")
        
        print(f"\n🎉 Venue context generation completed!")
        print(f"📁 Output files saved to: {args.output_dir or args.data_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
