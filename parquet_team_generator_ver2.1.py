#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import json
import os
from datetime import datetime
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import ast
import uuid

class ParquetTeamGeneratorV2:
    """
    🚀 ENHANCED PARQUET-BASED Historical Team Generator v2.1
    
    ✅ SOFT LABELING: Teams with score ≥94% get continuous labels (0.94-1.00)
    ✅ OWNERSHIP COMPLETE: ownership (50%), cown (9%), vcown (9%) defaults
    ✅ SYSTEMATIC ELITE GENERATION: No swap limits, targets 94%+ systematically  
    ✅ SCORE-AWARE NON-ELITE: Checks non-elite scores, moves 94%+ to elite
    ✅ ALL TEAMS LEGAL: Guaranteed legal team generation
    ✅ ENHANCED FEATURES: Full squad context + ownership arrays
    """
    
    def __init__(self, output_dir: str = "R1_team_output"):
        self.output_dir = Path(output_dir)
        self.elite_dir = self.output_dir / "elite_teams"
        self.nonelite_dir = self.output_dir / "non_elite_teams"
        self.progress_file = self.output_dir / "progress.json"
        
        # Create directories
        self.elite_dir.mkdir(parents=True, exist_ok=True)
        self.nonelite_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CSV data
        print("📊 Loading CSV data...")
        self.matches_df = pd.read_csv("database_join/matches.csv")
        self.players_df = pd.read_csv("database_join/players.csv")
        
        # Convert complex columns
        self._convert_complex_columns()
        
        print(f"✅ Loaded {len(self.matches_df)} matches and {len(self.players_df)} players")
        print("🆕 Enhanced v2.1: Soft labeling + complete ownership + systematic elite generation")
        
    def _convert_complex_columns(self):
        """Convert string representations of lists/dicts to actual objects"""
        print("🔧 Converting complex data types...")
        
        # Convert bowling_phases from string to list
        def safe_eval_list(x):
            if pd.isna(x) or x == '' or x == '[]':
                return [0.0, 0.0, 0.0]
            try:
                return ast.literal_eval(x)
            except:
                return [0.0, 0.0, 0.0]
        
        # Convert bowling_phases_detail from string to dict
        def safe_eval_dict(x):
            if pd.isna(x) or x == '' or x == '{}':
                return {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
            try:
                return ast.literal_eval(x)
            except:
                return {"powerplay": 0.0, "middle": 0.0, "death": 0.0}
        
        # Convert last10_fantasy_scores from string to list
        def safe_eval_scores(x):
            if pd.isna(x) or x == '' or x == '[]':
                return []
            try:
                return ast.literal_eval(x)
            except:
                return []
        
        self.players_df['bowling_phases_list'] = self.players_df['bowling_phases'].apply(safe_eval_list)
        self.players_df['bowling_phases_detail_dict'] = self.players_df['bowling_phases_detail'].apply(safe_eval_dict)
        self.players_df['last10_fantasy_scores_list'] = self.players_df['last10_fantasy_scores'].apply(safe_eval_scores)
        
        # Fill missing values with appropriate defaults
        self.players_df['batting_order'] = self.players_df['batting_order'].fillna(11)
        self.players_df['fantasy_points'] = self.players_df['fantasy_points'].fillna(0.0)
        self.players_df['avg_fantasy_points_last5'] = self.players_df['avg_fantasy_points_last5'].fillna(0.0)
        self.players_df['avg_balls_faced_last5'] = self.players_df['avg_balls_faced_last5'].fillna(0.0)
        self.players_df['avg_overs_bowled_last5'] = self.players_df['avg_overs_bowled_last5'].fillna(0.0)
        
        # ENHANCED OWNERSHIP DEFAULTS - CORRECTED VALUES
        self.players_df['ownership'] = self.players_df['ownership'].fillna(0.5)    # 50% default
        self.players_df['cown'] = self.players_df.get('cown', pd.Series([np.nan] * len(self.players_df))).fillna(0.09)      # 9% captain ownership
        self.players_df['vcown'] = self.players_df.get('vcown', pd.Series([np.nan] * len(self.players_df))).fillna(0.09)    # 9% VC ownership
        
        # Add did_not_play column (fantasy_points > 0 means they played)
        self.players_df['did_not_play'] = (self.players_df['fantasy_points'] <= 0).astype(int)
        
        print("✅ Complex data types converted successfully")
        print("✅ Ownership defaults applied: ownership=50%, cown=9%, vcown=9%")
        
    def load_progress(self) -> Dict:
        """Load progress from checkpoint file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'processed_matches': [],
            'failed_matches': [],
            'current_batch': 0,
            'total_matches': len(self.matches_df),
            'elite_teams_count': 0,
            'nonelite_teams_count': 0,
            'start_time': None
        }
    
    def save_progress(self, progress: Dict):
        """Save progress to checkpoint file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def log_failed_match(self, match_id: str, reason: str, details: Optional[Dict] = None):
        """Log a match that failed to generate teams"""
        failed_match = {
            'match_id': match_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        # Load current progress
        progress = self.load_progress()
        
        # Add to failed matches if not already present
        existing_failed = [fm['match_id'] for fm in progress['failed_matches']]
        if match_id not in existing_failed:
            progress['failed_matches'].append(failed_match)
            self.save_progress(progress)
            print(f"📝 Logged failed match: {match_id} - {reason}")
    
    def get_failed_match_stats(self) -> Dict:
        """Get statistics about failed matches"""
        progress = self.load_progress()
        failed_matches = progress.get('failed_matches', [])
        
        if not failed_matches:
            return {
                'total_failed': 0,
                'failure_reasons': {},
                'failure_rate': 0.0
            }
        
        # Count failure reasons
        failure_reasons = {}
        for failed_match in failed_matches:
            reason = failed_match.get('reason', 'unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        total_matches = progress.get('total_matches', 0)
        failure_rate = (len(failed_matches) / total_matches * 100) if total_matches > 0 else 0.0
        
        return {
            'total_failed': len(failed_matches),
            'failure_reasons': failure_reasons,
            'failure_rate': failure_rate,
            'failed_matches': failed_matches
        }
    
    def print_failed_match_report(self):
        """Print a detailed report of failed matches"""
        stats = self.get_failed_match_stats()
        
        print("\n📊 FAILED MATCH REPORT")
        print("=" * 50)
        
        if stats['total_failed'] == 0:
            print("✅ No failed matches!")
            return
        
        print(f"❌ Total failed matches: {stats['total_failed']}")
        print(f"📈 Failure rate: {stats['failure_rate']:.2f}%")
        
        print("\n🔍 Failure reasons:")
        for reason, count in stats['failure_reasons'].items():
            print(f"  • {reason}: {count} matches")
        
        print("\n📋 Failed match details:")
        for failed_match in stats['failed_matches']:
            print(f"  • {failed_match['match_id']}: {failed_match['reason']}")
            if failed_match.get('details'):
                for key, value in failed_match['details'].items():
                    print(f"    - {key}: {value}")
        
        print("=" * 50)
    
    def generate_teams_for_match(self, match_id: str, match_number: int = 0, total_matches: int = 0) -> Tuple[List[Dict], List[Dict]]:
        """Generate teams for a single match with SOFT LABELING - returns elite and non-elite team data"""
        
        print(f"🔄 {match_number}/{total_matches} processing match: {match_id}")
        
        # Get match data
        match_data = self.matches_df[self.matches_df['cricsheet_match_id'] == match_id]
        if match_data.empty:
            print(f"❌ Match {match_id} not found")
            self.log_failed_match(match_id, "match_not_found", {"error": "Match ID not found in matches.csv"})
            return [], []
        
        match_row = match_data.iloc[0]
        
        # Get player data for this match
        players_match = self.players_df[self.players_df['cricsheet_match_id'] == match_id]
        if players_match.empty:
            print(f"❌ No players found for match {match_id}")
            self.log_failed_match(match_id, "no_players_found", {"error": "No players found for match in players.csv"})
            return [], []
        
        batting_first_team = match_row['batting_first_team']
        chasing_team = match_row['chasing_team']
        
        # Generate match max team using sophisticated 4-strategy approach
        match_max_team = self.generate_match_max_team(players_match)
        
        if not match_max_team:
            print(f"❌ Could not create match max team for {match_id}")
            
            # Gather diagnostic information
            eligible_players = players_match[players_match['did_not_play'] == 0]
            role_counts = {k: int(v) for k, v in eligible_players['role'].value_counts().items()} if len(eligible_players) > 0 else {}
            team_counts = {k: int(v) for k, v in eligible_players['team'].value_counts().items()} if len(eligible_players) > 0 else {}
            
            self.log_failed_match(match_id, "match_max_generation_failed", {
                "total_players": int(len(players_match)),
                "eligible_players": int(len(eligible_players)),
                "role_counts": role_counts,
                "team_counts": team_counts,
                "batting_first_team": str(batting_first_team),
                "chasing_team": str(chasing_team)
            })
            return [], []
        
        match_max_score = self.calculate_team_score(match_max_team, players_match)
        
        print(f"🎯 Match max score: {match_max_score:.1f}")
        
        # Generate soft elite teams (94%+ of max score) with NO SWAP LIMITS
        all_soft_elite_teams = self.generate_soft_elite_teams(match_max_team, players_match, match_max_score)
        
        # Create elite team records with soft labels
        elite_teams_data = []
        for team in all_soft_elite_teams:
            team_score = self.calculate_team_score(team, players_match)
            soft_label = min(team_score / match_max_score, 1.0)  # Continuous label
            
            elite_teams_data.append(self.create_team_record(
                match_row, team, players_match, batting_first_team, chasing_team, 
                is_elite=1, soft_label=soft_label
            ))
        
        # Generate diverse non-elite teams (target: 500 teams)
        initial_non_elite_teams = self.generate_diverse_non_elite_teams(
            players_match, match_row, batting_first_team, chasing_team, n_nonelite=500
        )
        
        # CHECK NON-ELITE SCORES: Move 94%+ teams to elite
        final_non_elite_teams = []
        additional_elite_teams = []
        
        for team in initial_non_elite_teams:
            team_score = self.calculate_team_score(team, players_match)
            score_percentage = team_score / match_max_score
            
            if score_percentage >= 0.94:
                # Move to elite with soft label
                soft_label = min(score_percentage, 1.0)
                additional_elite_teams.append(self.create_team_record(
                    match_row, team, players_match, batting_first_team, chasing_team,
                    is_elite=1, soft_label=soft_label
                ))
            else:
                # Keep as non-elite with label 0
                final_non_elite_teams.append(self.create_team_record(
                    match_row, team, players_match, batting_first_team, chasing_team,
                    is_elite=0, soft_label=0.0
                ))
        
        # Combine all elite teams
        all_elite_teams = elite_teams_data + additional_elite_teams
        
        print(f"✅ Generated {len(all_elite_teams)} soft elite teams (94%+) + {len(final_non_elite_teams)} non-elite teams")
        print(f"📊 Moved {len(additional_elite_teams)} teams from non-elite to elite due to 94%+ scores")
        
        return all_elite_teams, final_non_elite_teams
    
    def generate_match_max_team(self, players_df):
        """Generate the optimal team using 4-strategy approach"""
        
        # Strategy 1: Smart greedy selection
        team1 = self._find_max_team_smart_greedy(players_df)
        
        # Strategy 2: Role-balanced approach
        team2 = self._find_max_team_role_balanced(players_df)
        
        # Strategy 3: Captain-aware selection
        team3 = self._find_max_team_captain_aware(players_df)
        
        # Strategy 4: Local search optimization
        team4 = self._find_max_team_local_search(players_df)
        
        # Evaluate all strategies and return the best
        strategies = [team1, team2, team3, team4]
        best_team = None
        best_score = 0
        
        for team in strategies:
            if team and self.is_legal_team(team, players_df):
                score = self.calculate_team_score(team, players_df)
                if score > best_score:
                    best_score = score
                    best_team = team
        
        return best_team
    
    def generate_soft_elite_teams(self, match_max_team, players_df, match_max_score):
        """Generate systematic soft elite teams (94%+ of max score) with NO SWAP LIMITS"""
        
        elite_teams = [match_max_team]  # Start with match max
        eligible_players = players_df[players_df['did_not_play'] == 0]
        target_score = match_max_score * 0.94  # 94% threshold
        
        print(f"🎯 Generating soft elite teams with 94%+ threshold ({target_score:.1f}+)")
        
        # SYSTEMATIC APPROACH: Try different swap strategies
        max_teams = 100  # Target more elite teams
        
        # Strategy 1: Single role substitutions
        elite_teams.extend(self._generate_role_substitution_variants(
            match_max_team, players_df, eligible_players, target_score, max_swaps=1))
        
        # Strategy 2: Dual role substitutions  
        elite_teams.extend(self._generate_role_substitution_variants(
            match_max_team, players_df, eligible_players, target_score, max_swaps=2))
        
        # Strategy 3: Team balance variations
        elite_teams.extend(self._generate_team_balance_variants(
            match_max_team, players_df, eligible_players, target_score))
        
        # Strategy 4: Performance tier swaps (high->medium players)
        elite_teams.extend(self._generate_performance_tier_variants(
            match_max_team, players_df, eligible_players, target_score))
        
        # Strategy 5: Aggressive multi-swaps (up to 6 players if needed)
        if len(elite_teams) < max_teams:
            elite_teams.extend(self._generate_aggressive_swap_variants(
                match_max_team, players_df, eligible_players, target_score, max_swaps=6))
        
        # Deduplicate and validate
        unique_elite_teams = self._deduplicate_and_validate_elite(elite_teams, players_df, target_score)
        
        print(f"✅ Generated {len(unique_elite_teams)} unique soft elite teams")
        return unique_elite_teams[:max_teams]
    
    def _generate_role_substitution_variants(self, base_team, players_df, eligible_players, target_score, max_swaps=1):
        """Generate variants by substituting players within same roles"""
        variants = []
        
        for swap_count in range(1, max_swaps + 1):
            for attempt in range(50):  # Multiple attempts per swap count
                variant = base_team[:]
                
                # Pick random positions to swap
                swap_positions = random.sample(range(11), swap_count)
                
                for pos in swap_positions:
                    old_player = variant[pos]
                    old_player_data = players_df[players_df['player_id'] == old_player].iloc[0]
                    old_role = old_player_data['role']
                    
                    # Find replacement of same role
                    candidates = eligible_players[
                        (eligible_players['role'] == old_role) & 
                        (eligible_players['player_id'] != old_player) &
                        (~eligible_players['player_id'].isin(variant))
                    ]
                    
                    if not candidates.empty:
                        new_player = candidates.sample(1).iloc[0]['player_id']
                        variant[pos] = new_player
                
                # Check if variant meets criteria
                if (self.is_legal_team(variant, players_df) and
                    self.calculate_team_score(variant, players_df) >= target_score and
                    variant not in variants):
                    variants.append(variant)
        
        return variants
    
    def _generate_team_balance_variants(self, base_team, players_df, eligible_players, target_score):
        """Generate variants with different team balance ratios"""
        variants = []
        
        # Different team splits to try
        balance_targets = [(6, 5), (5, 6), (7, 4), (4, 7)]
        
        for target_split in balance_targets:
            for attempt in range(20):
                variant = self._rebuild_team_for_balance(base_team, players_df, eligible_players, target_split)
                
                if (variant and self.is_legal_team(variant, players_df) and
                    self.calculate_team_score(variant, players_df) >= target_score and
                    variant not in variants):
                    variants.append(variant)
        
        return variants
    
    def _rebuild_team_for_balance(self, base_team, players_df, eligible_players, target_split):
        """Rebuild team to achieve specific team balance"""
        team1_target, team2_target = target_split
        
        # Get the two teams involved
        base_team_data = players_df[players_df['player_id'].isin(base_team)]
        teams = base_team_data['team'].unique()
        
        if len(teams) < 2:
            return None
        
        team1, team2 = teams[0], teams[1]
        
        # Build new team with target balance
        new_team = []
        role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        team_counts = {team1: 0, team2: 0}
        
        # Prioritize high scorers from each team according to target split
        team1_players = eligible_players[eligible_players['team'] == team1].sort_values('fantasy_points', ascending=False)
        team2_players = eligible_players[eligible_players['team'] == team2].sort_values('fantasy_points', ascending=False)
        
        # First ensure minimum role requirements, then fill to balance
        all_candidates = pd.concat([team1_players, team2_players]).sort_values('fantasy_points', ascending=False)
        
        for _, player in all_candidates.iterrows():
            if len(new_team) >= 11:
                break
            
            if (player['player_id'] not in new_team and
                role_counts.get(player['role'], 0) < 8 and
                team_counts.get(player['team'], 0) < 7):
                
                new_team.append(player['player_id'])
                role_counts[player['role']] += 1
                team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
        
        return new_team if len(new_team) == 11 else None
    
    def _generate_performance_tier_variants(self, base_team, players_df, eligible_players, target_score):
        """Generate variants by swapping high performers with medium performers"""
        variants = []
        
        # Categorize players by performance tiers
        sorted_players = eligible_players.sort_values('fantasy_points', ascending=False)
        total_players = len(sorted_players)
        
        high_tier = sorted_players.iloc[:total_players//3]  # Top 33%
        medium_tier = sorted_players.iloc[total_players//3:2*total_players//3]  # Middle 33%
        
        for attempt in range(30):
            variant = base_team[:]
            
            # Find high tier players in current team
            variant_data = players_df[players_df['player_id'].isin(variant)]
            high_tier_in_team = variant_data[variant_data['player_id'].isin(high_tier['player_id'])]
            
            if not high_tier_in_team.empty:
                # Pick one high tier player to swap
                player_to_swap = high_tier_in_team.sample(1).iloc[0]
                swap_role = player_to_swap['role']
                swap_pos = variant.index(player_to_swap['player_id'])
                
                # Find medium tier replacement of same role
                medium_candidates = medium_tier[
                    (medium_tier['role'] == swap_role) &
                    (~medium_tier['player_id'].isin(variant))
                ]
                
                if not medium_candidates.empty:
                    replacement = medium_candidates.sample(1).iloc[0]['player_id']
                    variant[swap_pos] = replacement
                    
                    if (self.is_legal_team(variant, players_df) and
                        self.calculate_team_score(variant, players_df) >= target_score and
                        variant not in variants):
                        variants.append(variant)
        
        return variants
    
    def _generate_aggressive_swap_variants(self, base_team, players_df, eligible_players, target_score, max_swaps=6):
        """Generate variants with aggressive multi-player swaps"""
        variants = []
        
        for num_swaps in range(3, max_swaps + 1):
            for attempt in range(20):  # Fewer attempts for expensive operations
                variant = base_team[:]
                
                # Pick multiple positions to swap
                swap_positions = random.sample(range(11), num_swaps)
                
                # Perform all swaps
                for pos in swap_positions:
                    old_player = variant[pos]
                    old_player_data = players_df[players_df['player_id'] == old_player].iloc[0]
                    old_role = old_player_data['role']
                    
                    # Find any eligible replacement of same role
                    candidates = eligible_players[
                        (eligible_players['role'] == old_role) & 
                        (~eligible_players['player_id'].isin(variant))
                    ]
                    
                    if not candidates.empty:
                        new_player = candidates.sample(1).iloc[0]['player_id']
                        variant[pos] = new_player
                
                # Check if variant meets criteria
                if (self.is_legal_team(variant, players_df) and
                    self.calculate_team_score(variant, players_df) >= target_score and
                    variant not in variants):
                    variants.append(variant)
        
        return variants
    
    def _deduplicate_and_validate_elite(self, all_teams, players_df, target_score):
        """Deduplicate and validate elite teams"""
        unique_teams = []
        seen_teams = set()
        
        for team in all_teams:
            team_tuple = tuple(sorted(team))
            
            if (team_tuple not in seen_teams and 
                self.is_legal_team(team, players_df) and
                self.calculate_team_score(team, players_df) >= target_score):
                
                seen_teams.add(team_tuple)
                unique_teams.append(team)
        
        return unique_teams
    
    def _find_max_team_smart_greedy(self, players_df):
        """Smart greedy selection with constraints"""
        eligible_players = players_df[players_df['did_not_play'] == 0]
        
        if len(eligible_players) < 11:
            return None
        
        # Sort by fantasy points (descending)
        sorted_players = eligible_players.sort_values('fantasy_points', ascending=False)
        
        team = []
        role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        team_counts = {}
        
        for _, player in sorted_players.iterrows():
            if len(team) >= 11:
                break
            
            player_team = player['team']
            player_role = player['role']
            
            # Check constraints
            if (role_counts.get(player_role, 0) < 8 and
                team_counts.get(player_team, 0) < 7):
                
                team.append(player['player_id'])
                role_counts[player_role] += 1
                team_counts[player_team] = team_counts.get(player_team, 0) + 1
        
        return team if len(team) == 11 else None
    
    def _find_max_team_role_balanced(self, players_df):
        """Role-balanced team selection"""
        eligible_players = players_df[players_df['did_not_play'] == 0]
        
        team = []
        role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        team_counts = {}
        
        # Target: 1-2 WK, 3-5 BAT, 2-4 AR, 3-5 BOWL
        role_priorities = ['WK', 'BAT', 'AR', 'BOWL']
        
        for role in role_priorities:
            # Ensure at least 1 of each role
            if role_counts[role] == 0:
                role_players = eligible_players[eligible_players['role'] == role]
                role_players = role_players.sort_values('fantasy_points', ascending=False)
                
                for _, player in role_players.iterrows():
                    if (len(team) < 11 and
                        team_counts.get(player['team'], 0) < 7 and
                        player['player_id'] not in team):
                        
                        team.append(player['player_id'])
                        role_counts[role] += 1
                        team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
                        break
        
        # Fill remaining spots with best available players
        remaining_players = eligible_players[~eligible_players['player_id'].isin(team)]
        remaining_players = remaining_players.sort_values('fantasy_points', ascending=False)
        
        for _, player in remaining_players.iterrows():
            if len(team) >= 11:
                break
            
            if (role_counts.get(player['role'], 0) < 8 and
                team_counts.get(player['team'], 0) < 7):
                
                team.append(player['player_id'])
                role_counts[player['role']] += 1
                team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
        
        return team if len(team) == 11 else None
    
    def _find_max_team_captain_aware(self, players_df):
        """Captain-aware team selection"""
        eligible_players = players_df[players_df['did_not_play'] == 0]
        
        # Find potential captains (high fantasy points)
        potential_captains = eligible_players.nlargest(5, 'fantasy_points')
        
        best_team = None
        best_score = 0
        
        for _, captain in potential_captains.iterrows():
            # Build team around this captain
            team = self._build_team_around_captain(captain['player_id'], players_df, eligible_players)
            
            if team and self.is_legal_team(team, players_df):
                score = self.calculate_team_score(team, players_df)
                if score > best_score:
                    best_score = score
                    best_team = team
        
        return best_team
    
    def _build_team_around_captain(self, captain_id, players_df, eligible_players):
        """Build a legal team around a specific captain"""
        
        # Start with captain
        team = [captain_id]
        remaining_players = eligible_players[eligible_players['player_id'] != captain_id]
        
        # Get captain's team and role
        captain_data = players_df[players_df['player_id'] == captain_id].iloc[0]
        captain_team = captain_data['team']
        captain_role = captain_data['role']
        
        # Track constraints
        role_counts = {captain_role: 1, 'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        team_counts = {captain_team: 1}
        
        # Ensure at least one of each role
        for role in ['WK', 'BAT', 'AR', 'BOWL']:
            if role_counts.get(role, 0) == 0:
                role_players = remaining_players[remaining_players['role'] == role]
                role_players = role_players.sort_values('fantasy_points', ascending=False)
                
                for _, player in role_players.iterrows():
                    if (player['player_id'] not in team and
                        team_counts.get(player['team'], 0) < 7):
                        
                        team.append(player['player_id'])
                        role_counts[role] += 1
                        team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
                        break
        
        # Fill remaining spots with best available players
        remaining_players = remaining_players[~remaining_players['player_id'].isin(team)]
        remaining_players = remaining_players.sort_values('fantasy_points', ascending=False)
        
        for _, player in remaining_players.iterrows():
            if len(team) >= 11:
                break
            
            if (role_counts.get(player['role'], 0) < 8 and
                team_counts.get(player['team'], 0) < 7):
                
                team.append(player['player_id'])
                role_counts[player['role']] += 1
                team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
        
        return team if len(team) == 11 else None
    
    def _find_max_team_local_search(self, players_df):
        """Local search optimization starting from greedy solution"""
        
        # Start with greedy solution
        initial_team = self._find_max_team_smart_greedy(players_df)
        
        if not initial_team:
            return None
        
        # Perform local search optimization
        optimized_team = self._local_search_optimize(initial_team, players_df)
        
        return optimized_team
    
    def _local_search_optimize(self, initial_team, players_df):
        """Perform local search optimization"""
        
        current_team = initial_team[:]
        current_score = self.calculate_team_score(current_team, players_df)
        
        eligible_players = players_df[players_df['did_not_play'] == 0]
        out_players = eligible_players[~eligible_players['player_id'].isin(current_team)]
        
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try swapping each player in the team with players outside
            for i, in_player in enumerate(current_team):
                for _, out_player_row in out_players.iterrows():
                    out_player = out_player_row['player_id']
                    
                    # Try swap
                    test_team = current_team[:]
                    test_team[i] = out_player
                    
                    if self.is_legal_team(test_team, players_df):
                        test_score = self.calculate_team_score(test_team, players_df)
                        
                        if test_score > current_score:
                            current_team = test_team
                            current_score = test_score
                            improved = True
                            break
                
                if improved:
                    break
        
        return current_team
    
    def calculate_team_score(self, team, players_df):
        """Calculate total team score with captain/VC bonuses"""
        team_data = players_df[players_df['player_id'].isin(team)]
        
        if len(team_data) != 11:
            return 0
        
        # Calculate base score
        base_score = team_data['fantasy_points'].sum()
        
        # Apply captain and VC bonuses
        captain_id = team[0]  # First player is captain
        vc_id = team[1]       # Second player is vice-captain
        
        captain_points = team_data[team_data['player_id'] == captain_id]['fantasy_points'].iloc[0] if len(team_data[team_data['player_id'] == captain_id]) > 0 else 0
        vc_points = team_data[team_data['player_id'] == vc_id]['fantasy_points'].iloc[0] if len(team_data[team_data['player_id'] == vc_id]) > 0 else 0
        
        # Add bonuses: Captain gets 2x (so +1x bonus), VC gets 1.5x (so +0.5x bonus)
        captain_bonus = captain_points * 1.0  # Additional 1x (total becomes 2x)
        vc_bonus = vc_points * 0.5           # Additional 0.5x (total becomes 1.5x)
        
        total_score = base_score + captain_bonus + vc_bonus
        
        return total_score
    
    def is_legal_team(self, team, players_df):
        """Check if team satisfies all constraints"""
        if len(team) != 11 or len(set(team)) != 11:
            return False
        
        team_data = players_df[players_df['player_id'].isin(team)]
        if len(team_data) != 11:
            return False
        
        # Check role constraints (at least 1 of each role)
        role_counts = team_data['role'].value_counts()
        if any(role_counts.get(role, 0) == 0 for role in ['WK', 'BAT', 'AR', 'BOWL']):
            return False
        
        # Check role limits (max 8 of any role)
        if any(count > 8 for count in role_counts.values):
            return False
        
        # Check team balance (max 7 from either team)
        team_counts = team_data['team'].value_counts()
        if any(count > 7 for count in team_counts.values):
            return False
        
        return True
    
    def generate_diverse_non_elite_teams(self, players_df, match_data, batting_first_team, chasing_team, n_nonelite=500):
        """Generate diverse non-elite teams using systematic approach"""
        
        eligible_players = players_df[players_df['did_not_play'] == 0]['player_id'].tolist()
        
        if len(eligible_players) < 11:
            return []
        
        all_teams = []
        
        # Strategy 1: Captain diversity (125 teams)
        captain_teams = self._generate_captain_diverse_teams(players_df, eligible_players, 125)
        all_teams.extend(captain_teams)
        
        # Strategy 2: Structure diversity (125 teams)
        structure_teams = self._generate_structure_diverse_teams(players_df, eligible_players, 125)
        all_teams.extend(structure_teams)
        
        # Strategy 3: Balance diversity (125 teams)
        balance_teams = self._generate_balance_diverse_teams(players_df, eligible_players, 125)
        all_teams.extend(balance_teams)
        
        # Strategy 4: Random fill (125 teams)
        random_teams = self._generate_random_teams(players_df, eligible_players, 125)
        all_teams.extend(random_teams)
        
        # Deduplicate and validate
        unique_teams = self._deduplicate_and_validate(all_teams, players_df)
        
        # If we don't have enough, generate more random teams
        while len(unique_teams) < n_nonelite:
            additional_teams = self._generate_random_teams(players_df, eligible_players, n_nonelite - len(unique_teams))
            unique_teams.extend(self._deduplicate_and_validate(additional_teams, players_df))
            
            # Prevent infinite loop
            if len(additional_teams) == 0:
                break
        
        return unique_teams[:n_nonelite]
    
    def _generate_captain_diverse_teams(self, players_df, eligible_players, target_count):
        """Generate teams with diverse captains"""
        
        teams = []
        
        # Get top fantasy scorers as potential captains
        top_scorers = players_df[players_df['player_id'].isin(eligible_players)].nlargest(50, 'fantasy_points')
        
        for _, captain_row in top_scorers.iterrows():
            if len(teams) >= target_count:
                break
            
            captain_id = captain_row['player_id']
            
            # Build systematic team around captain
            team = self._build_legal_team_systematic(players_df, eligible_players, captain_id)
            
            if team and self.is_legal_team(team, players_df):
                teams.append(team)
        
        return teams
    
    def _build_legal_team_systematic(self, players_df, eligible_players, captain_id=None):
        """Build a legal team systematically"""
        
        # Start with captain or pick one
        if captain_id:
            team = [captain_id]
            remaining = [p for p in eligible_players if p != captain_id]
        else:
            team = []
            remaining = eligible_players[:]
        
        # Get all legal structures
        structures = [
            (1, 4, 3, 3),  # 1WK, 4BAT, 3AR, 3BOWL
            (1, 3, 4, 3),  # 1WK, 3BAT, 4AR, 3BOWL
            (1, 3, 3, 4),  # 1WK, 3BAT, 3AR, 4BOWL
            (1, 5, 2, 3),  # 1WK, 5BAT, 2AR, 3BOWL
            (1, 5, 3, 2),  # 1WK, 5BAT, 3AR, 2BOWL
            (2, 3, 3, 3),  # 2WK, 3BAT, 3AR, 3BOWL
            (2, 4, 2, 3),  # 2WK, 4BAT, 2AR, 3BOWL
            (2, 4, 3, 2),  # 2WK, 4BAT, 3AR, 2BOWL
        ]
        
        # Try each structure
        for wk_count, bat_count, ar_count, bowl_count in structures:
            test_team = team[:]
            test_remaining = remaining[:]
            
            role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
            team_counts = {}
            
            # Count existing players if captain provided
            if captain_id:
                captain_data = players_df[players_df['player_id'] == captain_id].iloc[0]
                role_counts[captain_data['role']] += 1
                team_counts[captain_data['team']] = 1
            
            # Fill each role to target count
            for role, needed in [('WK', wk_count), ('BAT', bat_count), ('AR', ar_count), ('BOWL', bowl_count)]:
                role_players = players_df[
                    (players_df['player_id'].isin(test_remaining)) & 
                    (players_df['role'] == role)
                ].sort_values('fantasy_points', ascending=False)
                
                for _, player in role_players.iterrows():
                    if (role_counts[role] < needed and
                        team_counts.get(player['team'], 0) < 7 and
                        len(test_team) < 11):
                        
                        test_team.append(player['player_id'])
                        test_remaining.remove(player['player_id'])
                        role_counts[role] += 1
                        team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
                        
                        if role_counts[role] >= needed:
                            break
            
            # If we have exactly 11 players and all constraints met
            if (len(test_team) == 11 and 
                all(role_counts[role] >= min_count for role, min_count in [('WK', 1), ('BAT', 1), ('AR', 1), ('BOWL', 1)])):
                return test_team
        
        return None
    
    def _generate_structure_diverse_teams(self, players_df, eligible_players, target_count):
        """Generate teams with diverse structures"""
        
        teams = []
        
        # All possible legal structures
        structures = [
            (1, 4, 3, 3), (1, 3, 4, 3), (1, 3, 3, 4), (1, 5, 2, 3), (1, 5, 3, 2),
            (1, 4, 2, 4), (1, 4, 4, 2), (1, 2, 4, 4), (1, 2, 5, 3), (1, 3, 5, 2),
            (2, 3, 3, 3), (2, 4, 2, 3), (2, 4, 3, 2), (2, 3, 2, 4), (2, 3, 4, 2),
            (2, 2, 3, 4), (2, 2, 4, 3), (2, 5, 2, 2), (2, 2, 5, 2), (2, 2, 2, 5),
        ]
        
        # Generate teams for each structure
        teams_per_structure = max(1, target_count // len(structures))
        
        for structure in structures:
            if len(teams) >= target_count:
                break
            
            for attempt in range(teams_per_structure * 3):  # 3 attempts per structure
                if len(teams) >= target_count:
                    break
                
                team = self._build_team_for_structure(structure, players_df, eligible_players)
                
                if team and self.is_legal_team(team, players_df):
                    teams.append(team)
                    
                    if len(teams) % teams_per_structure == 0:
                        break
        
        return teams
    
    def _build_team_for_structure(self, structure, players_df, eligible_players):
        """Build a team for a specific role structure"""
        wk_needed, bat_needed, ar_needed, bowl_needed = structure
        
        team = []
        role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        team_counts = {}
        
        # Fill each role up to the required count
        for role, needed in [('WK', wk_needed), ('BAT', bat_needed), ('AR', ar_needed), ('BOWL', bowl_needed)]:
            role_players = players_df[
                (players_df['player_id'].isin(eligible_players)) & 
                (players_df['role'] == role)
            ].sample(frac=1)  # Randomize order
            
            for _, player in role_players.iterrows():
                if (role_counts[role] < needed and
                    player['player_id'] not in team and
                    team_counts.get(player['team'], 0) < 7):
                    
                    team.append(player['player_id'])
                    role_counts[role] += 1
                    team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
                    
                    if role_counts[role] >= needed:
                        break
        
        return team if len(team) == 11 else None
    
    def _generate_balance_diverse_teams(self, players_df, eligible_players, target_count):
        """Generate teams with diverse team balance"""
        
        teams = []
        
        # Different team balance ratios
        balance_ratios = [(6, 5), (5, 6), (7, 4), (4, 7), (5, 4), (4, 5), (6, 3), (3, 6)]
        teams_per_ratio = max(1, target_count // len(balance_ratios))
        
        for ratio in balance_ratios:
            if len(teams) >= target_count:
                break
            
            for attempt in range(teams_per_ratio * 3):
                if len(teams) >= target_count:
                    break
                
                team = self._build_team_for_balance_ratio(ratio, players_df, eligible_players)
                
                if team and self.is_legal_team(team, players_df):
                    teams.append(team)
                    
                    if len(teams) % teams_per_ratio == 0:
                        break
        
        return teams
    
    def _build_team_for_balance_ratio(self, ratio, players_df, eligible_players):
        """Build a team for a specific team balance ratio"""
        team1_target, team2_target = ratio
        
        # Get available teams
        available_teams = players_df[players_df['player_id'].isin(eligible_players)]['team'].unique()
        if len(available_teams) < 2:
            return None
        
        # Pick two teams randomly
        selected_teams = random.sample(list(available_teams), 2)
        team1_name, team2_name = selected_teams
        
        team = []
        role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        team_counts = {team1_name: 0, team2_name: 0}
        
        # First ensure minimum role requirements
        for role in ['WK', 'BAT', 'AR', 'BOWL']:
            if role_counts[role] == 0:
                role_players = players_df[
                    (players_df['player_id'].isin(eligible_players)) & 
                    (players_df['role'] == role)
                ]
                
                for _, player in role_players.iterrows():
                    if (player['player_id'] not in team and
                        team_counts.get(player['team'], 0) < 7):
                        
                        team.append(player['player_id'])
                        role_counts[role] += 1
                        team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
                        break
        
        # Fill remaining spots trying to maintain ratio
        remaining_players = players_df[
            (players_df['player_id'].isin(eligible_players)) & 
            (~players_df['player_id'].isin(team))
        ]
        
        for _, player in remaining_players.iterrows():
            if len(team) >= 11:
                break
            
            if (role_counts.get(player['role'], 0) < 8 and
                team_counts.get(player['team'], 0) < 7):
                
                team.append(player['player_id'])
                role_counts[player['role']] += 1
                team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
        
        return team if len(team) == 11 else None
    
    def _generate_random_teams(self, players_df, eligible_players, target_count):
        """Generate random legal teams"""
        
        teams = []
        attempts = 0
        max_attempts = target_count * 10  # Allow more attempts for random generation
        
        while len(teams) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Random selection
            team = random.sample(eligible_players, 11)
            
            if self.is_legal_team(team, players_df):
                teams.append(team)
        
        return teams
    
    def _deduplicate_and_validate(self, all_teams, players_df):
        """Remove duplicates and validate all teams"""
        
        unique_teams = []
        seen_teams = set()
        
        for team in all_teams:
            # Sort team to create consistent representation
            team_tuple = tuple(sorted(team))
            
            if team_tuple not in seen_teams and self.is_legal_team(team, players_df):
                seen_teams.add(team_tuple)
                unique_teams.append(team)
        
        return unique_teams
    
    def extract_squad_data(self, players_match, team_name):
        """Extract squad data for a specific team (batting first or chasing)
        
        Returns pre-match feature arrays for all players from specified team who played
        """
        # Filter players from this team who actually played (did_not_play = 0)
        team_players = players_match[
            (players_match['team'] == team_name) & 
            (players_match['did_not_play'] == 0)
        ].copy()
        
        if team_players.empty:
            return {
                'squad_player_ids': [],
                'squad_names': [],
                'squad_roles': [],
                'squad_batting_styles': [],
                'squad_bowling_styles': [],
                'squad_batting_orders': [],
                'squad_avg_fantasy_points_last5': [],
                'squad_avg_balls_faced_last5': [],
                'squad_avg_overs_bowled_last5': [],
                'squad_last10_fantasy_scores': [],
                'squad_bowling_phases': [],
                'squad_ownership': [],
                'squad_cown': [],
                'squad_vcown': [],
                'squad_size': 0
            }
        
        # Sort by typical batting order for consistency (fallback to avg fantasy points)
        team_players = team_players.sort_values(['batting_order', 'avg_fantasy_points_last5'], 
                                               ascending=[True, False])
        
        # Extract pre-match squad features (NO POST-MATCH DATA)
        squad_data = {
            'squad_player_ids': team_players['player_id'].tolist(),
            'squad_names': team_players['name'].tolist(),
            'squad_roles': team_players['role'].tolist(),
            'squad_batting_styles': team_players['batting_style'].tolist(),
            'squad_bowling_styles': team_players['bowling_style'].tolist(),
            'squad_batting_orders': team_players['batting_order'].tolist(),
            'squad_avg_fantasy_points_last5': team_players['avg_fantasy_points_last5'].tolist(),
            'squad_avg_balls_faced_last5': team_players['avg_balls_faced_last5'].tolist(),
            'squad_avg_overs_bowled_last5': team_players['avg_overs_bowled_last5'].tolist(),
            'squad_last10_fantasy_scores': team_players['last10_fantasy_scores_list'].tolist(),
            'squad_bowling_phases': team_players['bowling_phases_list'].tolist(),
            'squad_ownership': team_players['ownership'].tolist(),
            'squad_cown': team_players['cown'].tolist(),
            'squad_vcown': team_players['vcown'].tolist(),
            'squad_size': len(team_players)
        }
        
        return squad_data
    
    def create_team_record(self, match_row, player_ids, players_df, batting_first_team, chasing_team, is_elite, soft_label):
        """Create a team record with all required fields including squad context data and SOFT LABELING"""
        
        team_data = players_df[players_df['player_id'].isin(player_ids)]
        
        # Create correct contextual template
        contextual_template = self._create_correct_contextual_template(team_data, batting_first_team, chasing_team)
        
        # Generate unique team UUID
        team_uuid = str(uuid.uuid4())
        
        # Extract squad context data for both teams
        batfirst_squad = self.extract_squad_data(players_df, batting_first_team)
        chase_squad = self.extract_squad_data(players_df, chasing_team)
        
        # Create enhanced record with squad context and SOFT LABELS
        record = {
            'team_uuid': team_uuid,
            'match_id': match_row['cricsheet_match_id'],
            'date': match_row['date'],
            'venue': match_row['venue'],
            'league': match_row['league'],
            'gender': match_row['gender'],
            'toss_winner': match_row['toss_winner'],
            'toss_decision': match_row['toss_decision'],
            'batting_first_team': batting_first_team,
            'chasing_team': chasing_team,
            'pitch': match_row['pitch'],
            'captain_id': player_ids[0],
            'vice_captain_id': player_ids[1],
            'contextual_template': contextual_template,
            
            # Selected team data (existing 11 players)
            'player_ids': player_ids,
            'roles': team_data['role'].tolist(),
            'team_ids': team_data['team'].tolist(),
            'batting_order_array': team_data['batting_order'].tolist(),
            'batting_style_array': team_data['batting_style'].tolist(),
            'bowling_style_array': team_data['bowling_style'].tolist(),
            'bowling_phases_array': team_data['bowling_phases_list'].tolist(),
            'ownership_array': team_data['ownership'].tolist(),
            'cown_array': team_data['cown'].tolist(),
            'vcown_array': team_data['vcown'].tolist(),
            'last10_fantasy_scores_array': team_data['last10_fantasy_scores_list'].tolist(),
            'avg_fantasy_points_last5_array': team_data['avg_fantasy_points_last5'].tolist(),
            'avg_balls_faced_last5_array': team_data['avg_balls_faced_last5'].tolist(),
            'avg_overs_bowled_last5_array': team_data['avg_overs_bowled_last5'].tolist(),
            
            # NEW: Batting First Team Squad Context (all available players)
            'batfirst_squad_player_ids': batfirst_squad['squad_player_ids'],
            'batfirst_squad_names': batfirst_squad['squad_names'],
            'batfirst_squad_roles': batfirst_squad['squad_roles'],
            'batfirst_squad_batting_styles': batfirst_squad['squad_batting_styles'],
            'batfirst_squad_bowling_styles': batfirst_squad['squad_bowling_styles'],
            'batfirst_squad_batting_orders': batfirst_squad['squad_batting_orders'],
            'batfirst_squad_avg_fantasy_points_last5': batfirst_squad['squad_avg_fantasy_points_last5'],
            'batfirst_squad_avg_balls_faced_last5': batfirst_squad['squad_avg_balls_faced_last5'],
            'batfirst_squad_avg_overs_bowled_last5': batfirst_squad['squad_avg_overs_bowled_last5'],
            'batfirst_squad_last10_fantasy_scores': batfirst_squad['squad_last10_fantasy_scores'],
            'batfirst_squad_bowling_phases': batfirst_squad['squad_bowling_phases'],
            'batfirst_squad_ownership': batfirst_squad['squad_ownership'],
            'batfirst_squad_cown': batfirst_squad['squad_cown'],
            'batfirst_squad_vcown': batfirst_squad['squad_vcown'],
            'batfirst_squad_size': batfirst_squad['squad_size'],
            
            # NEW: Chasing Team Squad Context (all available players)
            'chase_squad_player_ids': chase_squad['squad_player_ids'],
            'chase_squad_names': chase_squad['squad_names'],
            'chase_squad_roles': chase_squad['squad_roles'],
            'chase_squad_batting_styles': chase_squad['squad_batting_styles'],
            'chase_squad_bowling_styles': chase_squad['squad_bowling_styles'],
            'chase_squad_batting_orders': chase_squad['squad_batting_orders'],
            'chase_squad_avg_fantasy_points_last5': chase_squad['squad_avg_fantasy_points_last5'],
            'chase_squad_avg_balls_faced_last5': chase_squad['squad_avg_balls_faced_last5'],
            'chase_squad_avg_overs_bowled_last5': chase_squad['squad_avg_overs_bowled_last5'],
            'chase_squad_last10_fantasy_scores': chase_squad['squad_last10_fantasy_scores'],
            'chase_squad_bowling_phases': chase_squad['squad_bowling_phases'],
            'chase_squad_ownership': chase_squad['squad_ownership'],
            'chase_squad_cown': chase_squad['squad_cown'],
            'chase_squad_vcown': chase_squad['squad_vcown'],
            'chase_squad_size': chase_squad['squad_size'],
            
            # SOFT LABELING - CONTINUOUS LABELS
            'is_elite': is_elite,
            'soft_label': soft_label  # NEW: Continuous label (0.94-1.00 for elite, 0.0 for non-elite)
        }
        
        return record
    
    def _create_correct_contextual_template(self, team_data, batting_first_team, chasing_team):
        """Create the correct contextual template showing role distribution by team"""
        
        # Count players by role and team (batting first vs chasing)
        bat_first_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        chase_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        
        for _, player in team_data.iterrows():
            role = player['role']
            team = player['team']
            
            if team == batting_first_team:
                bat_first_counts[role] += 1
            elif team == chasing_team:
                chase_counts[role] += 1
        
        # Create correct format: BF_WK1_Ch_WK1__BF_BAT2_Ch_BAT1__BF_AR1_Ch_AR2__BF_Bowl1_Ch_Bowl2
        template = f"BF_WK{bat_first_counts['WK']}_Ch_WK{chase_counts['WK']}__"
        template += f"BF_BAT{bat_first_counts['BAT']}_Ch_BAT{chase_counts['BAT']}__"
        template += f"BF_AR{bat_first_counts['AR']}_Ch_AR{chase_counts['AR']}__"
        template += f"BF_Bowl{bat_first_counts['BOWL']}_Ch_Bowl{chase_counts['BOWL']}"
        
        return template
    
    def save_teams_to_parquet(self, elite_teams: List[Dict], non_elite_teams: List[Dict], batch_num: int):
        """Save teams to parquet files with proper data type preservation"""
        
        # Save elite teams
        if elite_teams:
            elite_df = pd.DataFrame(elite_teams)
            elite_file = self.elite_dir / f"batch_{batch_num:03d}_elite.parquet"
            elite_df.to_parquet(elite_file, compression='snappy', index=False)
            print(f"💾 Saved {len(elite_teams)} elite teams to {elite_file}")
        
        # Save non-elite teams
        if non_elite_teams:
            nonelite_df = pd.DataFrame(non_elite_teams)
            nonelite_file = self.nonelite_dir / f"batch_{batch_num:03d}_nonelite.parquet"
            nonelite_df.to_parquet(nonelite_file, compression='snappy', index=False)
            print(f"💾 Saved {len(non_elite_teams)} non-elite teams to {nonelite_file}")
    
    def process_matches_batch(self, match_ids: List[str], batch_num: int, starting_match_num: int, total_matches: int):
        """Process a batch of matches and save to parquet"""
        
        print(f"\n📦 Processing batch {batch_num} ({len(match_ids)} matches)")
        
        batch_elite_teams = []
        batch_non_elite_teams = []
        
        for i, match_id in enumerate(match_ids):
            match_number = starting_match_num + i
            
            elite_teams, non_elite_teams = self.generate_teams_for_match(
                match_id, match_number, total_matches
            )
            
            batch_elite_teams.extend(elite_teams)
            batch_non_elite_teams.extend(non_elite_teams)
        
        # Save to parquet files
        self.save_teams_to_parquet(batch_elite_teams, batch_non_elite_teams, batch_num)
        
        return len(batch_elite_teams), len(batch_non_elite_teams)
    
    def run_generation(self, batch_size: int = 100, n_workers: int = 6):
        """Run the complete team generation process"""
        
        print("🚀 ENHANCED PARQUET-BASED Historical Team Generator v2.1")
        print("=" * 70)
        print(f"🎯 CONFIGURATION:")
        print(f"   📦 Batch size: {batch_size} matches per batch")
        print(f"   🔄 Workers: {n_workers} parallel processes")
        print(f"   🎲 Target: ~500 non-elite teams per match")
        print(f"   💾 Output: R1_team_output compressed parquet files")
        print(f"   🔍 Legal teams: ALL teams guaranteed legal")
        print(f"   🆕 SOFT LABELING: Continuous labels for 94%+ teams")
        print(f"   📊 Complete ownership: ownership=50%, cown=9%, vcown=9%")
        print(f"   🎯 No swap limits: Up to 6 player swaps for elite generation")
        print("=" * 70)
        
        # Load progress
        progress = self.load_progress()
        
        # Get all match IDs
        all_match_ids = self.matches_df['cricsheet_match_id'].tolist()
        total_matches = len(all_match_ids)
        
        # Filter out already processed matches
        remaining_matches = [mid for mid in all_match_ids if mid not in progress['processed_matches']]
        
        if not remaining_matches:
            print("✅ All matches already processed!")
            return
        
        print(f"📋 Processing {len(remaining_matches)} remaining matches out of {total_matches} total")
        
        # Create batches
        batches = []
        for i in range(0, len(remaining_matches), batch_size):
            batch = remaining_matches[i:i+batch_size]
            batches.append(batch)
        
        print(f"📋 Created {len(batches)} batches")
        
        # Set start time if not already set
        if not progress['start_time']:
            progress['start_time'] = time.time()
        
        start_time = time.time()
        
        # Process batches with true multiprocessing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            print(f"🔄 Starting TRUE PARALLEL processing with {n_workers} workers...")
            
            # Submit all batches
            future_to_batch = {}
            for batch_idx, batch_matches in enumerate(batches):
                batch_num = progress['current_batch'] + batch_idx + 1
                starting_match_num = len(progress['processed_matches']) + batch_idx * batch_size + 1
                
                future = executor.submit(
                    process_batch_worker, 
                    batch_matches, 
                    batch_num, 
                    starting_match_num, 
                    total_matches,
                    self.output_dir
                )
                future_to_batch[future] = (batch_idx, batch_matches, batch_num)
            
            # Process completed batches
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_idx, batch_matches, batch_num = future_to_batch[future]
                
                try:
                    elite_count, nonelite_count = future.result()
                    
                    # Update progress
                    progress['processed_matches'].extend(batch_matches)
                    progress['current_batch'] = batch_num
                    progress['elite_teams_count'] += elite_count
                    progress['nonelite_teams_count'] += nonelite_count
                    
                    completed_batches += 1
                    
                    elapsed = time.time() - start_time
                    progress_pct = (completed_batches / len(batches)) * 100
                    eta = (elapsed / completed_batches) * (len(batches) - completed_batches)
                    
                    print(f"✅ Batch {batch_num} completed ({progress_pct:.1f}%)")
                    print(f"   Elite: +{elite_count:,}, Non-elite: +{nonelite_count:,}")
                    print(f"   Total: {progress['elite_teams_count']:,} elite, {progress['nonelite_teams_count']:,} non-elite")
                    print(f"   Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
                    
                    # Save progress every 3 batches
                    if completed_batches % 3 == 0:
                        self.save_progress(progress)
                        print(f"📍 CHECKPOINT {completed_batches}: Progress saved")
                    
                except Exception as e:
                    print(f"❌ Batch {batch_num} failed: {e}")
        
        # Final statistics
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("🎉 ENHANCED PARQUET GENERATION COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"📊 Elite teams: {progress['elite_teams_count']:,}")
        print(f"📊 Non-elite teams: {progress['nonelite_teams_count']:,}")
        print(f"📊 Total teams: {progress['elite_teams_count'] + progress['nonelite_teams_count']:,}")
        print(f"💾 Teams per second: {(progress['elite_teams_count'] + progress['nonelite_teams_count'])/total_time:.1f}")
        print(f"🆕 SOFT LABELING: Continuous labels for enhanced ranking training")
        print(f"📊 Complete ownership: All fields captured for production")
        
        # Save final progress
        self.save_progress(progress)
        
        # Show failed match report
        self.print_failed_match_report()
        
        # Show upload instructions
        print(f"\n📤 R1 OUTPUT READY:")
        print(f"   Elite teams: {len(list(self.elite_dir.glob('*.parquet')))} parquet files")
        print(f"   Non-elite teams: {len(list(self.nonelite_dir.glob('*.parquet')))} parquet files")
        print(f"   📂 Output directory: {self.output_dir}")


def process_batch_worker(match_ids: List[str], batch_num: int, starting_match_num: int, total_matches: int, output_dir: Path):
    """Worker function for processing a batch of matches"""
    
    # Create a new generator instance for this worker
    generator = ParquetTeamGeneratorV2(str(output_dir))
    
    # Process the batch
    return generator.process_matches_batch(match_ids, batch_num, starting_match_num, total_matches)


if __name__ == "__main__":
    # Create generator and run with optimal settings
    generator = ParquetTeamGeneratorV2()
    
    # Run generation with true multiprocessing
    generator.run_generation(
        batch_size=100,    # 100 matches per batch  
        n_workers=6        # Use all 6 cores
    )