#!/usr/bin/env python3
"""
Comprehensive Team Validator for MGAG
Ensures all generated teams are completely legal according to Dream11 rules
"""

from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TeamValidator:
    """Validates teams according to all Dream11 rules"""
    
    def __init__(self):
        self.validation_stats = {
            'total_checked': 0,
            'valid_teams': 0,
            'invalid_teams': 0,
            'error_counts': {}
        }
    
    def validate_team(self, team: Dict, squad_context: Dict) -> Tuple[bool, List[str]]:
        """
        Comprehensive team validation
        
        Returns:
            (is_valid, list_of_errors)
        """
        self.validation_stats['total_checked'] += 1
        errors = []
        
        players = team.get('players', [])
        captain_idx = team.get('captain_idx', -1)
        vc_idx = team.get('vc_idx', -1)
        structure = team.get('structure', (0, 0, 0, 0))
        
        # Rule 1: Exactly 11 players
        if len(players) != 11:
            errors.append(f"Invalid player count: {len(players)} (must be 11)")
        
        # Rule 2: All players must be unique
        if len(set(players)) != len(players):
            errors.append("Duplicate players in team")
        
        # Rule 3: Valid captain and vice-captain indices
        if not (0 <= captain_idx < len(players)):
            errors.append(f"Invalid captain index: {captain_idx}")
        
        if not (0 <= vc_idx < len(players)):
            errors.append(f"Invalid vice-captain index: {vc_idx}")
        
        if captain_idx == vc_idx:
            errors.append("Captain and vice-captain cannot be the same player")
        
        # Rule 4: All players must exist in squad context
        batfirst_players = squad_context.get('batfirst_players', {})
        chase_players = squad_context.get('chase_players', {})
        all_available_players = set(batfirst_players.keys()) | set(chase_players.keys())
        
        for player_id in players:
            if player_id not in all_available_players:
                errors.append(f"Player {player_id} not in available squads")
        
        # Rule 5: Maximum 7 players from one team
        bf_count = sum(1 for p in players if p in batfirst_players)
        chase_count = len(players) - bf_count
        
        if bf_count > 7:
            errors.append(f"Too many players from batting first team: {bf_count} (max 7)")
        
        if chase_count > 7:
            errors.append(f"Too many players from chasing team: {chase_count} (max 7)")
        
        # Rule 6: Minimum 1 player from each team
        if bf_count < 1:
            errors.append("Must have at least 1 player from batting first team")
        
        if chase_count < 1:
            errors.append("Must have at least 1 player from chasing team")
        
        # Rule 7: Role distribution validation
        role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
        
        for player_id in players:
            if player_id in batfirst_players:
                role = batfirst_players[player_id].get('role', 'UNKNOWN')
            else:
                role = chase_players[player_id].get('role', 'UNKNOWN')
            
            if role in role_counts:
                role_counts[role] += 1
            else:
                errors.append(f"Unknown role for player {player_id}: {role}")
        
        # Rule 8: Structure validation
        expected_wk, expected_bat, expected_ar, expected_bowl = structure
        
        if role_counts['WK'] != expected_wk:
            errors.append(f"WK count mismatch: {role_counts['WK']} (expected {expected_wk})")
        
        if role_counts['BAT'] != expected_bat:
            errors.append(f"BAT count mismatch: {role_counts['BAT']} (expected {expected_bat})")
        
        if role_counts['AR'] != expected_ar:
            errors.append(f"AR count mismatch: {role_counts['AR']} (expected {expected_ar})")
        
        if role_counts['BOWL'] != expected_bowl:
            errors.append(f"BOWL count mismatch: {role_counts['BOWL']} (expected {expected_bowl})")
        
        # Rule 9: Minimum role requirements (Dream11 rules)
        if role_counts['WK'] < 1:
            errors.append("Must have at least 1 wicket-keeper")
        
        if role_counts['WK'] > 4:
            errors.append(f"Too many wicket-keepers: {role_counts['WK']} (max 4)")
        
        if role_counts['BAT'] + role_counts['AR'] < 3:
            errors.append("Must have at least 3 batsmen (BAT + AR)")
        
        if role_counts['BOWL'] + role_counts['AR'] < 3:
            errors.append("Must have at least 3 bowlers (BOWL + AR)")
        
        # Rule 10: Maximum role limits
        if role_counts['BAT'] > 6:
            errors.append(f"Too many batsmen: {role_counts['BAT']} (max 6)")
        
        if role_counts['BOWL'] > 6:
            errors.append(f"Too many bowlers: {role_counts['BOWL']} (max 6)")
        
        if role_counts['AR'] > 4:
            errors.append(f"Too many all-rounders: {role_counts['AR']} (max 4)")
        
        # Update statistics
        is_valid = len(errors) == 0
        
        if is_valid:
            self.validation_stats['valid_teams'] += 1
        else:
            self.validation_stats['invalid_teams'] += 1
            for error in errors:
                error_type = error.split(':')[0]
                self.validation_stats['error_counts'][error_type] = \
                    self.validation_stats['error_counts'].get(error_type, 0) + 1
        
        return is_valid, errors
    
    def validate_team_batch(self, teams: List[Dict], squad_context: Dict) -> List[Dict]:
        """
        Validate a batch of teams and return only valid ones
        
        Args:
            teams: List of teams to validate
            squad_context: Squad context with player information
            
        Returns:
            List of valid teams only
        """
        print(f"🔍 Validating {len(teams):,} teams for legality...")
        
        valid_teams = []
        invalid_count = 0
        
        for i, team in enumerate(teams):
            is_valid, errors = self.validate_team(team, squad_context)
            
            if is_valid:
                valid_teams.append(team)
            else:
                invalid_count += 1
                
                # Show first few validation errors for debugging
                if invalid_count <= 5:
                    print(f"   ⚠️  Team {i+1} invalid: {errors[0]}")
        
        validation_rate = len(valid_teams) / len(teams) * 100 if teams else 0
        
        print(f"   ✅ Validation complete: {len(valid_teams):,}/{len(teams):,} valid ({validation_rate:.1f}%)")
        
        if invalid_count > 5:
            print(f"   📊 {invalid_count - 5} more invalid teams (not shown)")
        
        return valid_teams
    
    def get_validation_report(self) -> str:
        """Generate a validation report"""
        stats = self.validation_stats
        
        if stats['total_checked'] == 0:
            return "No teams validated yet"
        
        success_rate = stats['valid_teams'] / stats['total_checked'] * 100
        
        report = f"""
🔍 TEAM VALIDATION REPORT:
   📊 Total teams checked: {stats['total_checked']:,}
   ✅ Valid teams: {stats['valid_teams']:,}
   ❌ Invalid teams: {stats['invalid_teams']:,}
   📈 Success rate: {success_rate:.1f}%
   
🚨 Top validation errors:
"""
        
        # Sort errors by frequency
        sorted_errors = sorted(stats['error_counts'].items(), 
                             key=lambda x: x[1], reverse=True)
        
        for error_type, count in sorted_errors[:5]:
            report += f"   - {error_type}: {count} teams\n"
        
        return report
    
    def print_validation_summary(self):
        """Print validation summary"""
        print(self.get_validation_report())


# Helper function for easy validation
def validate_teams(teams: List[Dict], squad_context: Dict) -> List[Dict]:
    """Quick validation function"""
    validator = TeamValidator()
    valid_teams = validator.validate_team_batch(teams, squad_context)
    validator.print_validation_summary()
    return valid_teams


if __name__ == "__main__":
    print("🔍 Team Validator Module")
    print("This module validates teams according to Dream11 rules")

