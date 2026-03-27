#!/usr/bin/env python3
"""
Smart MGAG Team Generator - ACTUALLY implements pattern-guided generation
Fixes the missing weighted player selection and smart generation logic
"""

from typing import Dict, List, Tuple, Any
import random
import numpy as np
from collections import defaultdict

class SmartMGAGGenerator:
    """
    Actually implements the smart generation that was missing from mgag_team_generator.py
    Uses pattern insights to guide player selection and structure focus
    """
    
    def __init__(self):
        self.exploit_ratio = 0.8  # 80% exploit, 20% explore
        self.mutation_probability = 0.3
        
    def generate_phase1_broad_exploration(self, squad_context: Dict, 
                                        budget: int = 200000) -> List[Dict]:
        """
        Phase 1: Proportional sampling with our improved structure priorities
        """
        print(f"🔍 SMART PHASE 1: Broad Exploration ({budget:,} teams)")
        
        # Use our enhanced elite structure generator
        from elite_structure_generator import CPLEliteStructureGenerator
        generator = CPLEliteStructureGenerator()
        generator.TOTAL_BUDGET = budget
        
        teams = generator.generate_all_elite_teams(squad_context)
        print(f"   ✅ Generated {len(teams):,} teams in Phase 1")
        return teams
        
    def generate_phase2_smart_refinement(self, patterns: Dict, 
                                       squad_context: Dict,
                                       search_state,
                                       budget: int = 120000) -> List[Dict]:
        """
        Phase 2: ACTUALLY smart generation based on learned patterns
        THIS is the missing smart logic!
        """
        print(f"🎯 SMART PHASE 2: Pattern-Guided Generation ({budget:,} teams)")
        
        # Extract pattern insights
        top_structures = patterns.get("top_structures", [])
        high_value_players = patterns.get("high_value_players", {})
        convergence_status = patterns.get("convergence_status", {})
        model_confidence = patterns.get("model_confidence", {})
        
        # Apply circuit breakers
        if convergence_status.get("is_converging", False):
            print("   ⚠️  Convergence detected - forcing exploration")
            return self._force_exploration_generation(squad_context, budget)
        
        if model_confidence.get("confidence") == "low":
            print("   ⚠️  Low model confidence - broad search")
            return self._broad_search_generation(squad_context, budget)
        
        # SMART adaptive generation
        return self._smart_adaptive_generation(top_structures, high_value_players, 
                                             squad_context, search_state, budget)
    
    def _smart_adaptive_generation(self, top_structures: List[Dict], 
                                 high_value_players: Dict[str, Dict],
                                 squad_context: Dict, search_state, budget: int) -> List[Dict]:
        """
        The CORE smart generation logic that was missing!
        Uses weighted player selection based on pattern insights
        """
        exploit_budget = int(budget * self.exploit_ratio)
        explore_budget = budget - exploit_budget
        
        print(f"   🎯 Smart Exploit: {exploit_budget:,} teams")
        print(f"   🔍 Explore: {explore_budget:,} teams")
        
        all_teams = []
        
        # EXPLOIT: Smart generation with pattern guidance
        exploit_teams = self._generate_smart_exploit_teams(
            top_structures, high_value_players, squad_context, exploit_budget
        )
        all_teams.extend(exploit_teams)
        
        # EXPLORE: Regular generation on unexplored structures
        explore_teams = self._generate_explore_teams(
            top_structures, squad_context, explore_budget
        )
        all_teams.extend(explore_teams)
        
        # MUTATE: Generate variants of best teams
        mutation_teams = self._generate_smart_mutations(
            search_state, squad_context, int(budget * 0.2)
        )
        all_teams.extend(mutation_teams)
        
        return all_teams
    
    def _generate_smart_exploit_teams(self, top_structures: List[Dict],
                                    high_value_players: Dict[str, Dict],
                                    squad_context: Dict, budget: int) -> List[Dict]:
        """
        SMART team generation using weighted player selection
        This is the key missing piece!
        """
        if not top_structures:
            return self._fallback_generation(squad_context, budget)
        
        print(f"      🧠 Using {len(high_value_players)} high-value players for guidance")
        
        # Focus on top 3 performing structures
        teams = []
        total_performance = sum(s.get("performance_score", 0) for s in top_structures[:3])
        
        if total_performance == 0:
            return self._fallback_generation(squad_context, budget)
        
        for struct_info in top_structures[:3]:
            structure = struct_info["structure"]
            performance_score = struct_info.get("performance_score", 0)
            
            # Allocate budget proportionally to performance
            structure_budget = int(budget * (performance_score / total_performance))
            if structure_budget < 1000:  # Minimum viable budget
                continue
            
            print(f"      📊 {structure}: {structure_budget:,} teams (perf={performance_score:.3f})")
            
            # Generate teams with SMART weighted selection
            structure_teams = self._generate_weighted_teams_for_structure(
                structure, squad_context, high_value_players, structure_budget
            )
            teams.extend(structure_teams)
        
        return teams
    
    def _generate_weighted_teams_for_structure(self, structure: Tuple, squad_context: Dict,
                                             high_value_players: Dict, budget: int) -> List[Dict]:
        """
        Generate teams for a structure using WEIGHTED player selection
        This is the core smart generation logic that was missing!
        """
        wk_needed, bat_needed, ar_needed, bowl_needed = structure
        
        # Get available players by role
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        players_by_role = defaultdict(list)
        for player_id, player_info in all_players.items():
            role = player_info.get('role', 'BAT')
            players_by_role[role].append(player_id)
        
        # Check if structure is viable
        if (len(players_by_role['WK']) < wk_needed or
            len(players_by_role['BAT']) < bat_needed or
            len(players_by_role['AR']) < ar_needed or
            len(players_by_role['BOWL']) < bowl_needed):
            print(f"         ⚠️  Structure {structure} not viable - insufficient players")
            return []
        
        # Calculate SMART weights for each role
        role_weights = {}
        for role in ['WK', 'BAT', 'AR', 'BOWL']:
            role_weights[role] = self._calculate_smart_player_weights(
                players_by_role[role], high_value_players, all_players
            )
        
        # Generate teams using weighted selection
        teams = []
        attempts = 0
        max_attempts = budget * 2  # Reasonable limit
        
        print(f"         🎯 Generating with smart weights...")
        
        while len(teams) < budget and attempts < max_attempts:
            attempts += 1
            
            try:
                # SMART weighted selection (this was missing!)
                selected_wk = self._smart_weighted_sample(
                    players_by_role['WK'], role_weights['WK'], wk_needed
                )
                selected_bat = self._smart_weighted_sample(
                    players_by_role['BAT'], role_weights['BAT'], bat_needed
                )
                selected_ar = self._smart_weighted_sample(
                    players_by_role['AR'], role_weights['AR'], ar_needed
                )
                selected_bowl = self._smart_weighted_sample(
                    players_by_role['BOWL'], role_weights['BOWL'], bowl_needed
                )
                
                team_players = selected_wk + selected_bat + selected_ar + selected_bowl
                
                # Validate team legality
                if self._is_team_legal(team_players, squad_context):
                    # Generate C/VC combinations with smart captain selection
                    cvc_teams = self._generate_smart_cvc_combinations(
                        team_players, structure, high_value_players
                    )
                    teams.extend(cvc_teams)
                    
            except (ValueError, IndexError) as e:
                continue
        
        print(f"         ✅ Generated {len(teams):,} teams for {structure}")
        return teams[:budget]
    
    def _calculate_smart_player_weights(self, role_players: List[str], 
                                      high_value_players: Dict, all_players: Dict) -> Dict[str, float]:
        """
        Calculate SMART selection weights based on pattern analysis
        High-value players get higher selection probability
        """
        weights = {}
        
        for player_id in role_players:
            base_weight = 1.0
            
            # Check if player is high-value from pattern analysis
            if player_id in high_value_players:
                player_stats = high_value_players[player_id]
                value_score = player_stats.get("value_score", 0.0)
                avg_p_elite = player_stats.get("avg_p_elite", 0.0)
                
                # Calculate smart weight: base + value_bonus + quality_bonus
                value_bonus = min(3.0, value_score * 10.0)  # Cap at 3x bonus
                quality_bonus = min(2.0, avg_p_elite * 5.0)  # Cap at 2x bonus
                
                weights[player_id] = base_weight + value_bonus + quality_bonus
                
            else:
                # Use player's fantasy average as proxy for quality
                player_info = all_players.get(player_id, {})
                fantasy_avg = player_info.get('avg_fantasy_points_last5', 25.0)
                
                # Normalize fantasy average to weight (25.0 = baseline)
                if fantasy_avg > 0:
                    quality_factor = min(2.0, fantasy_avg / 25.0)  # Cap at 2x
                    weights[player_id] = base_weight * quality_factor
                else:
                    weights[player_id] = base_weight
        
        return weights
    
    def _smart_weighted_sample(self, population: List[str], weights: Dict[str, float], 
                             k: int) -> List[str]:
        """
        SMART weighted sampling without replacement
        This ensures high-value players are selected more often
        """
        if len(population) <= k:
            return population.copy()
        
        # Create probability distribution
        weight_list = [weights.get(player_id, 1.0) for player_id in population]
        total_weight = sum(weight_list)
        
        if total_weight == 0:
            # Fallback to random if all weights are zero
            return random.sample(population, k)
        
        probabilities = [w / total_weight for w in weight_list]
        
        # Sample without replacement using numpy
        try:
            selected_indices = np.random.choice(
                len(population), size=k, replace=False, p=probabilities
            )
            return [population[i] for i in selected_indices]
        except ValueError:
            # Fallback if probability distribution is invalid
            return random.sample(population, k)
    
    def _generate_smart_cvc_combinations(self, team_players: List[str], structure: Tuple,
                                       high_value_players: Dict) -> List[Dict]:
        """
        Generate C/VC combinations with preference for high-value players as leaders
        """
        if len(team_players) != 11:
            return []
        
        teams = []
        
        # Calculate captain weights based on value scores
        captain_weights = {}
        for i, player_id in enumerate(team_players):
            if player_id in high_value_players:
                player_stats = high_value_players[player_id]
                # Captains get bonus from leadership history
                captain_count = player_stats.get("captain_count", 0)
                avg_p_elite = player_stats.get("avg_p_elite", 0.0)
                
                captain_weights[i] = 1.0 + (captain_count * 0.2) + (avg_p_elite * 2.0)
            else:
                captain_weights[i] = 1.0
        
        # Generate smart C/VC combinations (10 combinations per team structure)
        for _ in range(10):
            # Weighted captain selection
            captain_idx = self._weighted_choice(captain_weights)
            
            # Weighted VC selection (excluding captain)
            vc_weights = {i: w for i, w in captain_weights.items() if i != captain_idx}
            vc_idx = self._weighted_choice(vc_weights) if vc_weights else random.choice(
                [i for i in range(11) if i != captain_idx]
            )
            
            team = {
                'players': team_players.copy(),
                'player_ids': team_players.copy(),
                'captain_id': team_players[captain_idx],
                'vice_captain_id': team_players[vc_idx],
                'captain_idx': captain_idx,
                'vc_idx': vc_idx,
                'structure': structure
            }
            teams.append(team)
        
        return teams
    
    def _weighted_choice(self, weights: Dict[int, float]) -> int:
        """Choose index based on weights"""
        if not weights:
            return 0
        
        total = sum(weights.values())
        if total == 0:
            return random.choice(list(weights.keys()))
        
        r = random.uniform(0, total)
        cumulative = 0
        
        for idx, weight in weights.items():
            cumulative += weight
            if cumulative >= r:
                return idx
        
        return list(weights.keys())[-1]  # Fallback
    
    def _generate_smart_mutations(self, search_state, squad_context: Dict, budget: int) -> List[Dict]:
        """Generate smart mutations of best teams"""
        # Get top teams from search state
        top_teams = search_state.get_teams_by_score("p_elite", 20)
        
        if not top_teams:
            return []
        
        mutations = []
        budget_per_team = max(5, budget // len(top_teams))
        
        for team in top_teams:
            team_mutations = self._mutate_team_smart(team, squad_context, budget_per_team)
            mutations.extend(team_mutations)
        
        return mutations[:budget]
    
    def _mutate_team_smart(self, base_team: Dict, squad_context: Dict, budget: int) -> List[Dict]:
        """Smart single-player mutations"""
        base_players = base_team.get('players', base_team.get('player_ids', []))
        if len(base_players) != 11:
            return []
        
        mutations = []
        
        # Get all available players
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        # Try single player swaps
        for i, player_id in enumerate(base_players):
            if len(mutations) >= budget:
                break
            
            player_info = all_players.get(player_id, {})
            role = player_info.get('role', 'BAT')
            
            # Find alternatives for this role
            alternatives = [pid for pid, pinfo in all_players.items()
                          if pinfo.get('role') == role and pid != player_id]
            
            # Try swapping with top 3 alternatives (by fantasy average)
            alternatives_with_scores = []
            for alt_id in alternatives:
                alt_info = all_players.get(alt_id, {})
                fantasy_avg = alt_info.get('avg_fantasy_points_last5', 0)
                alternatives_with_scores.append((alt_id, fantasy_avg))
            
            # Sort by fantasy average (best first)
            alternatives_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            for alt_id, _ in alternatives_with_scores[:3]:  # Top 3 alternatives
                mutated_players = base_players.copy()
                mutated_players[i] = alt_id
                
                if self._is_team_legal(mutated_players, squad_context):
                    structure = base_team.get('structure', (1, 4, 3, 3))
                    cvc_teams = self._generate_cvc_combinations_simple(mutated_players, structure)
                    mutations.extend(cvc_teams[:2])  # 2 C/VC per mutation
                    
                    if len(mutations) >= budget:
                        break
        
        return mutations
    
    def _generate_cvc_combinations_simple(self, team_players: List[str], structure: Tuple) -> List[Dict]:
        """Simple C/VC generation for mutations"""
        if len(team_players) != 11:
            return []
        
        teams = []
        for _ in range(2):  # 2 combinations
            captain_idx = random.randint(0, 10)
            vc_idx = random.randint(0, 10)
            while vc_idx == captain_idx:
                vc_idx = random.randint(0, 10)
            
            teams.append({
                'players': team_players.copy(),
                'player_ids': team_players.copy(),
                'captain_id': team_players[captain_idx],
                'vice_captain_id': team_players[vc_idx],
                'captain_idx': captain_idx,
                'vc_idx': vc_idx,
                'structure': structure
            })
        
        return teams
    
    def _generate_explore_teams(self, top_structures: List[Dict], squad_context: Dict, budget: int) -> List[Dict]:
        """Generate teams from unexplored structures"""
        top_structure_set = set(s["structure"] for s in top_structures[:3])
        
        # Get all structures
        from elite_structure_generator import CPLEliteStructureGenerator
        generator = CPLEliteStructureGenerator()
        all_structures = [s[0] for s in generator.ELITE_STRUCTURES]
        
        # Filter to unexplored
        explore_structures = [s for s in all_structures if s not in top_structure_set]
        
        if not explore_structures:
            return []
        
        teams = []
        budget_per_structure = max(1000, budget // len(explore_structures))
        
        for structure in explore_structures:
            structure_teams = generator.generate_teams_for_structure(
                structure, budget_per_structure, squad_context
            )
            teams.extend(structure_teams)
        
        return teams[:budget]
    
    def _is_team_legal(self, team_players: List[str], squad_context: Dict) -> bool:
        """Check Dream11 legality"""
        batfirst_players = set(squad_context.get('batfirst_players', {}).keys())
        chase_players = set(squad_context.get('chase_players', {}).keys())
        
        team1_count = sum(1 for p in team_players if p in batfirst_players)
        team2_count = sum(1 for p in team_players if p in chase_players)
        
        return team1_count <= 7 and team2_count <= 7
    
    def _force_exploration_generation(self, squad_context: Dict, budget: int) -> List[Dict]:
        """Force exploration when converged"""
        return self._fallback_generation(squad_context, budget)
    
    def _broad_search_generation(self, squad_context: Dict, budget: int) -> List[Dict]:
        """Broad search when model confidence low"""
        return self._fallback_generation(squad_context, budget)
    
    def _fallback_generation(self, squad_context: Dict, budget: int) -> List[Dict]:
        """Fallback to basic generation"""
        from elite_structure_generator import CPLEliteStructureGenerator
        generator = CPLEliteStructureGenerator()
        generator.TOTAL_BUDGET = budget
        
        return generator.generate_all_elite_teams(squad_context)

