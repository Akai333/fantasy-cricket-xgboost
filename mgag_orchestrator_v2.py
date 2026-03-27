#!/usr/bin/env python3
"""
MGAG Orchestrator v2.0 - Complete Implementation
Implements the full Model-Guided Adaptive Generation strategy from MGAG by Gemini 2.5.txt
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elite_structure_generator import CPLEliteStructureGenerator
from fixed_model_interface import FixedModelInterface
from match_data_loader import MatchDataLoader

class MGAGSearchState:
    """
    Centralized state management for MGAG search
    Addresses the "Complex State Management" challenge
    """
    
    def __init__(self):
        # Core state
        self.iteration = 0
        self.team_pool = {}  # Hash -> Team data (for deduplication)
        self.structure_performance = defaultdict(dict)
        self.player_frequency = defaultdict(float)
        
        # Performance tracking
        self.total_teams_generated = 0
        self.total_unique_teams = 0
        self.stage1_survival_rate = 0.0
        self.best_p_elite_seen = 0.0
        
        # Circuit breaker metrics
        self.structure_diversity_history = []
        self.confidence_history = []
        self.convergence_warnings = 0
        
    def add_teams_batch(self, teams: List[Dict]) -> int:
        """Add teams to pool with deduplication"""
        added_count = 0
        
        for team in teams:
            # Create hash for deduplication
            team_hash = self._hash_team(team)
            
            if team_hash not in self.team_pool:
                self.team_pool[team_hash] = team
                added_count += 1
                
                # Update best score
                p_elite = team.get('p_elite', 0.0)
                if p_elite > self.best_p_elite_seen:
                    self.best_p_elite_seen = p_elite
        
        self.total_unique_teams = len(self.team_pool)
        return added_count
    
    def _hash_team(self, team: Dict) -> str:
        """Create unique hash for team (based on players + C/VC)"""
        players = sorted(team.get('players', []))
        captain_idx = team.get('captain_idx', 0)
        vc_idx = team.get('vc_idx', 1)
        return f"{'-'.join(players)}:{captain_idx}:{vc_idx}"
    
    def get_top_teams(self, n: int = 1000) -> List[Dict]:
        """Get top N teams by P_elite"""
        all_teams = list(self.team_pool.values())
        # Sort by P_elite (handle missing scores)
        all_teams.sort(key=lambda x: x.get('p_elite', 0.0), reverse=True)
        return all_teams[:n]
    
    def calculate_structure_diversity(self, top_n: int = 1000) -> float:
        """Calculate structure diversity in top teams"""
        top_teams = self.get_top_teams(top_n)
        if not top_teams:
            return 0.0
        
        structures = [team.get('structure', (0,0,0,0)) for team in top_teams]
        unique_structures = len(set(structures))
        return unique_structures
    
    def update_performance_metrics(self, teams: List[Dict]):
        """Update structure performance and player frequency"""
        
        # Update structure performance
        structure_stats = defaultdict(list)
        for team in teams:
            structure = team.get('structure', (0,0,0,0))
            p_elite = team.get('p_elite', 0.0)
            structure_stats[structure].append(p_elite)
        
        for structure, scores in structure_stats.items():
            self.structure_performance[structure] = {
                'count': len(scores),
                'max_p_elite': max(scores) if scores else 0.0,
                'avg_p_elite': np.mean(scores) if scores else 0.0,
                'std_p_elite': np.std(scores) if scores else 0.0
            }
        
        # Update player frequency (for high-performing teams)
        high_performing_teams = [t for t in teams if t.get('p_elite', 0.0) > 0.3]
        
        if high_performing_teams:
            all_players = []
            for team in high_performing_teams:
                all_players.extend(team.get('players', []))
            
            # Count player appearances
            player_counts = Counter(all_players)
            total_appearances = len(high_performing_teams)
            
            # Update frequencies
            for player, count in player_counts.items():
                self.player_frequency[player] = count / total_appearances


class MGAGOrchestrator:
    """
    Model-Guided Adaptive Generation Orchestrator
    Implements the complete MGAG strategy with circuit breakers
    """
    
    def __init__(self, time_budget: int = 600):  # 10 minutes
        self.time_budget = time_budget
        self.start_time = None
        
        # Components
        self.structure_generator = CPLEliteStructureGenerator()
        self.model_interface = FixedModelInterface()
        self.search_state = MGAGSearchState()
        
        # MGAG Parameters
        self.phase1_budget_ratio = 0.2    # 20% for broad exploration
        self.phase2_budget_ratio = 0.7    # 70% for iterative refinement  
        self.phase3_budget_ratio = 0.1    # 10% for final mutations
        
        # Circuit breaker thresholds
        self.min_structure_diversity = 5   # Minimum unique structures in top 1000
        self.min_confidence_teams = 50     # Minimum teams with P_elite > 0.3
        self.max_iterations = 4            # Maximum refinement iterations
        
    def run_full_mgag_pipeline(self, match_id: str, league: str = "cpl") -> Dict:
        """
        Execute the complete MGAG pipeline
        Returns final results and performance metrics
        """
        
        self.start_time = time.time()
        print(f"🚀 STARTING MGAG PIPELINE")
        print(f"=" * 60)
        print(f"   🎯 Match: {match_id}")
        print(f"   ⏱️  Time Budget: {self.time_budget}s")
        print(f"   🏏 League: {league.upper()}")
        
        try:
            # Step 1: Load models and match data
            squad_context = self._load_match_data(match_id)
            if not squad_context:
                return self._create_error_result("Failed to load match data")
            
            # Step 2: Phase 1 - Broad Exploration
            phase1_result = self._execute_phase1_exploration(squad_context)
            if not phase1_result:
                return self._create_error_result("Phase 1 failed")
            
            # Step 3: Phase 2 - Iterative Refinement
            phase2_result = self._execute_phase2_refinement(squad_context)
            
            # Step 4: Phase 3 - Final Selection
            phase3_result = self._execute_phase3_final_selection(squad_context)
            
            # Step 5: Generate final results
            return self._create_final_results(match_id)
            
        except Exception as e:
            print(f"💥 MGAG Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(f"Pipeline error: {e}")
    
    def _load_match_data(self, match_id: str) -> Dict:
        """Load match data and initialize models"""
        print(f"\n📊 PHASE 0: INITIALIZATION")
        print(f"-" * 30)
        
        # Load models
        print(f"🎯 Loading trained models...")
        self.model_interface.load_models("cpl")
        
        # Load match data
        print(f"📋 Loading match data for {match_id}...")
        loader = MatchDataLoader()
        squad_context = loader.get_real_match_data(match_id)
        
        if squad_context:
            bf_count = len(squad_context.get('batfirst_players', {}))
            ch_count = len(squad_context.get('chase_players', {}))
            print(f"   ✅ Squad loaded: {bf_count} vs {ch_count} players")
        
        return squad_context
    
    def _execute_phase1_exploration(self, squad_context: Dict) -> bool:
        """Phase 1: Broad Exploration (20% budget)"""
        print(f"\n🔍 PHASE 1: BROAD EXPLORATION")
        print(f"-" * 30)
        
        phase1_budget = int(self.structure_generator.TOTAL_BUDGET * self.phase1_budget_ratio)
        
        # Generate initial teams using elite structures
        print(f"🎯 Generating {phase1_budget:,} teams across all elite structures...")
        
        # Temporarily reduce budget for Phase 1
        original_budget = self.structure_generator.TOTAL_BUDGET
        self.structure_generator.TOTAL_BUDGET = phase1_budget
        
        try:
            initial_teams = self.structure_generator.generate_all_elite_teams(squad_context)
            
            # Restore original budget
            self.structure_generator.TOTAL_BUDGET = original_budget
            
            if not initial_teams:
                print("❌ No teams generated in Phase 1")
                return False
            
            print(f"✅ Generated {len(initial_teams):,} teams")
            
            # Apply Stage 1 Filter
            print(f"🔍 Applying Stage 1 Filter...")
            filtered_teams = self.model_interface.apply_stage1_filter(initial_teams, squad_context, filter_threshold=0.3)
            
            if not filtered_teams:
                print("❌ No teams survived Stage 1 filter")
                return False
            
            # Apply Stage 2 Ranker
            print(f"🎯 Applying Stage 2 Ranker...")
            ranked_teams = self.model_interface.apply_stage2_ranker(filtered_teams, squad_context)
            
            # Add to search state
            added_count = self.search_state.add_teams_batch(ranked_teams)
            self.search_state.update_performance_metrics(ranked_teams)
            
            print(f"✅ Phase 1 Complete:")
            print(f"   📊 Teams generated: {len(initial_teams):,}")
            print(f"   🔍 Stage 1 survivors: {len(filtered_teams):,}")
            print(f"   🎯 Stage 2 ranked: {len(ranked_teams):,}")
            print(f"   🏆 Unique teams added: {added_count:,}")
            print(f"   📈 Best P_elite: {self.search_state.best_p_elite_seen:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 failed: {e}")
            return False
    
    def _execute_phase2_refinement(self, squad_context: Dict) -> bool:
        """Phase 2: Iterative Refinement (70% budget)"""
        print(f"\n🔄 PHASE 2: ITERATIVE REFINEMENT")
        print(f"-" * 30)
        
        phase2_budget = int(self.structure_generator.TOTAL_BUDGET * self.phase2_budget_ratio)
        budget_per_iteration = phase2_budget // self.max_iterations
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n   🔄 ITERATION {iteration}/{self.max_iterations}")
            
            # Check time budget
            elapsed = time.time() - self.start_time
            remaining = self.time_budget - elapsed
            
            if remaining < 60:  # Less than 1 minute remaining
                print(f"   ⏰ Time budget low ({remaining:.0f}s), stopping iterations")
                break
            
            # Analyze current state
            diversity = self.search_state.calculate_structure_diversity()
            top_teams = self.search_state.get_top_teams(1000)
            high_confidence_count = len([t for t in top_teams if t.get('p_elite', 0.0) > 0.3])
            
            print(f"   📊 Current state:")
            print(f"      Structure diversity: {diversity}")
            print(f"      High confidence teams: {high_confidence_count}")
            print(f"      Best P_elite: {self.search_state.best_p_elite_seen:.4f}")
            
            # Circuit breaker checks
            should_fallback = self._check_circuit_breakers(diversity, high_confidence_count)
            
            if should_fallback:
                print(f"   ⚠️  Circuit breaker triggered - using fallback strategy")
                success = self._execute_fallback_generation(budget_per_iteration, squad_context)
            else:
                print(f"   ✅ Normal refinement mode")
                success = self._execute_guided_generation(budget_per_iteration, squad_context)
            
            if not success:
                print(f"   ❌ Iteration {iteration} failed")
                break
                
            # Update circuit breaker history
            self.search_state.structure_diversity_history.append(diversity)
            self.search_state.confidence_history.append(high_confidence_count)
        
        print(f"\n✅ Phase 2 Complete:")
        print(f"   🔄 Iterations completed: {iteration}")
        print(f"   🏆 Total unique teams: {self.search_state.total_unique_teams:,}")
        print(f"   📈 Best P_elite: {self.search_state.best_p_elite_seen:.4f}")
        
        return True
    
    def _execute_phase3_final_selection(self, squad_context: Dict) -> bool:
        """Phase 3: Final Selection (10% budget for mutations)"""
        print(f"\n🏆 PHASE 3: FINAL SELECTION")
        print(f"-" * 30)
        
        # Get top teams for final refinement
        top_teams = self.search_state.get_top_teams(100)
        
        if not top_teams:
            print("❌ No teams available for final selection")
            return False
        
        # Generate mutations of best teams
        phase3_budget = int(self.structure_generator.TOTAL_BUDGET * self.phase3_budget_ratio)
        print(f"🔧 Generating {phase3_budget:,} mutations of top teams...")
        
        mutation_teams = self._generate_team_mutations(top_teams[:20], phase3_budget, squad_context)
        
        if mutation_teams:
            # Filter and rank mutations
            filtered_mutations = self.model_interface.apply_stage1_filter(mutation_teams, squad_context, filter_threshold=0.2)
            
            if filtered_mutations:
                ranked_mutations = self.model_interface.apply_stage2_ranker(filtered_mutations, squad_context)
                added_count = self.search_state.add_teams_batch(ranked_mutations)
                
                print(f"✅ Mutations complete:")
                print(f"   🔧 Generated: {len(mutation_teams):,}")
                print(f"   🔍 Survived filter: {len(filtered_mutations):,}")
                print(f"   🏆 New unique teams: {added_count:,}")
        
        return True
    
    def _check_circuit_breakers(self, diversity: float, confidence_count: int) -> bool:
        """Check if circuit breaker conditions are met"""
        
        # Convergence check
        if diversity < self.min_structure_diversity:
            print(f"      ⚠️  Low diversity detected ({diversity} < {self.min_structure_diversity})")
            return True
        
        # Confidence check  
        if confidence_count < self.min_confidence_teams:
            print(f"      ⚠️  Low confidence detected ({confidence_count} < {self.min_confidence_teams})")
            return True
        
        return False
    
    def _execute_guided_generation(self, budget: int, squad_context: Dict) -> bool:
        """Execute model-guided generation focusing on high-performing patterns"""
        
        # Get top performing structures
        top_structures = []
        for structure, perf in self.search_state.structure_performance.items():
            if perf.get('count', 0) > 5:  # Only structures with enough samples
                avg_score = perf.get('avg_p_elite', 0.0)
                top_structures.append((structure, avg_score))
        
        top_structures.sort(key=lambda x: x[1], reverse=True)
        
        if not top_structures:
            return self._execute_fallback_generation(budget, squad_context)
        
        # Focus 80% budget on top 5 structures
        focused_structures = top_structures[:5]
        focused_budget = int(budget * 0.8)
        
        # Generate teams for focused structures
        teams_generated = []
        
        for i, (structure, score) in enumerate(focused_structures):
            structure_budget = focused_budget // len(focused_structures)
            print(f"      🎯 Focusing {structure_budget:,} teams on {structure} (avg: {score:.4f})")
            
            structure_teams = self.structure_generator.generate_teams_for_structure(
                structure, structure_budget, squad_context)
            teams_generated.extend(structure_teams)
        
        # Apply models
        if teams_generated:
            filtered_teams = self.model_interface.apply_stage1_filter(teams_generated, squad_context, filter_threshold=0.25)
            if filtered_teams:
                ranked_teams = self.model_interface.apply_stage2_ranker(filtered_teams, squad_context)
                added_count = self.search_state.add_teams_batch(ranked_teams)
                self.search_state.update_performance_metrics(ranked_teams)
                
                print(f"      ✅ Guided generation: {added_count:,} new teams added")
                return True
        
        return False
    
    def _execute_fallback_generation(self, budget: int, squad_context: Dict) -> bool:
        """Execute fallback generation with broader exploration"""
        print(f"      🔄 Fallback: Broad structure exploration")
        
        # Use original elite structure generator with reduced budget
        original_budget = self.structure_generator.TOTAL_BUDGET
        self.structure_generator.TOTAL_BUDGET = budget
        
        try:
            teams = self.structure_generator.generate_all_elite_teams(squad_context)
            
            if teams:
                filtered_teams = self.model_interface.apply_stage1_filter(teams, squad_context, filter_threshold=0.2)
                if filtered_teams:
                    ranked_teams = self.model_interface.apply_stage2_ranker(filtered_teams, squad_context)
                    added_count = self.search_state.add_teams_batch(ranked_teams)
                    self.search_state.update_performance_metrics(ranked_teams)
                    
                    print(f"      ✅ Fallback generation: {added_count:,} new teams added")
                    return True
        finally:
            self.structure_generator.TOTAL_BUDGET = original_budget
        
        return False
    
    def _generate_team_mutations(self, base_teams: List[Dict], budget: int, squad_context: Dict) -> List[Dict]:
        """Generate mutations of high-performing teams"""
        
        mutations = []
        
        # Get all available players
        all_players = {}
        all_players.update(squad_context.get('batfirst_players', {}))
        all_players.update(squad_context.get('chase_players', {}))
        
        players_by_role = defaultdict(list)
        for player_id, player_info in all_players.items():
            role = player_info.get('role', 'BAT')
            players_by_role[role].append(player_id)
        
        attempts = 0
        max_attempts = budget * 2
        
        while len(mutations) < budget and attempts < max_attempts:
            attempts += 1
            
            # Pick random base team
            base_team = np.random.choice(base_teams)
            base_players = base_team.get('players', [])
            
            if len(base_players) != 11:
                continue
            
            try:
                # Single player swap mutation
                mutated_players = base_players.copy()
                
                # Pick random position to mutate
                pos_to_change = np.random.randint(0, 11)
                old_player = base_players[pos_to_change]
                
                # Find old player's role
                old_player_role = None
                for role, players in players_by_role.items():
                    if old_player in players:
                        old_player_role = role
                        break
                
                if old_player_role and len(players_by_role[old_player_role]) > 1:
                    # Pick different player with same role
                    available_players = [p for p in players_by_role[old_player_role] if p != old_player]
                    if available_players:
                        new_player = np.random.choice(available_players)
                        mutated_players[pos_to_change] = new_player
                        
                        # Create mutated team
                        mutated_team = {
                            'players': mutated_players,
                            'captain_idx': np.random.randint(0, 11),
                            'vc_idx': np.random.randint(0, 11),
                            'structure': base_team.get('structure', (1, 3, 4, 3))
                        }
                        
                        # Ensure captain != vc
                        while mutated_team['vc_idx'] == mutated_team['captain_idx']:
                            mutated_team['vc_idx'] = np.random.randint(0, 11)
                        
                        mutations.append(mutated_team)
                        
            except (IndexError, ValueError):
                continue
        
        return mutations
    
    def _create_final_results(self, match_id: str) -> Dict:
        """Create final results summary"""
        
        elapsed_time = time.time() - self.start_time
        top_teams = self.search_state.get_top_teams(100)
        
        # Calculate elite discovery metrics
        elite_counts = {
            'very_high': len([t for t in top_teams if t.get('p_elite', 0.0) >= 0.5]),
            'high': len([t for t in top_teams if t.get('p_elite', 0.0) >= 0.3]),
            'medium': len([t for t in top_teams if t.get('p_elite', 0.0) >= 0.2]),
            'low': len([t for t in top_teams if t.get('p_elite', 0.0) >= 0.1]),
            'any_signal': len([t for t in top_teams if t.get('p_elite', 0.0) >= 0.05])
        }
        
        results = {
            'match_id': match_id,
            'success': True,
            'execution_time': elapsed_time,
            'total_unique_teams': self.search_state.total_unique_teams,
            'final_top_100': top_teams,
            'elite_discovery': elite_counts,
            'performance_metrics': {
                'best_p_elite': self.search_state.best_p_elite_seen,
                'avg_p_elite_top_100': np.mean([t.get('p_elite', 0.0) for t in top_teams]) if top_teams else 0.0,
                'structure_diversity': self.search_state.calculate_structure_diversity(100),
                'iterations_completed': len(self.search_state.structure_diversity_history),
                'circuit_breaker_triggers': self.search_state.convergence_warnings
            }
        }
        
        # Print final summary
        print(f"\n🏆 MGAG PIPELINE COMPLETED")
        print(f"=" * 60)
        print(f"   ⏱️  Total time: {elapsed_time:.1f}s")
        print(f"   🎯 Match: {match_id}")
        print(f"   🏆 Unique teams: {self.search_state.total_unique_teams:,}")
        print(f"   📈 Best P_elite: {self.search_state.best_p_elite_seen:.4f}")
        print(f"   📊 Elite discovery:")
        for level, count in elite_counts.items():
            print(f"      {level}: {count}")
        
        return results
    
    def _create_error_result(self, error_msg: str) -> Dict:
        """Create error result"""
        return {
            'success': False,
            'error': error_msg,
            'execution_time': time.time() - self.start_time if self.start_time else 0
        }


# Test function
def test_mgag_orchestrator():
    """Test the MGAG orchestrator on a historical match"""
    
    orchestrator = MGAGOrchestrator(time_budget=300)  # 5 minutes for testing
    result = orchestrator.run_full_mgag_pipeline("1351074", "cpl")
    
    return result


if __name__ == "__main__":
    print("🧪 Testing MGAG Orchestrator...")
    result = test_mgag_orchestrator()
    
    if result.get('success'):
        print(f"✅ Test successful!")
        print(f"📊 Elite discovery: {result['elite_discovery']}")
    else:
        print(f"❌ Test failed: {result.get('error', 'Unknown error')}")

