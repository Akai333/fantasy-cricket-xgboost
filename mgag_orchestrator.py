#!/usr/bin/env python3
"""
MGAG Orchestrator - Main controller for Model-Guided Adaptive Generation
Implements Gemini 2.5's complete MGAG strategy
"""

import time
from typing import Dict, List, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from search_state import SearchState
from pattern_analyzer import ElitePatternAnalyzer
from mgag_team_generator import MGAGTeamGenerator
from fixed_model_interface import FixedModelInterface

class MGAGOrchestrator:
    """
    Main orchestrator that implements Gemini's 3-phase MGAG strategy:
    Phase 1: Broad Exploration (1-2 minutes)
    Phase 2: Iterative Refinement (6-7 minutes) 
    Phase 3: Final Selection (1 minute)
    """
    
    def __init__(self, time_budget: int = 600):
        self.time_budget = time_budget  # 10 minutes
        self.phase1_time = 120  # 2 minutes
        self.phase3_time = 60   # 1 minute
        self.phase2_time = time_budget - self.phase1_time - self.phase3_time
        
        # Initialize components
        self.search_state = SearchState()
        self.pattern_analyzer = ElitePatternAnalyzer()
        self.team_generator = MGAGTeamGenerator()
        self.model_interface = FixedModelInterface()
        
        # Load models
        self.model_interface.load_models("cpl")
        
    def run_adaptive_search(self, squad_context: Dict, 
                          league: str = "cpl") -> Dict[str, Any]:
        """
        Execute the complete MGAG pipeline
        Returns final portfolio of top 100 teams
        """
        print("🚀 STARTING MGAG ADAPTIVE SEARCH")
        print("=" * 60)
        
        self.search_state.start_time = time.time()
        
        try:
            # Phase 1: Broad Exploration
            phase1_results = self._execute_phase1(squad_context)
            
            # Phase 2: Iterative Refinement (multiple loops)
            phase2_results = self._execute_phase2(squad_context)
            
            # Phase 3: Final Selection
            final_results = self._execute_phase3()
            
            # Compile final results
            results = {
                "search_state": self.search_state,
                "final_teams": final_results["top_100_teams"],
                "performance_summary": self._generate_performance_summary(),
                "phase1_results": phase1_results,
                "phase2_results": phase2_results,
                "phase3_results": final_results
            }
            
            return results
            
        except Exception as e:
            print(f"❌ MGAG search failed: {e}")
            return self._emergency_fallback(squad_context)
    
    def _execute_phase1(self, squad_context: Dict) -> Dict[str, Any]:
        """
        Phase 1: Broad Exploration - Proportional sampling across structures
        Time budget: ~2 minutes
        """
        print(f"\n📊 PHASE 1: BROAD EXPLORATION")
        print(f"⏱️  Time budget: {self.phase1_time}s")
        
        phase1_start = time.time()
        
        # Generate initial broad sample
        initial_teams = self.team_generator.generate_phase1_broad_exploration(
            squad_context, budget=150000
        )
        
        print(f"   📊 Generated {len(initial_teams):,} initial teams")
        
        # Add teams to search state (with deduplication)
        new_teams_added = self.search_state.add_teams_batch(initial_teams)
        print(f"   ✅ Added {new_teams_added:,} unique teams to search state")
        
        # Apply Stage 1 Filter (fast filtering)
        filtered_teams = self._apply_stage1_filter(list(self.search_state.team_pool.values()))
        print(f"   🎯 Stage 1 Filter: {len(filtered_teams):,} teams passed")
        
        # Update filter results in search state
        for team_hash, team in self.search_state.team_pool.items():
            team['stage1_passed'] = team_hash in [t['team_hash'] for t in filtered_teams]
        
        phase1_time = time.time() - phase1_start
        self.search_state.metrics["iteration_times"].append(phase1_time)
        
        return {
            "teams_generated": len(initial_teams),
            "unique_teams_added": new_teams_added,
            "teams_after_stage1": len(filtered_teams),
            "time_taken": phase1_time
        }
    
    def _execute_phase2(self, squad_context: Dict) -> Dict[str, Any]:
        """
        Phase 2: Iterative Refinement - Multiple loops of pattern learning
        Time budget: ~6-7 minutes
        """
        print(f"\n🎯 PHASE 2: ITERATIVE REFINEMENT")
        print(f"⏱️  Time budget: {self.phase2_time}s")
        
        phase2_start = time.time()
        iteration_results = []
        
        # Run iterative loops until time budget exhausted
        while (time.time() - phase2_start) < self.phase2_time:
            iteration_start = time.time()
            self.search_state.iteration += 1
            
            print(f"\n🔄 ITERATION {self.search_state.iteration}")
            
            # Get teams that passed Stage 1 filter
            filtered_teams = [team for team in self.search_state.team_pool.values()
                            if team.get('stage1_passed', False)]
            
            if not filtered_teams:
                print("   ⚠️  No teams passed Stage 1 filter - using fallback")
                break
            
            # Apply Stage 2 Ranker to get P_elite scores
            ranked_teams = self._apply_stage2_ranker(filtered_teams)
            print(f"   📈 Stage 2 Ranker: scored {len(ranked_teams):,} teams")
            
            # Update search state with scores
            self._update_search_state_scores(ranked_teams)
            
            # Analyze patterns from current results
            patterns = self.pattern_analyzer.analyze_patterns(self.search_state)
            
            # Check circuit breakers
            if patterns["convergence_status"]["is_converging"]:
                print("   ⚠️  Convergence detected!")
                self.search_state.convergence_warnings += 1
                
            if patterns["model_confidence"]["confidence"] == "low":
                print("   ⚠️  Low model confidence!")
                self.search_state.low_confidence_warnings += 1
            
            # Generate new teams based on patterns
            new_teams = self.team_generator.generate_phase2_iterative_refinement(
                patterns, squad_context, self.search_state, budget=100000
            )
            
            # Add new teams and filter them
            if new_teams:
                new_teams_added = self.search_state.add_teams_batch(new_teams)
                print(f"   ✅ Added {new_teams_added:,} new unique teams")
                
                # Apply Stage 1 filter to new teams only
                new_filtered = self._apply_stage1_filter(new_teams)
                print(f"   🎯 Stage 1 Filter: {len(new_filtered):,} new teams passed")
                
                # Update filter status for new teams
                new_team_hashes = set(self.search_state.hash_team(t.get('players', [])) for t in new_filtered)
                for team_hash, team in self.search_state.team_pool.items():
                    if team.get('added_iteration') == self.search_state.iteration:
                        team['stage1_passed'] = team_hash in new_team_hashes
            
            iteration_time = time.time() - iteration_start
            
            iteration_result = {
                "iteration": self.search_state.iteration,
                "new_teams_generated": len(new_teams) if new_teams else 0,
                "new_teams_added": new_teams_added if new_teams else 0,
                "patterns_extracted": len(patterns["top_structures"]),
                "convergence_warning": patterns["convergence_status"]["is_converging"],
                "time_taken": iteration_time
            }
            iteration_results.append(iteration_result)
            
            # Check if we should continue
            remaining_time = self.phase2_time - (time.time() - phase2_start)
            if remaining_time < 60:  # Need at least 1 minute for final phase
                print(f"   ⏱️  Stopping iterations - {remaining_time:.1f}s remaining")
                break
        
        phase2_time = time.time() - phase2_start
        
        return {
            "iterations_completed": len(iteration_results),
            "total_time": phase2_time,
            "iteration_details": iteration_results
        }
    
    def _execute_phase3(self) -> Dict[str, Any]:
        """
        Phase 3: Final Selection - Aggregate and rank all teams
        Time budget: ~1 minute
        """
        print(f"\n🏆 PHASE 3: FINAL SELECTION")
        print(f"⏱️  Time budget: {self.phase3_time}s")
        
        phase3_start = time.time()
        
        # Get all teams that passed Stage 1
        candidate_teams = [team for team in self.search_state.team_pool.values()
                         if team.get('stage1_passed', False)]
        
        print(f"   📊 Candidate teams: {len(candidate_teams):,}")
        
        # Ensure all teams have P_elite scores
        unscored_teams = [team for team in candidate_teams if 'p_elite' not in team]
        if unscored_teams:
            print(f"   🔄 Scoring {len(unscored_teams):,} remaining teams...")
            scored_teams = self._apply_stage2_ranker(unscored_teams)
            self._update_search_state_scores(scored_teams)
        
        # Get top 100 teams by P_elite
        top_100_teams = self.search_state.get_teams_by_score("p_elite", 100)
        
        # Apply ownership penalty (simplified - using default ownership)
        for team in top_100_teams:
            p_elite = team.get('p_elite', 0.0)
            ownership = 0.5  # Default 50% ownership
            team['ownership_adjusted_score'] = p_elite - 0.1 * ownership  # k=0.1 for small field
        
        # Re-sort by ownership-adjusted score
        top_100_teams.sort(key=lambda x: x.get('ownership_adjusted_score', 0), reverse=True)
        
        phase3_time = time.time() - phase3_start
        
        return {
            "top_100_teams": top_100_teams,
            "candidate_pool_size": len(candidate_teams),
            "time_taken": phase3_time
        }
    
    def _apply_stage1_filter(self, teams: List[Dict]) -> List[Dict]:
        """Apply Stage 1 global filter to teams"""
        if not teams:
            return []
        
        try:
            # Use model interface to apply Stage 1 filter
            squad_context = {}  # Will be passed by caller in real implementation
            results = self.model_interface.predict_stage1_batch(teams, squad_context)
            
            # Filter teams that pass threshold
            filtered_teams = []
            for team, prediction in zip(teams, results):
                if prediction >= 0.001:  # Low threshold for high recall
                    team['stage1_score'] = prediction
                    filtered_teams.append(team)
            
            return filtered_teams
            
        except Exception as e:
            print(f"   ⚠️  Stage 1 filter error: {e}")
            # Fallback: return all teams
            return teams
    
    def _apply_stage2_ranker(self, teams: List[Dict]) -> List[Dict]:
        """Apply Stage 2 ranker to get P_elite scores"""
        if not teams:
            return []
        
        try:
            squad_context = {}  # Will be passed by caller in real implementation
            results = self.model_interface.predict_stage2_batch(teams, squad_context)
            
            # Add P_elite scores to teams
            for team, prediction in zip(teams, results):
                team['p_elite'] = prediction
            
            return teams
            
        except Exception as e:
            print(f"   ⚠️  Stage 2 ranker error: {e}")
            # Fallback: assign random scores
            for team in teams:
                team['p_elite'] = 0.1  # Low default score
            return teams
    
    def _update_search_state_scores(self, scored_teams: List[Dict]):
        """Update search state with new scores"""
        for team in scored_teams:
            team_hash = team.get('team_hash')
            if team_hash and team_hash in self.search_state.team_pool:
                self.search_state.team_pool[team_hash].update({
                    'p_elite': team.get('p_elite'),
                    'stage1_score': team.get('stage1_score')
                })
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate summary of MGAG performance"""
        all_teams = list(self.search_state.team_pool.values())
        scored_teams = [t for t in all_teams if 'p_elite' in t]
        
        if not scored_teams:
            return {"error": "No teams were scored"}
        
        p_elite_scores = [t['p_elite'] for t in scored_teams]
        
        return {
            "total_iterations": self.search_state.iteration,
            "unique_teams_generated": len(all_teams),
            "teams_scored": len(scored_teams),
            "best_p_elite": max(p_elite_scores),
            "avg_p_elite": sum(p_elite_scores) / len(p_elite_scores),
            "convergence_warnings": self.search_state.convergence_warnings,
            "low_confidence_warnings": self.search_state.low_confidence_warnings,
            "structures_explored": len(self.search_state.structure_performance),
            "total_time": time.time() - self.search_state.start_time if self.search_state.start_time else 0
        }
    
    def _emergency_fallback(self, squad_context: Dict) -> Dict[str, Any]:
        """Emergency fallback if MGAG fails"""
        print("🚨 EMERGENCY FALLBACK - Using simple generation")
        
        try:
            from elite_structure_generator import CPLEliteStructureGenerator
            generator = CPLEliteStructureGenerator()
            fallback_teams = generator.generate_all_elite_teams(squad_context)
            
            return {
                "final_teams": fallback_teams[:100],
                "fallback_used": True,
                "performance_summary": {"error": "Emergency fallback used"}
            }
        except Exception as e:
            print(f"❌ Even fallback failed: {e}")
            return {
                "final_teams": [],
                "fallback_used": True,
                "error": str(e)
            }