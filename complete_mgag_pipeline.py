#!/usr/bin/env python3
"""
Complete MGAG Pipeline with Iterative Refinement and Team Mutations
Implements the full Model-Guided Adaptive Generation strategy
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import sys
import os
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elite_structure_generator import CPLEliteStructureGenerator
from fixed_model_interface import FixedModelInterface
from match_data_loader import MatchDataLoader

def process_team_batch_parallel(args):
    """Process a batch of teams in parallel"""
    teams_batch, squad_context, batch_id, filter_threshold = args
    
    try:
        model_interface = FixedModelInterface()
        model_interface.load_models("cpl")
        
        filtered_teams = model_interface.apply_stage1_filter(
            teams_batch, squad_context, filter_threshold=filter_threshold
        )
        
        return {
            'batch_id': batch_id,
            'filtered_teams': filtered_teams,
            'input_count': len(teams_batch),
            'output_count': len(filtered_teams)
        }
        
    except Exception as e:
        print(f"❌ Batch {batch_id} failed: {e}")
        return {'batch_id': batch_id, 'filtered_teams': [], 'input_count': len(teams_batch), 'output_count': 0}

class CompleteMGAGPipeline:
    """Complete MGAG Pipeline with iterative refinement and mutations"""
    
    def __init__(self, time_budget: int = 300):
        self.time_budget = time_budget
        self.start_time = None
        
        # MGAG Parameters
        self.max_iterations = 3
        self.num_processes = max(1, mp.cpu_count() - 1)
        self.batch_size = 1500
        
        # Progressive thresholds
        self.stage1_thresholds = [0.001, 0.002, 0.005]
        
        # Budget allocation
        self.phase1_budget = 30000
        self.iteration_budget = 20000
        self.mutation_budget = 15000
        
        # MGAG State
        self.elite_teams_pool = []
        self.structure_performance = defaultdict(list)
        self.high_value_players = Counter()
        
        print(f"🚀 Complete MGAG Pipeline initialized")
        print(f"   ⏱️  Time budget: {time_budget}s")
        print(f"   🔄 Max iterations: {self.max_iterations}")
        print(f"   💻 Parallel processes: {self.num_processes}")
    
    def run_complete_mgag(self, match_id: str = "1351074") -> Dict:
        """Run the complete MGAG pipeline with iterative refinement"""
        
        self.start_time = time.time()
        print(f"🚀 COMPLETE MGAG PIPELINE - ITERATIVE REFINEMENT")
        print(f"=" * 70)
        
        try:
            # Load match data
            squad_context = self._load_match_data(match_id)
            if not squad_context:
                return {'success': False, 'error': 'Failed to load match data'}
            
            # Phase 1: Initial Broad Exploration
            phase1_success = self._phase1_broad_exploration(squad_context)
            if not phase1_success:
                return {'success': False, 'error': 'Phase 1 failed'}
            
            # Phase 2: Iterative Refinement Loop
            for iteration in range(1, self.max_iterations + 1):
                elapsed = time.time() - self.start_time
                remaining = self.time_budget - elapsed
                
                if remaining < 60:
                    print(f"   ⏰ Time budget low ({remaining:.0f}s), stopping iterations")
                    break
                
                print(f"\n🔄 ITERATION {iteration}/{self.max_iterations}")
                print(f"=" * 50)
                
                self._execute_iteration(iteration, squad_context)
                self._print_iteration_summary()
            
            # Phase 3: Team Mutations
            self._phase3_team_mutations(squad_context)
            
            return self._create_final_results(match_id)
            
        except Exception as e:
            print(f"💥 Complete MGAG failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}