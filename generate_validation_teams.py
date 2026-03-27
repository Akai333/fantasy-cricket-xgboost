#!/usr/bin/env python3
"""
Generate teams using elite structures only - bypass XGBoost filtering for manual validation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_team_generator import LiveTeamGenerator
from datetime import datetime
import pandas as pd
from pathlib import Path

def generate_elite_structure_teams():
    """Generate teams using elite structures for manual score validation"""
    
    print("🏗️ GENERATING ELITE STRUCTURE TEAMS FOR MANUAL VALIDATION")
    print("="*60)
    
    # Initialize generator
    generator = LiveTeamGenerator()
    
    # Set match parameters
    match_folder = "Live_Matches/Zimbabwe_T20I_Tri-Series_2025_New_Zealand_vs_South_Africa_25-07-2025"
    
    print(f"📁 Using match folder: {match_folder}")
    print(f"🎯 Generating teams with elite structures only")
    print(f"🔍 For manual score validation in the app")
    
    try:
        # Generate teams with elite structures
        # Focus on smaller batch for easier manual validation
        teams = generator.generate_live_teams(
            num_balanced_teams=0,        # Skip balanced
            num_aggressive_teams=0,      # Skip aggressive  
            num_elite_teams=200,         # Generate 200 elite teams only
            match_folder=match_folder
        )
        
        if not teams:
            print("❌ No teams generated!")
            return False
        
        print(f"✅ Generated {len(teams):,} teams successfully")
        
        # Convert to DataFrame for easier inspection
        df = pd.DataFrame(teams)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save for manual validation
        output_file = f"{match_folder}/Elite_Teams_For_Validation_{timestamp}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"💾 Saved teams to: {Path(output_file).name}")
        
        # Show team summary for validation
        print(f"\n📊 TEAM SUMMARY FOR MANUAL VALIDATION:")
        print(f"   Total teams: {len(df):,}")
        
        # Captain distribution
        if 'captain_id' in df.columns:
            captain_counts = df['captain_id'].value_counts()
            print(f"   Unique captains: {len(captain_counts)}")
            
            print(f"\n👑 Top 10 Captains:")
            for i, (captain, count) in enumerate(captain_counts.head(10).items(), 1):
                percentage = (count / len(df)) * 100
                print(f"      {i:2d}. {captain:<20} {count:3d} teams ({percentage:5.1f}%)")
        
        # Show captain roles if available
        if 'roles' in df.columns and 'player_ids' in df.columns:
            print(f"\n🎯 Captain Role Distribution:")
            captain_roles = []
            
            for idx, row in df.iterrows():
                try:
                    captain_id = row['captain_id']
                    player_ids = list(row['player_ids'])
                    roles = list(row['roles'])
                    
                    captain_idx = player_ids.index(captain_id)
                    captain_role = roles[captain_idx]
                    captain_roles.append(captain_role)
                except:
                    captain_roles.append('UNKNOWN')
            
            role_counts = pd.Series(captain_roles).value_counts()
            for role, count in role_counts.items():
                percentage = (count / len(df)) * 100
                print(f"      {role:<8} {count:3d} teams ({percentage:5.1f}%)")
        
        print(f"\n🔍 NEXT STEPS:")
        print(f"   1. Load teams in the CEM app")
        print(f"   2. Manually verify player scores are correct")
        print(f"   3. Check Dream11 scoring calculation")
        print(f"   4. Validate bowler phases and batting positions")
        print(f"   5. Once data is validated, we can re-run full pipeline")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error generating teams: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    generate_elite_structure_teams()
