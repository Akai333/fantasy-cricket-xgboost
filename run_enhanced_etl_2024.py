#!/usr/bin/env python3
"""
Run Enhanced Cricket ETL Pipeline for All Series in 2024
Uses the enhanced pipeline with data leakage prevention, DNP handling, and invalid match skipping
"""

import os
import sys
import subprocess
from datetime import datetime

# Add database_join to path
sys.path.append('database_join')

def get_available_series():
    """Get all available series folders"""
    cricsheet_base = "Cricsheet"
    if not os.path.exists(cricsheet_base):
        print(f"❌ Cricsheet directory not found: {cricsheet_base}")
        return []
    
    series_folders = []
    for item in os.listdir(cricsheet_base):
        item_path = os.path.join(cricsheet_base, item)
        if os.path.isdir(item_path) and item not in ['names.csv', 'people.csv']:
            series_folders.append(item)
    
    return sorted(series_folders)

def run_etl_for_series(series_name, year, append=False):
    """Run the enhanced ETL pipeline for a specific series"""
    print(f"\n{'='*60}")
    print(f"🏏 PROCESSING: {series_name} ({year})")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable, 
        "database_join/cricket_etl_pipeline.py",
        "--series", series_name,
        "--year", str(year)
    ]
    
    if append:
        cmd.append("--append")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {series_name}")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"❌ FAILED: {series_name}")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {series_name} took longer than 30 minutes")
        return False
    except Exception as e:
        print(f"💥 ERROR: {series_name} - {e}")
        return False

def main():
    print("🚀 Enhanced Cricket ETL Pipeline - All Series for 2024")
    print("Features: Data Leakage Prevention, DNP Handling, Invalid Match Skipping")
    print("="*80)
    
    # Get all available series
    series_list = get_available_series()
    if not series_list:
        print("❌ No series folders found!")
        return
    
    print(f"📋 Found {len(series_list)} series to process:")
    for i, series in enumerate(series_list, 1):
        print(f"  {i:2d}. {series}")
    
    print(f"\n🎯 Processing all series for year: 2024")
    
    # Track results
    successful_series = []
    failed_series = []
    start_time = datetime.now()
    
    # Process each series
    for i, series_name in enumerate(series_list, 1):
        append_mode = i > 1  # Append for all except first series
        
        print(f"\n📊 Progress: {i}/{len(series_list)} series")
        success = run_etl_for_series(series_name, 2024, append_mode)
        
        if success:
            successful_series.append(series_name)
        else:
            failed_series.append(series_name)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n" + "="*80)
    print(f"🎉 FINAL SUMMARY - Enhanced ETL Pipeline 2024")
    print(f"="*80)
    print(f"⏱️  Total Duration: {duration}")
    print(f"✅ Successful: {len(successful_series)}/{len(series_list)} series")
    print(f"❌ Failed: {len(failed_series)}/{len(series_list)} series")
    
    if successful_series:
        print(f"\n✅ SUCCESSFUL SERIES:")
        for series in successful_series:
            print(f"  • {series}")
    
    if failed_series:
        print(f"\n❌ FAILED SERIES:")
        for series in failed_series:
            print(f"  • {series}")
    
    print(f"\n📁 Output files: matches.csv and players.csv")
    print("="*80)

if __name__ == "__main__":
    main() 