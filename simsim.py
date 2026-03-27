#!/usr/bin/env python3
"""
🏏 SIMSIM - CSV Generator for Cricket Matches
Fast, accurate, pre-filled CSV generation for live cricket matches

FEATURES:
- 26 pitch descriptors with checkboxes (multiple selection)
- Player cache integration for instant role/style filling
- Last 10 scores from previous matches in same series
- DNP and abandoned match handling
- Excel dropdown validations
- Accuracy checking mechanisms
- Parallelized data fetching (API-safe)
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import concurrent.futures
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Setup paths
current_dir = Path(__file__).parent
app_root = current_dir
scripts_dir = app_root / 'scripts'
database_dir = app_root / 'database_join'

sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(app_root))

# Import API and cache modules
try:
    from scripts.api_fetcher import CricketDataAPI
    print("✅ API fetcher imported successfully")
except ImportError as e:
    print(f"❌ API import failed: {e}")
    CricketDataAPI = None


class SimSimCSVGenerator:
    """Fast CSV generator for cricket matches with accuracy checking"""
    
    def __init__(self):
        self.api = CricketDataAPI() if CricketDataAPI else None
        self.player_cache = self.load_player_cache()
        self.live_matches_dir = app_root / "Live_Matches"
        self.live_matches_dir.mkdir(exist_ok=True)
        
        # 26 Pitch Descriptors (exact match to feature extractor)
        self.pitch_descriptors = [
            'green', 'grassy', 'damp', 'wet', 'moist', 'sticky', 'sticky_wicket',  # Grass/Moisture (7)
            'dry', 'dusty', 'cracks', 'crumbling', 'rough',  # Dry/Cracked (5)
            'flat', 'hard', 'true', 'even_bounce', 'dead', 'batting_paradise', 'road',  # Batting-Friendly (7)
            'live', 'lively', 'seam', 'swing',  # Pace-Related (4)
            'turn', 'turning_track', 'variable_bounce'  # Spin-Related (3)
        ]
        
        # Pitch explanations
        self.pitch_explanations = {
            'green': 'Fresh grass cover',
            'grassy': 'Good grass coverage', 
            'damp': 'Moisture in surface',
            'wet': 'Significant moisture',
            'moist': 'Some dampness',
            'sticky': 'Tacky surface',
            'sticky_wicket': 'Very tacky, unpredictable',
            'dry': 'Lacks moisture',
            'dusty': 'Powdery dry surface',
            'cracks': 'Visible cracks',
            'crumbling': 'Surface breaking apart',
            'rough': 'Uneven surface',
            'flat': 'Even, predictable bounce',
            'hard': 'Firm surface',
            'true': 'Consistent bounce',
            'even_bounce': 'Predictable bounce',
            'dead': 'Very slow, no pace',
            'batting_paradise': 'Extremely bat-friendly',
            'road': 'Easy batting surface',
            'live': 'Good pace and bounce',
            'lively': 'Extra pace and bounce',
            'seam': 'Movement off the seam',
            'swing': 'Atmospheric swing',
            'turn': 'Spin-friendly surface',
            'turning_track': 'Significant turn',
            'variable_bounce': 'Unpredictable bounce'
        }
        
        print("🚀 SimSim CSV Generator initialized")
        print(f"📊 Player cache: {len(self.player_cache)} players loaded")
        print(f"🏟️ Pitch descriptors: {len(self.pitch_descriptors)} categories available")
        
    def load_player_cache(self) -> Dict:
        """Load player cache for instant role/style filling"""
        cache_path = database_dir / "player_cache.json"
        
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            print(f"✅ Player cache loaded: {len(cache)} players")
            return cache
        except Exception as e:
            print(f"⚠️ Player cache not available: {e}")
            return {}
    
    def generate_match_csv(self, match_identifier: str) -> bool:
        """
        Generate CSV files for a match
        
        Args:
            match_identifier: Either match_id directly, or auto-detect from recent matches
        """
        print(f"🎯 Generating CSV for: {match_identifier}")
        
        # Step 1: Get match data
        match_data = self.get_match_data(match_identifier)
        if not match_data:
            print("❌ Could not retrieve match data")
            return False
        
        # Step 2: Create match folder
        match_folder = self.create_match_folder(match_data)
        if not match_folder:
            print("❌ Could not create match folder")
            return False
        
        # Step 3: Generate CSVs with progress tracking
        print("📊 Generating CSV files...")
        
        # Context CSV (fast)
        context_success = self.generate_match_context_csv(match_data, match_folder)
        
        # Squad CSV (slower - with progress)
        squad_success = self.generate_squads_csv_with_progress(match_data, match_folder)
        
        if context_success and squad_success:
            print(f"✅ CSV generation completed!")
            print(f"📁 Files created in: {match_folder}")
            print(f"📋 Next steps:")
            print(f"   1. Open Excel: {match_folder}/match_context.csv")
            print(f"   2. Update ownership data from expert sheets")
            print(f"   3. Review and correct any data inaccuracies")
            
            # Show accuracy report
            self.show_accuracy_report(match_folder)
            return True
        else:
            print("❌ CSV generation failed")
            return False
    
    def get_match_data(self, identifier: str) -> Optional[Dict]:
        """Get comprehensive match data from API"""
        
        # If identifier looks like a match ID
        if len(identifier) > 20 and '-' in identifier:
            print(f"🔍 Fetching data for match ID: {identifier}")
            return self.get_match_by_id(identifier)
        
        # Otherwise, search in recent series
        print(f"🔍 Searching for matches with: {identifier}")
        return self.search_recent_matches(identifier)
    
    def get_match_by_id(self, match_id: str) -> Optional[Dict]:
        """Get match data by direct match ID with improved efficiency and fallbacks"""
        if not self.api:
            print("❌ API not available")
            return None
        
        try:
            # STEP 1: Get target match info (teams, date, series)
            print(f"🎯 Getting target match info...")
            metadata = self.api.get_match_metadata(match_id)
            if not metadata or metadata.get('status') == 'error':
                print(f"❌ Could not get match metadata for {match_id}")
                return None
            
            series_id = metadata.get('series_id', 'unknown')
            match_date = metadata.get('date', '')
            teams = metadata.get('teams', [])
            
            print(f"🏏 Target Match: {metadata.get('name', 'Unknown')}")
            print(f"📅 Date: {match_date}")
            print(f"👥 Teams: {', '.join(teams)}")
            print(f"🆔 Series: {series_id}")
            
            # STEP 2: Get efficient squad data (series_squad first, then match_squad fallback)
            print(f"📊 Getting squad data with smart fallbacks...")
            squad_data = self.get_efficient_squad_data(series_id, match_id)
            
            # STEP 3: Get filtered match history for these teams only
            print(f"📈 Getting filtered match history for teams before {match_date}...")
            team_match_history = self.get_team_history_before_date(teams, series_id, match_date)
            
            match_data = {
                'match_id': match_id,
                'metadata': metadata,
                'squad_data': squad_data,
                'series_id': series_id,
                'series_name': metadata.get('series_name', 'Unknown Series'),
                'teams': teams,
                'match_date': match_date,
                'team_match_history': team_match_history  # Pre-filtered relevant matches
            }
            
            print(f"✅ Enhanced match data retrieved for {match_id}")
            print(f"📊 Found {len(team_match_history)} relevant historical matches")
            return match_data
            
        except Exception as e:
            print(f"❌ Error getting match data: {e}")
            return None
    
    def search_recent_matches(self, search_term: str) -> Optional[Dict]:
        """Search for matches in recent series"""
        if not self.api:
            print("❌ API not available")
            return None
        
        try:
            # Get recent series
            today = datetime.now().strftime("%d/%m/%Y")
            series_list = self.api.get_series_for_date(today)
            
            if not series_list:
                print("❌ No recent series found")
                return None
            
            # Search through series for matching matches
            for series in series_list[:5]:  # Check first 5 series
                series_id = series.get('id')
                series_name = series.get('name', '')
                
                print(f"🔍 Searching in: {series_name}")
                
                matches = self.api.get_matches_for_series(series_id, today)
                if not matches:
                    continue
                
                # Look for match containing search term
                for match in matches:
                    match_name = match.get('name', '').lower()
                    if search_term.lower() in match_name:
                        print(f"✅ Found match: {match.get('name')}")
                        
                        # Get full match data
                        match_id = match.get('id')
                        metadata = self.api.get_match_metadata(match_id)
                        squad_data = self.api.get_squad(match_id)
                        
                        return {
                            'match_id': match_id,
                            'metadata': metadata,
                            'squad_data': squad_data,
                            'series_id': series_id,
                            'series_name': series_name
                        }
            
            print(f"❌ No matches found for: {search_term}")
            return None
            
        except Exception as e:
            print(f"❌ Error searching matches: {e}")
            return None
    
    def get_efficient_squad_data(self, series_id: str, match_id: str) -> Optional[Dict]:
        """Get squad data with smart fallbacks: series_squad first, then match_squad"""
        if not self.api:
            return None
        
        try:
            # STEP 1: Try series_squad first (works for most cases, faster)
            print("   🔄 Trying series_squad endpoint (preferred)...")
            series_squad_response = self.api._make_request("series_squad", {"id": series_id})
            
            if series_squad_response.get("status") == "success":
                squad_data = series_squad_response.get("data", [])
                if squad_data and len(squad_data) > 0:
                    total_players = sum(len(team.get("players", [])) for team in squad_data)
                    print(f"   ✅ series_squad SUCCESS: {len(squad_data)} teams, {total_players} players")
                    return {"data": squad_data, "source": "series_squad"}
            
            print(f"   ⚠️ series_squad failed, trying match_squad fallback...")
            
            # STEP 2: Fallback to match_squad if series_squad fails
            print("   🔄 Trying match_squad endpoint (fallback)...")
            match_squad_data = self.api.get_squad(match_id)
            
            if match_squad_data and match_squad_data.get('status') != 'error':
                print(f"   ✅ match_squad SUCCESS")
                return {"data": match_squad_data, "source": "match_squad"}
            
            print(f"   ❌ Both squad endpoints failed")
            return None
            
        except Exception as e:
            print(f"   ❌ Error getting squad data: {e}")
            return None
    
    def get_team_history_before_date(self, teams: List[str], series_id: str, target_date: str) -> List[Dict]:
        """Get filtered match history: only matches before target date where either team played"""
        if not self.api or not teams:
            return []
        
        try:
            # Get all matches in series
            print(f"   🔄 Getting all series matches to filter...")
            all_matches = self.api.get_all_series_matches(series_id)
            
            if not all_matches:
                print(f"   ⚠️ No matches found in series {series_id}")
                return []
            
            print(f"   📊 Found {len(all_matches)} total matches in series")
            
            # Filter matches: before target date AND involving either team
            relevant_matches = []
            target_teams_lower = [team.lower() for team in teams]
            
            for match in all_matches:
                match_date = match.get('date', '')
                match_name = match.get('name', '').lower()
                match_status = match.get('status', '').lower()
                
                # Filter 1: Must be before target date
                if match_date >= target_date:
                    continue
                
                # Filter 2: Must be completed (has data)
                if 'not started' in match_status or 'match not started' in match_status:
                    continue
                
                # Filter 3: Must involve one of our target teams
                team_involved = False
                for target_team in target_teams_lower:
                    if target_team in match_name:
                        team_involved = True
                        break
                
                if team_involved:
                    relevant_matches.append(match)
                    print(f"   ✅ Relevant: {match.get('name', 'Unknown')} - {match_date}")
            
            # Sort chronologically (oldest first for Last 10 processing)
            relevant_matches.sort(key=lambda x: x.get('date', ''))
            
            print(f"   📈 Filtered to {len(relevant_matches)} relevant historical matches")
            return relevant_matches
            
        except Exception as e:
            print(f"   ❌ Error filtering match history: {e}")
            return []
    
    def create_match_folder(self, match_data: Dict) -> Optional[Path]:
        """Create match folder with standardized naming"""
        try:
            metadata = match_data.get('metadata', {})
            teams = metadata.get('teams', [])
            date = metadata.get('date', datetime.now().strftime("%Y-%m-%d"))
            series_name = match_data.get('series_name', 'Unknown_Series')
            
            if len(teams) < 2:
                print("❌ Insufficient team data for folder creation")
                return None
            
            # Clean names for folder
            series_clean = series_name.replace(' ', '_').replace('/', '_').replace(',', '')
            team1_clean = teams[0].replace(' ', '').replace('/', '_')
            team2_clean = teams[1].replace(' ', '').replace('/', '_')
            date_clean = date.replace('-', '')[:8]  # YYYYMMDD format
            
            folder_name = f"{series_clean}_{team1_clean}_vs_{team2_clean}_{date_clean}"
            match_folder = self.live_matches_dir / folder_name
            
            # Create folder and subfolders
            match_folder.mkdir(exist_ok=True)
            (match_folder / "output").mkdir(exist_ok=True)
            
            print(f"📁 Match folder created: {folder_name}")
            return match_folder
            
        except Exception as e:
            print(f"❌ Error creating match folder: {e}")
            return None
    
    def generate_match_context_csv(self, match_data: Dict, match_folder: Path) -> bool:
        """Generate match_context.csv with Excel validations"""
        try:
            metadata = match_data.get('metadata', {})
            teams = metadata.get('teams', [])
            
            # Prepare context data
            context_data = [
                ['field', 'value', 'validation_list', 'notes'],
                ['series_id', match_data.get('series_id', ''), '', 'API filled'],
                ['series_name', match_data.get('series_name', ''), '', 'API filled'],
                ['match_id', match_data.get('match_id', ''), '', 'API filled'],
                ['match_name', metadata.get('name', ''), '', 'API filled'],
                ['venue', metadata.get('venue', ''), '', 'API filled'],
                ['date', metadata.get('date', ''), '', 'API filled'],
                ['team_a', teams[0] if len(teams) > 0 else '', '', 'API filled - for dropdown'],
                ['team_b', teams[1] if len(teams) > 1 else '', '', 'API filled - for dropdown'],
                ['toss_winner', '', f'{teams[0] if len(teams) > 0 else ""},{teams[1] if len(teams) > 1 else ""}', 'Excel dropdown - you pick winner'],
                ['toss_decision', '', 'Bat,Field', 'Excel dropdown - Bat First or Field First'],
                ['batting_first_team', '', 'Auto-calculated', 'Auto-filled based on toss'],
                ['chasing_team', '', 'Auto-calculated', 'Auto-filled based on toss'],
                ['pitch_conditions', '', 'Multiple checkboxes - see pitch_modal.html', 'You select multiple from 26 descriptors']
            ]
            
            # Save to CSV
            context_file = match_folder / "match_context.csv"
            with open(context_file, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerows(context_data)
            
            # Create pitch selection HTML for easy selection
            self.create_pitch_selection_html(match_folder)
            
            print(f"✅ Match context CSV created: {context_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating match context CSV: {e}")
            return False
    
    def create_pitch_selection_html(self, match_folder: Path):
        """Create HTML file for easy pitch selection with checkboxes"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Pitch Conditions Selection</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .category { margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }
        .category h3 { margin-top: 0; color: #333; }
        .descriptor { margin: 5px 0; }
        .descriptor input { margin-right: 8px; }
        .descriptor label { cursor: pointer; }
        .explanation { color: #666; font-size: 0.9em; margin-left: 25px; }
        .selected-output { margin: 20px 0; padding: 15px; background: #f0f8ff; border-radius: 5px; }
        .copy-button { background: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h2>🏟️ Pitch Conditions Selection (26 Descriptors)</h2>
    <p><strong>Instructions:</strong> Select multiple conditions that describe the pitch. Selected values will be formatted for CSV entry.</p>
    
    <div class="category">
        <h3>🌱 Grass & Moisture (7)</h3>
"""
        
        # Add categories with checkboxes
        categories = [
            ("🌱 Grass & Moisture", ['green', 'grassy', 'damp', 'wet', 'moist', 'sticky', 'sticky_wicket']),
            ("🏜️ Dry & Cracked", ['dry', 'dusty', 'cracks', 'crumbling', 'rough']),
            ("🏏 Batting Friendly", ['flat', 'hard', 'true', 'even_bounce', 'dead', 'batting_paradise', 'road']),
            ("⚡ Pace Related", ['live', 'lively', 'seam', 'swing']),
            ("🌀 Spin Related", ['turn', 'turning_track', 'variable_bounce'])
        ]
        
        for i, (cat_name, descriptors) in enumerate(categories):
            if i > 0:  # Add category div for categories after the first
                html_content += f'''
    </div>
    <div class="category">
        <h3>{cat_name}</h3>
'''
            
            for desc in descriptors:
                explanation = self.pitch_explanations.get(desc, 'No explanation')
                html_content += f'''
        <div class="descriptor">
            <input type="checkbox" id="{desc}" onchange="updateSelected()">
            <label for="{desc}"><strong>{desc}</strong></label>
            <div class="explanation">{explanation}</div>
        </div>
'''
        
        html_content += '''
    </div>
    
    <div class="selected-output">
        <h3>📋 Selected Conditions:</h3>
        <p id="selected-list">None selected</p>
        <p><strong>CSV Format:</strong> <span id="csv-format">""</span></p>
        <button class="copy-button" onclick="copyToClipboard()">Copy CSV Value</button>
    </div>
    
    <script>
    function updateSelected() {
        const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
        const selected = Array.from(checkboxes).map(cb => cb.id);
        
        const listElement = document.getElementById('selected-list');
        const csvElement = document.getElementById('csv-format');
        
        if (selected.length === 0) {
            listElement.textContent = 'None selected';
            csvElement.textContent = '""';
        } else {
            listElement.textContent = selected.join(', ');
            csvElement.textContent = '"' + selected.join(',') + '"';
        }
    }
    
    function copyToClipboard() {
        const csvValue = document.getElementById('csv-format').textContent;
        navigator.clipboard.writeText(csvValue).then(() => {
            alert('Copied to clipboard! Paste this into the pitch_conditions field in Excel.');
        });
    }
    </script>
</body>
</html>
'''
        
        # Save HTML file
        html_file = match_folder / "pitch_selection.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🏟️ Pitch selection HTML created: {html_file}")
        print(f"   → Open this file in browser to select pitch conditions")
    
    def generate_squads_csv_with_progress(self, match_data: Dict, match_folder: Path) -> bool:
        """Generate squads_combined.csv with progress tracking"""
        try:
            print("📊 Fetching squad data with progress tracking...")
            print("⚡ Using optimized mode for speed (limited historical data)")
            
            # Extract squad data
            squad_data = match_data.get('squad_data', {})
            metadata = match_data.get('metadata', {})
            teams = metadata.get('teams', [])
            
            if not squad_data or len(teams) < 2:
                print("❌ Insufficient squad data")
                return False
            
            # Prepare CSV data
            csv_data = []
            csv_data.append([
                'team', 'player_name', 'role', 'batting_style', 'bowling_style', 
                'batting_order', 'ownership_pct', 'captain_pct', 'vc_pct',
                'bowl_phases', 'last_10_scores', 'avg_balls_faced', 'avg_overs_bowled', 
                'avg_fantasy_pts', 'notes'
            ])
            
            # Process each team
            for team_name in teams:
                print(f"🔄 Processing {team_name}...")
                
                team_players = self.get_team_players(squad_data, team_name)
                if not team_players:
                    print(f"⚠️ No players found for {team_name}")
                    continue
                
                # Process players with progress
                for i, player in enumerate(team_players):
                    print(f"   {i+1}/{len(team_players)}: {player.get('name', 'Unknown')}")
                    
                    player_row = self.create_player_row(
                        player, team_name, match_data
                    )
                    csv_data.append(player_row)
            
            # Save CSV
            squad_file = match_folder / "squads_combined.csv"
            with open(squad_file, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            print(f"✅ Combined squads CSV created: {squad_file}")
            print(f"📊 Total players: {len(csv_data) - 1}")  # -1 for header
            return True
            
        except Exception as e:
            print(f"❌ Error creating squads CSV: {e}")
            return False
    
    def get_team_players(self, squad_data: Dict, team_name: str) -> List[Dict]:
        """Extract players for a specific team from squad data with smart fallback support"""
        try:
            if not squad_data:
                print(f"⚠️ No squad data available for {team_name}")
                return []
            
            # Handle new efficient squad data structure
            actual_data = squad_data.get('data', squad_data)
            data_source = squad_data.get('source', 'unknown')
            
            print(f"   🔍 Extracting {team_name} players from {data_source}...")
            
            # Handle series_squad format (list of teams)
            if isinstance(actual_data, list):
                for team_info in actual_data:
                    if isinstance(team_info, dict):
                        team_name_api = team_info.get('teamName', team_info.get('name', ''))
                        if team_name_api and team_name.lower() in team_name_api.lower():
                            players = team_info.get('players', [])
                            print(f"   ✅ Found {len(players)} players for {team_name}")
                            return players
            
            # Handle match_squad format (dict structure)
            elif isinstance(actual_data, dict):
                # Try direct team access
                if team_name in actual_data:
                    team_data = actual_data[team_name]
                    if isinstance(team_data, dict) and 'players' in team_data:
                        players = team_data['players']
                        print(f"   ✅ Found {len(players)} players for {team_name}")
                        return players
                    elif isinstance(team_data, list):
                        print(f"   ✅ Found {len(team_data)} players for {team_name}")
                        return team_data
                
                # Try team1/team2 structure
                for key in ['team1', 'team2']:
                    if key in actual_data:
                        team_data = actual_data[key]
                        if isinstance(team_data, dict):
                            if team_data.get('name') == team_name and 'players' in team_data:
                                players = team_data['players']
                                print(f"   ✅ Found {len(players)} players for {team_name}")
                                return players
                
                # Try looking in nested data array
                if 'data' in actual_data:
                    nested_data = actual_data['data']
                    if isinstance(nested_data, list):
                        for team_info in nested_data:
                            if isinstance(team_info, dict):
                                team_name_api = team_info.get('teamName', team_info.get('name', ''))
                                if team_name_api and team_name.lower() in team_name_api.lower():
                                    players = team_info.get('players', [])
                                    print(f"   ✅ Found {len(players)} players for {team_name}")
                                    return players
            
            print(f"   ⚠️ Could not find players for {team_name} in {data_source} data")
            return []
            
        except Exception as e:
            print(f"   ❌ Error extracting team players: {e}")
            return []
    
    def create_player_row(self, player: Dict, team_name: str, match_data: Dict) -> List:
        """Create a CSV row for a player with all required data"""
        try:
            player_name = player.get('name', 'Unknown')
            
            print(f"   📋 Processing {player_name}...")
            
            # Get cached data for static info (roles/styles only)
            cached_data = self.get_cached_player_data(player_name)
            
            # Get API data for ALL statistical data (NEVER use cache for stats)
            api_data = self.get_api_player_data(player_name, team_name, match_data)
            
            # Merge data - CACHE for static info, API for stats
            role = cached_data.get('role') or api_data.get('role') or 'Unknown'
            batting_style = cached_data.get('batting_style') or api_data.get('batting_style') or 'Unknown'
            bowling_style = cached_data.get('bowling_style') or api_data.get('bowling_style') or 'None'
            
            # CRITICAL: Get REAL statistical data from API only
            avg_balls_faced = api_data.get('avg_balls_faced', 0.0)
            avg_overs_bowled = api_data.get('avg_overs_bowled', 0.0)
            avg_fantasy_pts = api_data.get('avg_fantasy_pts', 0.0)
            
            # Get Last 10 scores from the API data we already fetched
            print(f"   🔄 Processing Last 10 scores from API data...")
            last_10_array = api_data.get('last_10_scores', [])
            if isinstance(last_10_array, list) and last_10_array:
                last_10_scores = ', '.join(map(str, last_10_array))
                print(f"   ✅ Got Last 10 scores: {last_10_scores}")
            else:
                last_10_scores = ""
                print(f"   ⚠️ No Last 10 scores available")
            
            # Default bowling phases based on role (single comma-separated string)
            pp, mo, death = self.get_default_bowling_phases(role, bowling_style)
            bowl_phases = f"{pp},{mo},{death}"
            
            # Log data sources
            data_source = []
            if cached_data: data_source.append("Role/Style: Cache")
            if api_data: data_source.append("Stats: API")
            if last_10_scores: data_source.append("Last10: API")
            notes = " | ".join(data_source) if data_source else "No data"
            
            # Create row
            return [
                team_name,                      # team
                player_name,                    # player_name
                role,                          # role
                batting_style,                 # batting_style
                bowling_style,                 # bowling_style
                '',                           # batting_order (you fill post-toss)
                '1.0',                        # ownership_pct (you update from expert)
                '1.0',                        # captain_pct (you update from Dream11)
                '1.0',                        # vc_pct (you update from Dream11)
                bowl_phases,                  # bowl_phases (single comma-separated string)
                last_10_scores,               # last_10_scores (REAL API DATA)
                str(avg_balls_faced),         # avg_balls_faced (REAL API DATA)
                str(avg_overs_bowled),        # avg_overs_bowled (REAL API DATA)
                str(avg_fantasy_pts),         # avg_fantasy_pts (REAL API DATA)
                notes                         # notes showing data sources
            ]
            
        except Exception as e:
            print(f"⚠️ Error creating row for {player.get('name', 'Unknown')}: {e}")
            return [team_name, player.get('name', 'Unknown'), 'Unknown', 'Unknown', 'None', 
                   '', '1.0', '1.0', '1.0', '0.0,0.0,0.0', '', '0.0', '0.0', '0.0', 'Error']
    
    def get_cached_player_data(self, player_name: str) -> Dict:
        """Get player data from cache (instant)"""
        if not self.player_cache:
            return {}
        
        try:
            # Direct match first
            for cache_key, player_data in self.player_cache.items():
                cached_name = player_data.get('name', '')
                if cached_name.lower() == player_name.lower():
                    return player_data
            
            # Fuzzy match
            name_parts = player_name.lower().split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                for cache_key, player_data in self.player_cache.items():
                    cached_name = player_data.get('name', '').lower()
                    if last_name in cached_name:
                        return player_data
            
            return {}
            
        except Exception as e:
            print(f"⚠️ Cache lookup error for {player_name}: {e}")
            return {}
    
    def get_api_player_data(self, player_name: str, team_name: str, match_data: Dict) -> Dict:
        """Get player statistics using EFFICIENT pre-filtered match history"""
        if not self.api:
            return {}
        
        try:
            print(f"   🔍 Getting EFFICIENT API stats for {player_name}...")
            
            # Use pre-filtered match history from match_data (much faster!)
            team_match_history = match_data.get('team_match_history', [])
            series_id = match_data.get('series_id')
            
            if not team_match_history:
                print(f"   ⚠️ No pre-filtered match history available")
                return {}
            
            # Calculate stats from relevant matches only
            stats = self.calculate_player_stats_efficient(player_name, team_name, team_match_history, series_id)
            
            if stats:
                print(f"   ✅ Efficient stats calculated for {player_name}")
                return stats
            else:
                print(f"   ⚠️ Could not calculate stats for {player_name}")
                return {}
            
        except Exception as e:
            print(f"⚠️ API lookup error for {player_name}: {e}")
            return {}
    
    def calculate_player_stats_efficient(self, player_name: str, team_name: str, 
                                       relevant_matches: List[Dict], series_id: str) -> Dict:
        """Calculate player stats using only relevant pre-filtered matches"""
        if not self.api or not relevant_matches:
            return {}
        
        try:
            # Get player ID from series squad (cache it for efficiency)
            if not hasattr(self, '_player_id_cache') or self._player_id_cache_series != series_id:
                self._player_id_cache = self.api.build_player_id_cache(series_id)
                self._player_id_cache_series = series_id
            
            player_key = player_name.lower()
            if player_key not in self._player_id_cache:
                print(f"   ⚠️ Player {player_name} not found in series squad")
                return {}
            
            player_info = self._player_id_cache[player_key]
            player_id = player_info['id']
            
            print(f"   📊 Processing {len(relevant_matches)} relevant matches for {player_name}")
            
            # Initialize arrays
            fantasy_scores = []
            balls_faced_array = []
            overs_bowled_array = []
            
            # Process each relevant match
            for match in relevant_matches:
                match_id = match.get('id')
                match_name = match.get('name', 'Unknown')
                
                # Check if this player's team played in this match
                if team_name.lower() not in match_name.lower():
                    continue
                
                print(f"   ⚽ Processing: {match_name}")
                
                # Get player performance for this match
                fantasy_points, balls_faced, overs_bowled = self.get_player_match_performance(
                    player_id, player_name, match_id, match_name
                )
                
                fantasy_scores.append(fantasy_points)
                balls_faced_array.append(balls_faced)
                overs_bowled_array.append(overs_bowled)
            
            # Calculate averages (last 5 matches)
            if fantasy_scores:
                recent_5_fantasy = fantasy_scores[-5:] if len(fantasy_scores) >= 5 else fantasy_scores
                recent_5_balls = balls_faced_array[-5:] if len(balls_faced_array) >= 5 else balls_faced_array
                recent_5_overs = overs_bowled_array[-5:] if len(overs_bowled_array) >= 5 else overs_bowled_array
                
                avg_fantasy = sum(recent_5_fantasy) / len(recent_5_fantasy) if recent_5_fantasy else 0
                avg_balls = sum(recent_5_balls) / len(recent_5_balls) if recent_5_balls else 0
                avg_overs = sum(recent_5_overs) / len(recent_5_overs) if recent_5_overs else 0
                
                print(f"   📊 Stats: Avg Fantasy {avg_fantasy:.1f}, Avg Balls {avg_balls:.1f}, Avg Overs {avg_overs:.1f}")
                
                return {
                    'role': player_info.get('role', 'Unknown'),
                    'batting_style': player_info.get('batting_style', 'Unknown'),
                    'bowling_style': player_info.get('bowling_style', 'None'),
                    'avg_balls_faced': avg_balls,
                    'avg_overs_bowled': avg_overs,
                    'avg_fantasy_pts': avg_fantasy,
                    'last_10_scores': fantasy_scores  # All relevant matches chronologically
                }
            
            return {}
            
        except Exception as e:
            print(f"   ❌ Error calculating efficient stats: {e}")
            return {}
    
    def get_player_match_performance(self, player_id: str, player_name: str, 
                                   match_id: str, match_name: str) -> Tuple[int, int, float]:
        """Get player performance for a specific match (fantasy points, balls faced, overs bowled)"""
        try:
            fantasy_points = 0
            balls_faced = 0
            overs_bowled = 0.0
            played = False
            
            # Try match_points API first
            points_response = self.api._make_request("match_points", {"id": match_id, "ruleset": ""})
            
            if points_response.get("status") == "success":
                points_data = points_response.get("data", {})
                for player_data in points_data.get("totals", []):
                    if player_data.get("id", "") == player_id:
                        fantasy_points = player_data.get("points", 0)
                        played = True
                        break
            
            # Get detailed stats from scorecard
            scorecard_response = self.api._make_request("match_scorecard", {"id": match_id})
            
            if scorecard_response.get("status") == "success":
                scorecard_data = scorecard_response.get("data", {})
                scorecard = scorecard_data.get("scorecard", [])
                
                runs_scored = 0
                wickets_taken = 0
                runs_conceded = 0
                
                for innings in scorecard:
                    # Check batting
                    for batsman_data in innings.get("batting", []):
                        batsman = batsman_data.get("batsman", {})
                        if batsman.get("id", "") == player_id:
                            runs_scored = batsman_data.get('r', 0)
                            balls_faced = batsman_data.get('b', 0)
                            if not played:
                                played = True
                            break
                    
                    # Check bowling
                    for bowler_data in innings.get("bowling", []):
                        bowler = bowler_data.get("bowler", {})
                        if bowler.get("id", "") == player_id:
                            overs_bowled = float(bowler_data.get('o', 0))
                            wickets_taken = int(bowler_data.get('w', 0))
                            runs_conceded = int(bowler_data.get('r', 0))
                            if not played:
                                played = True
                            break
            
            # Handle missing fantasy points data with Dream11 calculation
            if played and points_response.get("status") != "success":
                print(f"   ⚡ Calculating fantasy points from scorecard for {match_name}")
                fantasy_points = self.calculate_dream11_points(runs_scored, balls_faced, overs_bowled, 
                                                             wickets_taken, runs_conceded)
            elif not played:
                fantasy_points = 0  # DNP
            elif fantasy_points <= 0 and played:
                fantasy_points = 4  # Minimum for playing
            else:
                fantasy_points = max(4, fantasy_points) if played else fantasy_points
            
            return fantasy_points, balls_faced, overs_bowled
            
        except Exception as e:
            print(f"   ❌ Error getting match performance: {e}")
            return 0, 0, 0.0
    
    def calculate_dream11_points(self, runs: int, balls: int, overs: float, 
                               wickets: int, runs_conceded: int) -> int:
        """Calculate fantasy points using Dream11 T20 rules"""
        points = 4  # Base points for playing
        
        # Batting points
        if runs > 0 or balls > 0:
            points += runs  # 1 point per run
            
            # Strike rate bonuses
            if balls >= 10:
                strike_rate = (runs / balls) * 100
                if strike_rate >= 170:
                    points += 6
                elif strike_rate >= 150:
                    points += 4
                elif strike_rate >= 130:
                    points += 2
                elif strike_rate < 70:
                    points -= 2
            
            # Milestone bonuses
            if runs >= 100:
                points += 16
            elif runs >= 50:
                points += 8
        
        # Bowling points
        if overs > 0:
            # Wicket bonuses
            points += wickets * 25
            
            # Wicket milestone bonuses
            if wickets >= 5:
                points += 16
            elif wickets >= 4:
                points += 8
            elif wickets >= 3:
                points += 4
            
            # Maiden over bonus
            if runs_conceded == 0 and overs >= 1:
                points += int(overs) * 12
            
            # Base bowling points
            points += int(overs * 4)
        
        return max(4, points)
    
    def calculate_last_10_scores(self, player_name: str, team_name: str, match_data: Dict) -> str:
        """Calculate Last 10 fantasy scores from API - REAL SCORES ONLY with robust error handling"""
        if not self.api:
            print(f"   ⚠️ No API available for {player_name} scores")
            return ""
        
        try:
            series_id = match_data.get('series_id')
            match_id = match_data.get('match_id')
            
            print(f"   🔍 Getting last 10 scores for {player_name}...")
            
            # Try to get team's last matches with timeout protection
            team_matches = []
            try:
                print(f"   🔄 Searching team history for {team_name}...")
                team_matches = self.api.get_team_last_matches(team_name, match_id, limit=10)
            except Exception as e:
                print(f"   ⚠️ Team matches API failed: {e}")
                team_matches = []
            
            # CRITICAL: This is 3rd T20I, so first 2 matches MUST exist
            # Try series matches more aggressively
            if not team_matches:
                print(f"   🔄 Searching for completed matches in series (this is 3rd T20I)...")
                try:
                    series_matches = self.api.get_all_series_matches(series_id)
                    if series_matches:
                        print(f"   📊 Found {len(series_matches)} total series matches")
                        
                        # Filter for previous matches (ignore status for now)
                        previous_matches = [m for m in series_matches 
                                          if (m.get('id') != match_id and 
                                              team_name in m.get('teams', []))]
                        
                        print(f"   📊 Found {len(previous_matches)} previous matches with {team_name}")
                        
                        # For debugging, show all matches
                        for i, match in enumerate(series_matches):
                            match_name = match.get('name', 'Unknown')
                            match_status = match.get('status', 'Unknown')
                            match_teams = match.get('teams', [])
                            print(f"   📋 Match {i+1}: {match_name} - Status: {match_status} - Teams: {match_teams}")
                        
                        team_matches = previous_matches
                    else:
                        print(f"   ⚠️ No series matches available")
                except Exception as e:
                    print(f"   ⚠️ Series matches API failed: {e}")
                    team_matches = []
            
            # If still no matches, return empty (new series scenario)
            if not team_matches:
                print(f"   ℹ️ No historical data for {player_name} (new series/player)")
                return ""
            
            # Sort by date (most recent first)
            team_matches.sort(key=lambda x: x.get('date', ''), reverse=True)
            print(f"   📊 Processing {len(team_matches)} matches for {player_name}")
            
            # Get scores with limited attempts (max 5 for speed)
            scores = []
            max_attempts = min(5, len(team_matches))
            
            for i, match in enumerate(team_matches[:max_attempts]):
                try:
                    match_match_id = match.get('id', '')
                    if not match_match_id:
                        continue
                        
                    print(f"   🔄 Score {i+1}/{max_attempts} from {match_match_id[:8]}...")
                    
                    # Get player stats with timeout protection
                    match_stats = None
                    try:
                        match_stats = self.api.get_player_match_stats(player_name, match_match_id)
                    except Exception as e:
                        print(f"   ⚠️ Match stats failed: {e}")
                        scores.append('0')  # DNP due to API error
                        continue
                    
                    if match_stats and match_stats.get('status') != 'error':
                        fantasy_points = match_stats.get('fantasy_points', 0)
                        
                        if isinstance(fantasy_points, (int, float)) and fantasy_points > 0:
                            scores.append(str(int(fantasy_points)))
                            print(f"   ✅ Score: {int(fantasy_points)}")
                        elif isinstance(fantasy_points, (int, float)) and fantasy_points == 0:
                            scores.append('4')  # Played but scored 0 = minimum 4 points
                            print(f"   ✅ Played: 4 (minimum)")
                        else:
                            scores.append('0')  # DNP
                            print(f"   ⚠️ DNP")
                    else:
                        scores.append('0')  # DNP - no valid stats
                        print(f"   ⚠️ No stats")
                        
                except Exception as e:
                    scores.append('0')  # Error = DNP
                    print(f"   ❌ Error: {str(e)[:50]}...")
            
            if scores:
                scores_str = '|'.join(scores)
                print(f"   ✅ Last {len(scores)} scores for {player_name}: {scores_str}")
                return scores_str
            else:
                print(f"   ℹ️ No historical scores for {player_name}")
                return ""
            
        except Exception as e:
            print(f"⚠️ Last 10 calculation error for {player_name}: {str(e)[:100]}...")
            return ""
    
    def get_default_bowling_phases(self, role: str, bowling_style: str) -> Tuple[float, float, float]:
        """Get default bowling phases based on role and style"""
        try:
            role = role.upper() if role else ""
            bowling_style = bowling_style.lower() if bowling_style else "none"
            
            # Non-bowlers
            if role in ['BAT', 'WK'] or 'none' in bowling_style:
                return (0.0, 0.0, 0.0)
            
            # Fast bowlers - typically PP and Death
            if any(term in bowling_style for term in ['fast', 'medium', 'pace']):
                return (0.6, 0.2, 0.2)  # PP heavy
            
            # Spinners - typically Middle overs
            if any(term in bowling_style for term in ['spin', 'leg', 'off', 'chinaman']):
                return (0.1, 0.8, 0.1)  # Middle heavy
            
            # All-rounders - balanced
            if role == 'AR':
                return (0.33, 0.34, 0.33)  # Equal phases
            
            # Default for bowlers
            if role == 'BOWL':
                return (0.4, 0.4, 0.2)
            
            return (0.0, 0.0, 0.0)
            
        except Exception:
            return (0.0, 0.0, 0.0)
    
    def debug_fantasy_scores(self, player_name: str, team_name: str, match_data: Dict):
        """Debug function to compare API scores vs expected Dream11 scores"""
        try:
            print(f"\n🔍 DEBUGGING FANTASY SCORES FOR {player_name}")
            print("=" * 60)
            
            series_id = match_data.get('series_id')
            if series_id:
                enhanced_stats = self.api.get_enhanced_player_stats(player_name, series_id, team_name)
                if enhanced_stats:
                    scores = enhanced_stats.get('all_match_scores', [])
                    print(f"📊 API Scores ({len(scores)} matches): {scores}")
                    print(f"📊 Last 10: {enhanced_stats.get('last_10_fantasy_scores', [])}")
                    print(f"📊 Avg Last 5: {enhanced_stats.get('avg_fantasy_points_last_5', 0)}")
                    print(f"📊 Total matches found: {enhanced_stats.get('total_matches_found', 0)}")
                    print(f"📊 Matches played: {enhanced_stats.get('matches_played', 0)}")
                    return enhanced_stats
                else:
                    print("❌ No enhanced stats returned")
            else:
                print("❌ No series_id available")
            return None
        except Exception as e:
            print(f"❌ Debug error: {e}")
            return None

    def show_accuracy_report(self, match_folder: Path):
        """Generate accuracy report for data validation"""
        try:
            print("\n📊 ACCURACY REPORT")
            print("=" * 50)
            
            # Read generated CSVs
            context_file = match_folder / "match_context.csv"
            squad_file = match_folder / "squads_combined.csv"
            
            # Context accuracy
            if context_file.exists():
                print("✅ Match context CSV: Generated successfully")
                with open(context_file, 'r', encoding='utf-8') as f:
                    import csv
                    reader = csv.reader(f)
                    rows = list(reader)
                    print(f"   📋 Fields: {len(rows) - 1}")  # -1 for header
            
            # Squad accuracy
            if squad_file.exists():
                squad_df = pd.read_csv(squad_file)
                print("✅ Squad CSV: Generated successfully")
                print(f"   👥 Total players: {len(squad_df)}")
                print(f"   🏏 Teams: {squad_df['team'].nunique()}")
                
                # Data quality checks
                missing_roles = (squad_df['role'] == 'Unknown').sum()
                missing_styles = (squad_df['batting_style'] == 'Unknown').sum()
                has_last10 = (squad_df['last_10_scores'] != '').sum()
                
                print(f"\n📊 Data Quality:")
                print(f"   Roles identified: {len(squad_df) - missing_roles}/{len(squad_df)}")
                print(f"   Batting styles: {len(squad_df) - missing_styles}/{len(squad_df)}")
                print(f"   Last 10 scores: {has_last10}/{len(squad_df)}")
                
                if missing_roles > 0:
                    print(f"   ⚠️ {missing_roles} players need role verification")
                if missing_styles > 0:
                    print(f"   ⚠️ {missing_styles} players need batting style")
                if has_last10 < len(squad_df) * 0.8:
                    print(f"   ⚠️ Last 10 scores may need manual verification")
            
            print(f"\n📋 Manual Tasks Remaining:")
            print(f"   1. Open pitch_selection.html → Select pitch conditions")
            print(f"   2. Update ownership data from expert sheets")
            print(f"   3. Verify player roles and styles")
            print(f"   4. Check Last 10 scores accuracy")
            print("=" * 50)
            
        except Exception as e:
            print(f"⚠️ Accuracy report error: {e}")


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='🏏 SimSim CSV Generator')
    parser.add_argument('match_id', nargs='?', type=str, 
                       help='Match ID or search term for the match')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with GUI match selection')
    
    args = parser.parse_args()
    
    print("🏏 SimSim CSV Generator v1.0")
    print("Fast, accurate cricket match CSV generation")
    print("=" * 50)
    
    generator = SimSimCSVGenerator()
    
    if args.interactive:
        # Interactive mode
        run_interactive_mode(generator)
    elif args.match_id:
        # Direct match ID mode
        success = generator.generate_match_csv(args.match_id)
        if not success:
            sys.exit(1)
    else:
        # Prompt mode
        match_id = input("Enter match ID or search term: ").strip()
        if not match_id:
            print("❌ No match identifier provided")
            sys.exit(1)
        
        success = generator.generate_match_csv(match_id)
        if not success:
            sys.exit(1)
    
    print("\n🎉 SimSim CSV generation completed!")


def run_interactive_mode(generator):
    """Run interactive GUI mode for match selection"""
    print("🖥️ Starting interactive mode...")
    
    root = tk.Tk()
    root.title("SimSim CSV Generator")
    root.geometry("600x400")
    
    # Main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill="both", expand=True)
    
    ttk.Label(main_frame, text="🏏 SimSim CSV Generator", 
             font=("Arial", 16, "bold")).pack(pady=10)
    
    # Match input
    ttk.Label(main_frame, text="Enter Match ID or Search Term:").pack(pady=5)
    match_entry = ttk.Entry(main_frame, width=50)
    match_entry.pack(pady=5)
    
    # Progress
    progress_var = tk.StringVar(value="Ready to generate...")
    progress_label = ttk.Label(main_frame, textvariable=progress_var)
    progress_label.pack(pady=10)
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)
    
    def generate_csv():
        match_id = match_entry.get().strip()
        if not match_id:
            messagebox.showerror("Error", "Please enter a match ID or search term")
            return
        
        progress_var.set("Generating CSV files...")
        root.update()
        
        success = generator.generate_match_csv(match_id)
        
        if success:
            progress_var.set("✅ CSV generation completed!")
            messagebox.showinfo("Success", "CSV files generated successfully!")
        else:
            progress_var.set("❌ CSV generation failed")
            messagebox.showerror("Error", "Failed to generate CSV files")
    
    ttk.Button(button_frame, text="Generate CSV", 
              command=generate_csv).pack(side="left", padx=5)
    ttk.Button(button_frame, text="Exit", 
              command=root.quit).pack(side="left", padx=5)
    
    root.mainloop()


if __name__ == "__main__":
    main()