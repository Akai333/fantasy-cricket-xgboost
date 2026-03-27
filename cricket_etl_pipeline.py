#!/usr/bin/env python3
"""
Cricket ETL Pipeline
Joins Cricsheet.org JSON files with Cricketdata.org API data
Creates matches.csv and players.csv files
"""

import os
import json
import csv
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import time
from pathlib import Path

# Configuration
API_BASE = "https://api.cricapi.com/v1"
API_KEY = "f03db4b7-c889-4ae9-938c-71eac676591f"
# CRITICAL FIX: Point to parent directory where Cricsheet folder is located
CRICSHEET_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Cricsheet")
# CRITICAL FIX: Point to parent directory where CSV files are located
OUTPUT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mapping for league names
LEAGUE_MAPPING = {
    "IPL": "Indian Premier League",
    "T20 Blast": "T20 Blast",
    "Vitality Blast": "T20 Blast",
    "NTB": "T20 Blast",
    "MLC": "Major League Cricket",
    "CPL": "Caribbean Premier League",
    "SMAT": "Syed Mushtaq Ali Trophy",
    "BBL": "Big Bash League",
    "SA20": "SA20",
    "WPL": "Women's Premier League", 
    "WBBL": "Women's Big Bash League",
    "LPL": "Lanka Premier League",
    "Super Smash": "Super Smash",
    "ILT20": "International League T20",
    "PSL": "Pakistan Super League",
    "BPL": "Bangladesh Premier League",
    "ICC Men's T20 World Cup": "ICC Men's T20 World Cup",
    "ICC Women's T20 World Cup": "ICC Women's T20 World Cup"
}

class CricketETL:
    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.player_cache = {}  # Cache keyed by player name for backward compatibility
        self.player_id_cache = {}  # Cache keyed by Cricsheet identifier for robust lookups
        self.series_cache = {}
        
        # Load Cricsheet player mappings
        self.names_df = self.load_cricsheet_names()
        
        # Load persistent player cache for API cost reduction
        self.persistent_player_cache = self.load_player_cache()
        
        # **FIX: Load existing historical data for proper historical calculations**
        self.existing_player_data = self.load_existing_players_data()
        
    def load_cricsheet_names(self) -> pd.DataFrame:
        """Load Cricsheet names.csv for player mapping"""
        try:
            names_path = os.path.join(CRICSHEET_BASE, "names.csv")
            
            if not os.path.exists(names_path):
                print(f"Warning: names.csv not found at {names_path}")
                return pd.DataFrame()
            
            names_df = pd.read_csv(names_path)
            print(f"Loaded {len(names_df)} name mappings from names.csv")
            
            # Display column info for debugging
            print(f"Names CSV columns: {list(names_df.columns)}")
            if not names_df.empty:
                print("Sample name mappings:")
                print(names_df.head(3))
            
            return names_df
        except Exception as e:
            print(f"Warning: Could not load Cricsheet name mappings: {e}")
            return pd.DataFrame()
    
    def load_player_cache(self) -> Dict:
        """Load persistent player cache from file"""
        cache_path = os.path.join(OUTPUT_BASE, "player_cache.json")
        
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"Loaded player cache with {len(cache)} players from {cache_path}")
                return cache
            else:
                print(f"No existing player cache found, starting with empty cache")
                return {}
        except Exception as e:
            print(f"Warning: Could not load player cache: {e}")
            return {}
    
    def save_player_cache(self):
        """Save persistent player cache to file"""
        cache_path = os.path.join(OUTPUT_BASE, "player_cache.json")
        
        try:
            # Ensure output directory exists
            os.makedirs(OUTPUT_BASE, exist_ok=True)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.persistent_player_cache, f, indent=2, ensure_ascii=False)
            print(f"Saved player cache with {len(self.persistent_player_cache)} players to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save player cache: {e}")
    
    def load_existing_players_data(self) -> List[Dict]:
        """**FIX: Load existing players.csv to use for historical calculations**"""
        players_file = os.path.join(OUTPUT_BASE, "players.csv")
        
        try:
            if os.path.exists(players_file):
                df = pd.read_csv(players_file)
                print(f"📂 Loaded {len(df)} raw records from existing players.csv")
                
                # **CRITICAL FIX: Remove duplicates before using for historical calculations**
                # Keep the last occurrence (most recent with historical data)
                df = df.drop_duplicates(subset=['cricsheet_match_id', 'name'], keep='last')
                print(f"✅ After deduplication: {len(df)} unique player records for historical calculations")
                
                # Convert to list of dicts
                existing_data = df.to_dict('records')
                
                # Ensure date format consistency (YYYY-MM-DD)
                for record in existing_data:
                    date_str = record.get('date', '')
                    if date_str and '-' in date_str:
                        try:
                            # If it's DD-MM-YYYY, convert to YYYY-MM-DD  
                            if len(date_str.split('-')[0]) <= 2:
                                parsed_date = datetime.strptime(date_str, '%d-%m-%Y')
                                record['date'] = parsed_date.strftime('%Y-%m-%d')
                        except ValueError:
                            pass  # Already in correct format
                
                return existing_data
            else:
                print("INFO: No existing players.csv found - starting fresh")
                return []
        except Exception as e:
            print(f"WARNING: Could not load existing players.csv: {e}")
            return []
    
    def api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting and error handling"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{API_BASE}/{endpoint}"
        
        try:
            print(f"    API call: {endpoint} with params: {params.get('search', params.get('id', params.get('series', params.get('series_id', ''))))}")
            response = self.session.get(url, params=params, timeout=15)
            
            print(f"    HTTP Status: {response.status_code}")
            
            if response.status_code == 429:  # Rate limit
                print(f"    Rate limited, waiting 2 seconds...")
                time.sleep(2)
                response = self.session.get(url, params=params, timeout=15)
            
            response.raise_for_status()
            result = response.json()
            
            if 'data' in result:
                data_count = len(result['data']) if isinstance(result['data'], list) else 1
                print(f"    API response: {data_count} results")
            else:
                print(f"    API response keys: {list(result.keys())}")
                
            time.sleep(0.3)  # Rate limiting
            return result
            
        except requests.exceptions.Timeout:
            print(f"    API request timed out for {endpoint}")
            return {}
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP error for {endpoint}: {e}")
            print(f"    Response text: {response.text[:500] if 'response' in locals() else 'N/A'}")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"    API request failed for {endpoint}: {e}")
            return {}
        except ValueError as e:
            print(f"    JSON decode error for {endpoint}: {e}")
            print(f"    Response text: {response.text[:500] if 'response' in locals() else 'N/A'}")
            return {}
        except Exception as e:
            print(f"    Unexpected error for {endpoint}: {e}")
            return {}
    
    def parse_readme(self, series_folder: str, filter_year: int = None) -> List[Dict]:
        """Parse README.txt to extract match information"""
        readme_path = os.path.join(series_folder, "README.txt")
        matches = []
        
        if not os.path.exists(readme_path):
            print(f"README.txt not found in {series_folder}")
            return matches
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
                
            parts = line.split(' - ')
            if len(parts) >= 6:
                date_str = parts[0]
                team_type = parts[1]
                match_type = parts[2]
                gender = parts[3]
                match_id = parts[4]
                teams_str = parts[5]
                
                # Only process matches from 2022 onwards (unless specific year filter is applied)
                try:
                    match_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Apply year filter if specified
                    if filter_year:
                        if match_date.year != filter_year:
                            continue
                    else:
                        # Default: only matches from 2022 onwards
                        if match_date.year < 2022:
                            continue
                            
                except ValueError:
                    continue
                
                # Parse teams
                teams = [team.strip() for team in teams_str.split(' vs ')]
                if len(teams) == 2:
                    matches.append({
                        'date': date_str,
                        'team_type': team_type,
                        'match_type': match_type,
                        'gender': gender,
                        'cricsheet_match_id': match_id,
                        'team1': teams[0],
                        'team2': teams[1],
                        'json_file': f"{match_id}.json"
                    })
        
        return matches
    
    def load_cricsheet_json(self, json_path: str) -> Dict:
        """Load and parse Cricsheet JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return {}
    
    def extract_match_metadata(self, cricsheet_data: Dict, readme_match: Dict) -> Dict:
        """Extract match metadata from Cricsheet JSON"""
        info = cricsheet_data.get('info', {})
        
        # Basic match info
        match_data = {
            'cricsheet_match_id': readme_match['cricsheet_match_id'],
            'date': readme_match['date'],
            'venue': info.get('venue', ''),
            'city': info.get('city', ''),
            'league': self.determine_league(info),
            'gender': info.get('gender', readme_match.get('gender', '')),
            'teams': info.get('teams', [readme_match['team1'], readme_match['team2']]),
            'players': info.get('players', {}),
            'toss': info.get('toss', {}),
            'outcome': info.get('outcome', {}),
            'overs': info.get('overs', 20)
        }
        
        return match_data
    
    def determine_league(self, info: Dict) -> str:
        """Determine league name from Cricsheet info"""
        event = info.get('event', {})
        event_name = event.get('name', '') if isinstance(event, dict) else str(event)
        
        for key, value in LEAGUE_MAPPING.items():
            if key.lower() in event_name.lower():
                return value
        
        return event_name
    
    def determine_match_gender(self, match_data: Dict, player_stats: Dict) -> str:
        """Determine if match is Men's or Women's cricket using multiple indicators"""
        
        # Strategy 1: Check league/series name for obvious indicators
        league = match_data.get('league', '').lower()
        series_indicators = {
            'women': ['women', 'wbbl', 'wpl', 'female'],
            'men': ['men', 'male']
        }
        
        for gender, keywords in series_indicators.items():
            if any(keyword in league for keyword in keywords):
                return 'Women' if gender == 'women' else 'Men'
        
        # Strategy 2: Check Cricsheet gender field if available
        cricsheet_gender = match_data.get('gender', '').lower()
        if cricsheet_gender in ['female', 'women']:
            return 'Women'
        elif cricsheet_gender in ['male', 'men']:
            return 'Men'
        
        # Strategy 3: Analyze player names using cached API data
        # Look for players we've successfully identified with gender indicators
        women_indicators = 0
        men_indicators = 0
        total_identified_players = 0
        
        for player_name in player_stats.keys():
            # Check if we have cached player info
            if hasattr(self, 'player_cache') and player_name in self.player_cache:
                player_info = self.player_cache[player_name]
                player_full_name = player_info.get('name', player_name).lower()
                
                # Common female cricket player first names
                female_names = [
                    'sarah', 'meg', 'ellyse', 'alyssa', 'beth', 'sophie', 'kate', 'katie',
                    'laura', 'georgia', 'rachael', 'jess', 'jessica', 'anna', 'anne',
                    'delissa', 'nicole', 'natasha', 'amanda', 'samantha', 'sam',
                    'hayley', 'hannah', 'emma', 'emily', 'amy', 'chloe', 'grace',
                    'stafanie', 'deandra', 'hayley', 'shemaine', 'britney', 'kycia',
                    'chinelle', 'natasha', 'chedean', 'shakera', 'aaliyah', 'qiana',
                    'chamari', 'harshitha', 'kavisha', 'nilakshi', 'oshadi', 'hasini',
                    'rashmi', 'sripali', 'madhuri', 'sanduni', 'imesha', 'anushka',
                    'richa', 'priya', 'mithali', 'harmanpreet', 'smriti', 'jemimah',
                    'deepti', 'radha', 'shafali', 'yastika', 'taniya', 'sneh', 'pooja',
                    'jemma', 'tahlia', 'ashleigh', 'megan', 'nicola', 'lauren', 'erin',
                    'heather', 'kim', 'marizanne', 'shabnim', 'sune', 'tazmin', 'danni',
                    'dane', 'lizelle', 'ayabonga', 'nonkululeko', 'masabata', 'tumi'
                ]
                
                # Extract first name from full player name
                name_parts = player_full_name.split()
                first_name = name_parts[0] if name_parts else ''
                
                # Check against female names list
                if first_name in female_names:
                    women_indicators += 1
                    total_identified_players += 1
                elif len(first_name) > 2:  # Only count meaningful names
                    # Common male indicators (less specific but still useful)
                    male_names = [
                        'mohammed', 'mohammad', 'muhammad', 'abdul', 'ahmed', 'ali',
                        'david', 'michael', 'james', 'john', 'robert', 'chris', 'andrew',
                        'shane', 'steve', 'mark', 'paul', 'matthew', 'daniel', 'jason',
                        'virat', 'rohit', 'ms', 'mahendra', 'kapil', 'sachin', 'rahul',
                        'hardik', 'jasprit', 'ravindra', 'ravichandran', 'bhuvneshwar',
                        'kane', 'trent', 'ross', 'martin', 'tim', 'mitchell', 'neil',
                        'joe', 'ben', 'jos', 'eoin', 'jonny', 'moeen', 'adil', 'chris',
                        'steve', 'mitchell', 'pat', 'josh', 'glenn', 'aaron', 'marcus',
                        'quinton', 'faf', 'ab', 'hashim', 'vernon', 'dale', 'kagiso'
                    ]
                    
                    if first_name in male_names or any(male_name in first_name for male_name in ['mohammad', 'mohammed', 'muhammad']):
                        men_indicators += 1
                        total_identified_players += 1
        
        # Strategy 4: If we have sufficient player name analysis, use it
        if total_identified_players >= 3:  # Need at least 3 identified players for confidence
            women_ratio = women_indicators / total_identified_players
            if women_ratio >= 0.3:  # 30% or more female names indicates women's cricket
                return 'Women'
            elif women_ratio == 0:  # No female names detected
                return 'Men'
        
        # Strategy 5: Check team names for women's specific patterns
        teams = match_data.get('teams', [])
        for team in teams:
            team_lower = team.lower()
            if any(indicator in team_lower for indicator in ['women', 'ladies', 'female']):
                return 'Women'
        
        # Strategy 6: League-specific defaults based on known patterns
        league_defaults = {
            'wbbl': 'Women',
            'wpl': 'Women',
            'women\'s premier league': 'Women',
            'women\'s big bash league': 'Women',
            'icc women\'s t20 world cup': 'Women',
            'indian premier league': 'Men',
            'big bash league': 'Men',
            'caribbean premier league': 'Men',
            't20 blast': 'Men',
            'pakistan super league': 'Men',
            'sa20': 'Men',
            'major league cricket': 'Men',
            'lanka premier league': 'Men',
            'super smash': 'Men',
            'international league t20': 'Men',
            'bangladesh premier league': 'Men',
            'icc men\'s t20 world cup': 'Men',
            'syed mushtaq ali trophy': 'Men'
        }
        
        for pattern, gender in league_defaults.items():
            if pattern in league:
                return gender
        
        # Default: assume Men's cricket (as it's more common in T20 leagues)
        return 'Men'
    
    def extract_deliveries(self, cricsheet_data: Dict) -> List[Dict]:
        """Extract ball-by-ball data from Cricsheet JSON"""
        deliveries = []
        
        for innings_idx, innings in enumerate(cricsheet_data.get('innings', [])):
            team = innings.get('team', '')
            
            for over_data in innings.get('overs', []):
                over_num = over_data.get('over', 0)
                
                for ball_idx, delivery in enumerate(over_data.get('deliveries', [])):
                    delivery_data = {
                        'innings': innings_idx + 1,
                        'team': team,
                        'over': over_num,
                        'ball': ball_idx + 1,
                        'batter': delivery.get('batter', ''),
                        'bowler': delivery.get('bowler', ''),
                        'runs_batter': delivery.get('runs', {}).get('batter', 0),
                        'runs_extras': delivery.get('runs', {}).get('extras', 0),
                        'runs_total': delivery.get('runs', {}).get('total', 0),
                        'extras': delivery.get('extras', {}),
                        'wickets': delivery.get('wickets', [])
                    }
                    deliveries.append(delivery_data)
        
        return deliveries
    
    def calculate_player_stats(self, deliveries: List[Dict], match_data: Dict) -> Dict:
        """Calculate player statistics from deliveries"""
        player_stats = {}
        
        # Initialize stats for all players
        for team, players in match_data['players'].items():
            for player in players:
                player_stats[player] = {
                    'team': team,
                    'balls_faced': 0,
                    'runs_scored': 0,
                    'balls_bowled': 0,
                    'overs_bowled': 0.0,
                    'runs_conceded': 0,
                    'wickets_taken': 0,
                    'batting_order': None,
                    'bowling_phases_detail': {'powerplay': 0.0, 'middle': 0.0, 'death': 0.0}  # Store overs directly
                }
        
        # Track batting order per team (reset for each innings)
        batting_order_by_team = {}
        
        for delivery in deliveries:
            batter = delivery['batter']
            bowler = delivery['bowler']
            over_num = delivery['over']
            
            # Batting stats
            if batter in player_stats:
                player_stats[batter]['balls_faced'] += 1
                player_stats[batter]['runs_scored'] += delivery['runs_batter']
                
                # Track batting order per team
                batter_team = player_stats[batter]['team']
                if batter_team not in batting_order_by_team:
                    batting_order_by_team[batter_team] = {}
                
                if batter not in batting_order_by_team[batter_team]:
                    batting_order_by_team[batter_team][batter] = len(batting_order_by_team[batter_team]) + 1
                    player_stats[batter]['batting_order'] = batting_order_by_team[batter_team][batter]
            
            # Bowling stats
            if bowler in player_stats:
                player_stats[bowler]['balls_bowled'] += 1
                player_stats[bowler]['runs_conceded'] += delivery['runs_total']
                
                # Bowling phases (T20 format) - store overs directly
                if over_num < 6:  # Powerplay
                    player_stats[bowler]['bowling_phases_detail']['powerplay'] += 1/6  # Convert balls to overs immediately
                elif over_num < 15:  # Middle overs
                    player_stats[bowler]['bowling_phases_detail']['middle'] += 1/6
                else:  # Death overs
                    player_stats[bowler]['bowling_phases_detail']['death'] += 1/6
                
                # Count wickets
                if delivery['wickets']:
                    player_stats[bowler]['wickets_taken'] += len(delivery['wickets'])
        
        # Convert balls bowled to overs
        for player in player_stats:
            balls = player_stats[player]['balls_bowled']
            player_stats[player]['overs_bowled'] = balls / 6.0
        
        # Assign batting order for players who didn't bat (per team)
        for team, players in match_data['players'].items():
            # Get current max batting order for this team
            team_batting_orders = [
                player_stats[p]['batting_order'] 
                for p in players 
                if player_stats[p]['batting_order'] is not None
            ]
            next_order = max(team_batting_orders) + 1 if team_batting_orders else 1
            
            # Assign order to unassigned players in this team
            for player in players:
                if player_stats[player]['batting_order'] is None:
                    if next_order <= 11:  # T20 teams have max 11 players
                        player_stats[player]['batting_order'] = next_order
                        next_order += 1
                    else:
                        # For substitute players or extras, don't assign batting order > 11
                        player_stats[player]['batting_order'] = None
        
        return player_stats
    
    def get_player_info_from_api(self, player_name: str, cricsheet_data: Dict = None, match_context: Dict = None) -> Dict:
        """Get player information from Cricketdata.org API with enhanced search strategies and persistent caching"""
        # Get Cricsheet identifier first
        cricsheet_identifier = ''
        if cricsheet_data:
            cricsheet_identifier = self.get_cricsheet_identifier(player_name, cricsheet_data)
            print(f"  Cricsheet ID: {cricsheet_identifier}")
        
        # FIRST: Check persistent cache by cricsheet identifier (most robust)
        if cricsheet_identifier and cricsheet_identifier in self.persistent_player_cache:
            cached_info = self.persistent_player_cache[cricsheet_identifier].copy()
            cached_info['cricsheet_identifier'] = cricsheet_identifier
            print(f"  ✅ Using CACHED player data (no API call needed)")
            return cached_info
        
        # SECOND: Check persistent cache by player name (fallback)  
        for cache_key, cache_data in self.persistent_player_cache.items():
            if cache_data.get('name', '').lower() == player_name.lower():
                cached_info = cache_data.copy()
                cached_info['cricsheet_identifier'] = cricsheet_identifier
                print(f"  ✅ Using CACHED player data by name match (no API call needed)")
                return cached_info
        
        # THIRD: Check old memory-based cache (backward compatibility)
        if cricsheet_identifier and cricsheet_identifier in self.player_id_cache:
            cached_info = self.player_id_cache[cricsheet_identifier].copy()
            cached_info['cricsheet_identifier'] = cricsheet_identifier
            return cached_info
        
        # FOURTH: Fallback to name-based cache
        if player_name in self.player_cache:
            cached_info = self.player_cache[player_name].copy()
            cached_info['cricsheet_identifier'] = cricsheet_identifier
            return cached_info
        
        player_info = {
            'player_id': '',
            'name': player_name,
            'role': '',
            'batting_style': '',
            'bowling_style': '',
            'country': '',
            'cricsheet_identifier': cricsheet_identifier
        }
        
        try:
            print(f"  Searching API for: {player_name}")
            
            # Strategy 1: Direct name search
            search_result = self.api_request("players", {"search": player_name})
            found_player = self.select_best_player_match(search_result.get('data', []), player_name, match_context)
            
            if not found_player:
                print(f"  No results for {player_name}, trying alternatives...")
                # Strategy 2: Try alternative names from Cricsheet mapping
                alt_names = self.get_alternative_names(player_name, cricsheet_identifier)
                print(f"  Found {len(alt_names)} alternative names: {alt_names[:5]}")
                
                for alt_name in alt_names[:5]:
                    print(f"  Trying alternative: {alt_name}")
                    search_result = self.api_request("players", {"search": alt_name})
                    found_player = self.select_best_player_match(search_result.get('data', []), alt_name, match_context)
                    if found_player:
                        print(f"  Found match with alternative: {alt_name}")
                        break
            
            if not found_player:
                # Strategy 3: Try surname-only search
                surname = self.extract_surname(player_name)
                if surname and len(surname) > 2:  # Avoid very short surnames
                    print(f"  Trying surname search: {surname}")
                    search_result = self.api_request("players", {"search": surname})
                    found_player = self.select_best_player_match(search_result.get('data', []), player_name, match_context)
                    if found_player:
                        print(f"  Found match with surname: {surname}")
            
            if not found_player:
                # Strategy 4: Try partial name variations
                variations = self.generate_name_variations(player_name)
                for variation in variations[:3]:  # Limit variations
                    print(f"  Trying variation: {variation}")
                    search_result = self.api_request("players", {"search": variation})
                    found_player = self.select_best_player_match(search_result.get('data', []), variation, match_context)
                    if found_player:
                        print(f"  Found match with variation: {variation}")
                        break
            
            if found_player:
                player_id = found_player.get('id', '')
                print(f"  Selected player ID: {player_id} ({found_player.get('name', '')})")
                
                if player_id:
                    # Get detailed player info
                    detailed_info = self.api_request("players_info", {"id": player_id})
                    if detailed_info.get('data'):
                        detail_data = detailed_info['data']
                        player_info.update({
                            'player_id': player_id,
                            'name': detail_data.get('name', player_name),
                            'role': self.normalize_role(detail_data.get('role', '')),
                            'batting_style': detail_data.get('battingStyle', ''),
                            'bowling_style': detail_data.get('bowlingStyle', ''),
                            'country': detail_data.get('country', ''),
                            'cricsheet_identifier': cricsheet_identifier
                        })
                        print(f"  Final player info: {player_info['name']}, {player_info['role']}, {player_info['country']}")
            else:
                print(f"  No player found after all strategies for: {player_name}")
                        
        except Exception as e:
            print(f"  Error fetching player info for {player_name}: {e}")
        
        # Cache the result both by name and identifier (legacy caches)
        self.player_cache[player_name] = player_info
        if cricsheet_identifier:
            self.player_id_cache[cricsheet_identifier] = player_info
        
        # SAVE TO PERSISTENT CACHE (for massive API cost reduction)
        if player_info.get('player_id') and cricsheet_identifier:
            # Only cache if we actually found the player in API
            cache_data = {
                'player_id': player_info['player_id'],
                'name': player_info['name'],
                'role': player_info['role'],
                'batting_style': player_info['batting_style'],
                'bowling_style': player_info['bowling_style'],
                'country': player_info['country'],
                'cached_date': datetime.now().strftime('%Y-%m-%d'),
                'alternative_names': [player_name] if player_name != player_info['name'] else []
            }
            self.persistent_player_cache[cricsheet_identifier] = cache_data
            print(f"  💾 CACHED player data for future runs (identifier: {cricsheet_identifier})")
        
        return player_info
    
    def get_cricsheet_identifier(self, player_name: str, cricsheet_data: Dict) -> str:
        """Get Cricsheet identifier for a player from the match registry"""
        registry = cricsheet_data.get('info', {}).get('registry', {})
        people = registry.get('people', {})
        
        # Direct lookup
        if player_name in people:
            identifier = people[player_name]
            print(f"    Found identifier '{identifier}' for '{player_name}'")
            return identifier
        
        print(f"    No identifier found for '{player_name}' in registry.people")
        return ''
    
    def get_alternative_names(self, player_name: str, cricsheet_identifier: str = '') -> List[str]:
        """Get alternative names for a player from Cricsheet mappings"""
        if self.names_df.empty:
            print(f"    Names DataFrame is empty!")
            return []
        
        alt_names = []
        
        # If we have a Cricsheet identifier, use it to find all name variations
        if cricsheet_identifier:
            print(f"    Searching for identifier: {cricsheet_identifier}")
            # Find all names associated with this identifier
            matching_rows = self.names_df[self.names_df['identifier'] == cricsheet_identifier]
            print(f"    Found {len(matching_rows)} matching rows")
            
            for _, row in matching_rows.iterrows():
                if pd.notna(row.get('name')):
                    alt_names.append(row['name'])
                    print(f"      Added: {row['name']}")
        else:
            # Fallback: try to find by name match
            matching_rows = self.names_df[
                (self.names_df['name'] == player_name)
            ]
            
            for _, row in matching_rows.iterrows():
                if pd.notna(row.get('name')):
                    alt_names.append(row['name'])
        
        # Remove duplicates and the original name
        alt_names = list(set(alt_names))
        if player_name in alt_names:
            alt_names.remove(player_name)
        
        print(f"    Final alternatives: {alt_names}")
        return alt_names
    
    def normalize_role(self, role: str) -> str:
        """Normalize player role to standard format"""
        if not role:
            return ''
        
        role_lower = role.lower().strip()
        
        # Wicket keeper patterns
        if any(pattern in role_lower for pattern in ['wicket', 'keeper', 'wk', 'wicketkeeper', 'wicket-keeper']):
            return 'WK'
        
        # All-rounder patterns
        elif any(pattern in role_lower for pattern in ['allround', 'all-round', 'all round', 'ar', 'batting allrounder', 'bowling allrounder']):
            return 'AR'
        
        # Bowler patterns
        elif any(pattern in role_lower for pattern in ['bowl', 'bowling', 'fast', 'spin', 'pace', 'seam', 'medium']):
            return 'BOWL'
        
        # Batsman patterns
        elif any(pattern in role_lower for pattern in ['bat', 'batting', 'batsman', 'batter', 'top order', 'middle order', 'opener']):
            return 'BAT'
        
        # Handle empty or dash cases
        elif role_lower in ['', '-', '--', 'n/a', 'na', 'unknown']:
            return ''
        
        # If no pattern matches but role is short, return uppercase
        elif len(role) <= 4:
            return role.upper()
        
        # Default fallback for longer unrecognized roles
        return ''
    
    def calculate_fantasy_points(self, player_stats: Dict, deliveries: List[Dict]) -> Dict:
        """Calculate fantasy points for each player (CORRECT Dream11 format)"""
        fantasy_points = {}
        
        # Count boundary bonuses for each player
        boundary_bonuses = {}
        catch_bonuses = {}
        
        for delivery in deliveries:
            batter = delivery['batter']
            runs_batter = delivery['runs_batter']
            
            # Boundary bonuses: +4 for four, +6 for six 
            if batter not in boundary_bonuses:
                boundary_bonuses[batter] = 0
            if runs_batter == 4:
                boundary_bonuses[batter] += 4  # +4 bonus for four
            elif runs_batter == 6:
                boundary_bonuses[batter] += 6  # +6 bonus for six
            
            # Fielding points: +4 per catch
            if delivery.get('wickets'):
                for wicket in delivery['wickets']:
                    fielder = wicket.get('fielders', [{}])[0].get('name') if wicket.get('fielders') else None
                    if fielder and wicket.get('kind') == 'caught':
                        if fielder not in catch_bonuses:
                            catch_bonuses[fielder] = 0
                        catch_bonuses[fielder] += 4  # +4 per catch
        
        for player, stats in player_stats.items():
            points = 0.0
            
            # Batting points
            points += stats['runs_scored'] * 1.0  # 1 point per run
            
            # **CORRECT Dream11 batting bonuses**
            if stats['runs_scored'] >= 25:
                points += 8  # Bonus for 25+ runs
            if stats['runs_scored'] >= 50:
                points += 4  # Additional bonus for 50+ runs  
            if stats['runs_scored'] >= 75:
                points += 4  # Additional bonus for 75+ runs
            if stats['runs_scored'] >= 100:
                points += 8  # Additional bonus for 100+ runs
            
            # Boundary bonuses
            points += boundary_bonuses.get(player, 0)
            
            # **CORRECT Dream11 bowling points**
            points += stats['wickets_taken'] * 30  # **30 points per wicket (not 25)**
            if stats['wickets_taken'] >= 3:
                points += 4  # Bonus for 3+ wickets
            if stats['wickets_taken'] >= 4:
                points += 8  # **Enhanced bonus for 4+ wickets**
            if stats['wickets_taken'] >= 5:
                points += 8  # Additional bonus for 5+ wickets
            
            # Fielding points
            points += catch_bonuses.get(player, 0)
            
            # Economy rate bonus/penalty (for bowlers who bowled at least 2 overs)
            if stats['overs_bowled'] >= 2:
                economy_rate = stats['runs_conceded'] / stats['overs_bowled']
                if economy_rate <= 5:
                    points += 6
                elif economy_rate <= 6:
                    points += 4
                elif economy_rate <= 7:
                    points += 2
                elif economy_rate >= 10:
                    points -= 2
                elif economy_rate >= 11:
                    points -= 4
                elif economy_rate >= 12:
                    points -= 6
            
            fantasy_points[player] = points
        
        return fantasy_points
    
    def process_series(self, series_folder: str, limit: int = None, filter_year: int = None) -> Tuple[List[Dict], List[Dict]]:
        """Process all matches in a series"""
        print(f"Processing series: {os.path.basename(series_folder)}")
        if filter_year:
            print(f"Filtering matches for year: {filter_year}")
        
        matches_data = []
        players_data = []
        # Load existing player data for historical calculations
        all_players_temp = self.load_existing_players_data()
        
        # Parse README to get match list (with optional year filter)
        readme_matches = self.parse_readme(series_folder, filter_year)
        if filter_year:
            print(f"Found {len(readme_matches)} matches for year {filter_year}")
        else:
            print(f"Found {len(readme_matches)} matches from 2022 onwards")
        
        # **CRITICAL FIX: Sort matches chronologically (oldest first) for proper historical data building**
        readme_matches.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        print(f"✅ Sorted {len(readme_matches)} matches chronologically (oldest first)")
        
        # Apply limit if specified (after sorting)
        if limit:
            readme_matches = readme_matches[:limit]
            print(f"Processing only first {len(readme_matches)} matches due to limit")
        
        for match_info in readme_matches:
            json_path = os.path.join(series_folder, match_info['json_file'])
            
            if not os.path.exists(json_path):
                print(f"JSON file not found: {json_path}")
                continue
            
            print(f"Processing match: {match_info['cricsheet_match_id']}")
            
            # Load Cricsheet data
            cricsheet_data = self.load_cricsheet_json(json_path)
            if not cricsheet_data:
                continue
            
            # Check if match was abandoned/incomplete
            if self.is_match_abandoned(cricsheet_data):
                print(f"  Skipping abandoned match: {match_info['cricsheet_match_id']}")
                continue
            
            # Extract match metadata
            match_data = self.extract_match_metadata(cricsheet_data, match_info)
            
            # Extract deliveries
            deliveries = self.extract_deliveries(cricsheet_data)
            
            # Calculate player stats
            player_stats = self.calculate_player_stats(deliveries, match_data)
            
            # Calculate fantasy points
            fantasy_points = self.calculate_fantasy_points(player_stats, deliveries)
            
            # Get API series and match information
            league = match_data['league']
            year = datetime.strptime(match_data['date'], '%Y-%m-%d').year
            series_info = self.get_series_info_from_api(league, year)
            
            # **FIX: Use fallback series info instead of skipping matches**
            if not series_info.get('series_id'):
                print(f"  ⚠️ No API series ID found for {league} {year} - using fallback")
                # Create fallback series info
                series_info = {
                    'series_id': f"fallback_{league.lower().replace(' ', '_')}_{year}",
                    'series_name': f"{league} {year}",
                    'start_date': f"{year}-01-01",
                    'end_date': f"{year}-12-31"
                }
                print(f"    Using fallback series ID: {series_info['series_id']}")
            
            # Get match info from API
            api_match_info = {'match_id': '', 'date': match_data['date']}
            teams = match_data.get('teams', [])
            if len(teams) >= 2:
                api_match_info = self.get_match_info_from_api(
                    series_info['series_id'], 
                    teams[0], 
                    teams[1], 
                    match_data['date']
                )
                
                # **FIX: Use fallback match info instead of skipping matches**
                if not api_match_info.get('match_id'):
                    print(f"  ⚠️ No API match ID found for {teams[0]} vs {teams[1]} on {match_data['date']} - using fallback")
                    # Create fallback match info
                    api_match_info = {
                        'match_id': f"fallback_{match_info['cricsheet_match_id']}",
                        'date': match_data['date'],
                        'venue': match_data.get('venue', ''),
                        'teams': teams
                    }
                    print(f"    Using fallback match ID: {api_match_info['match_id']}")
                
                # If we got a match ID from the API, fetch detailed match info
                if not api_match_info['match_id'].startswith('fallback_'):
                    print(f"    Found API match ID: {api_match_info['match_id']}, fetching detailed match info...")
                    detailed_match_info = self.get_match_details_by_id(api_match_info['match_id'])
                    if detailed_match_info:
                        # Update with detailed info from match_info endpoint
                        api_match_info.update(detailed_match_info)
                        print(f"    Updated match info with API details. Date: {api_match_info.get('date', 'N/A')}")
                
                # Use API date if available, otherwise fall back to Cricsheet date
                if not api_match_info.get('date'):
                    api_match_info['date'] = match_data['date']
            else:
                print(f"  ❌ INVALID MATCH: Insufficient team data - SKIPPING match {match_info['cricsheet_match_id']}")
                continue
            
            # Create match record
            toss = match_data.get('toss', {})
            outcome = match_data.get('outcome', {})
            
            # Determine batting first team
            toss_winner = toss.get('winner', '')
            toss_decision = toss.get('decision', '')
            teams = match_data.get('teams', [])
            
            if len(teams) == 2:
                if toss_decision.lower() == 'bat':
                    batting_first_team = toss_winner
                    chasing_team = teams[1] if teams[0] == toss_winner else teams[0]
                elif toss_decision.lower() in ['field', 'bowl']:
                    chasing_team = toss_winner
                    batting_first_team = teams[1] if teams[0] == toss_winner else teams[0]
                else:
                    # Fallback: use team order from match data
                    batting_first_team = teams[0]
                    chasing_team = teams[1]
            else:
                batting_first_team = ''
                chasing_team = ''
            
            # Determine margin
            margin_by = ''
            margin_value = 0
            if 'by' in outcome:
                by_data = outcome['by']
                if 'runs' in by_data:
                    margin_by = 'runs'
                    margin_value = by_data['runs']
                elif 'wickets' in by_data:
                    margin_by = 'wickets'
                    margin_value = by_data['wickets']
            
            # Use fallback logic: prefer Cricsheet data for toss/winner, use API for venue/date
            final_toss_winner = toss_winner or api_match_info.get('toss_winner', '')
            final_toss_decision = toss_decision or api_match_info.get('toss_decision', '')
            final_winner = outcome.get('winner', '') or api_match_info.get('match_winner', '')
            
            # Determine match gender using multiple strategies
            match_gender = self.determine_match_gender(match_data, player_stats)
            
            match_record = {
                'cricsheet_match_id': match_data['cricsheet_match_id'],
                'api_match_id': api_match_info.get('match_id', '') or '',
                'api_series_id': series_info.get('series_id', '') or '',
                'series_id': f"{match_data['league']}_{datetime.strptime(api_match_info.get('date', match_data['date']), '%Y-%m-%d').year}",
                'date': api_match_info.get('date', match_data['date']),  # Use API date if available
                'venue': api_match_info.get('venue', match_data['venue']),  # Prefer API venue
                'league': match_data['league'],
                'gender': match_gender,  # New gender field
                'toss_winner': final_toss_winner,  # Prefer Cricsheet data, fallback to API
                'toss_decision': final_toss_decision,  # Prefer Cricsheet data, fallback to API
                'batting_first_team': batting_first_team,
                'chasing_team': chasing_team,
                'winner': final_winner,  # Prefer Cricsheet data, fallback to API
                'margin_by': margin_by,
                'margin_value': margin_value,
                'pitch': None  # New column for pitch information
            }
            
            matches_data.append(match_record)
            
            # **DNP HANDLING: Get all players from squads for DNP tracking**
            all_squad_players = set()
            for team, players in match_data['players'].items():
                all_squad_players.update(players)
            
            # **Track which players actually played (have stats)**
            players_who_played = set(player_stats.keys())
            
            # **Create records for players who played**
            for player_name, stats in player_stats.items():
                print(f"Processing player: {player_name}")
                
                # Get player info from API with match context
                match_context = {
                    'league': match_data['league'],
                    'date': match_data['date'],
                    'venue': match_data['venue'],
                    'team': stats['team']
                }
                player_info = self.get_player_info_from_api(player_name, cricsheet_data, match_context)
                
                # Initialize actual_match_date early to prevent variable access errors
                actual_match_date = api_match_info.get('date', match_data['date'])
                
                # Determine bowling phases from historical matches (pre-match feature)
                bowling_phases = [0.00, 0.00, 0.00]  # Default for non-bowlers
                if stats['overs_bowled'] > 0:  # Only for bowlers
                    # Get historical bowling phase proportions from past matches
                    bowling_phases = self.get_historical_bowling_phases(player_name, actual_match_date, all_players_temp)
                
                # Ensure batting order is present (assign default if player didn't bat)
                batting_order = stats['batting_order']
                if batting_order is None:
                    # For non-batting players (substitutes, etc.), use a default value
                    # This helps with analysis while maintaining data integrity
                    batting_order = 12  # Indicates non-batting player
                
                # Enhanced role assignment with better fallback logic
                role = player_info.get('role', '')
                if not role:
                    # Try to infer role from match statistics
                    overs_bowled = stats['overs_bowled']
                    balls_faced = stats['balls_faced']
                    wickets_taken = stats['wickets_taken']
                    
                    # Check if player is a wicket keeper (look for keeper-like names)
                    player_name_lower = player_name.lower()
                    if any(term in player_name_lower for term in ['keeper', 'kiper']) or stats['team'] in match_data.get('players', {}):
                        # Additional check: see if this player is listed in a keeper position
                        team_players = match_data.get('players', {}).get(stats['team'], [])
                        if team_players and player_name in team_players[:7]:  # Typically keepers bat in top 7
                            # Check match context for keeper patterns
                            if 'keeper' in player_name_lower or batting_order in [1, 2, 3, 4, 5, 6, 7]:
                                pass  # Could be keeper, but we need more info
                    
                    # Role inference based on performance
                    if overs_bowled >= 2.0 and balls_faced >= 10:
                        role = 'AR'  # All-rounder: bowled significant overs and batted
                    elif overs_bowled >= 3.0:
                        role = 'BOWL'  # Primary bowler
                    elif wickets_taken >= 2:
                        role = 'BOWL'  # Effective bowler
                    elif balls_faced >= 15:
                        role = 'BAT'  # Primary batsman
                    elif overs_bowled > 0:
                        role = 'AR'  # Bowled some overs, likely all-rounder
                    else:
                        role = 'BAT'  # Default to batsman
                
                # Final fallback: if role is still empty or invalid, use stats
                if not role or role == '--':
                    if stats['overs_bowled'] > 1.0:
                        role = 'BOWL'
                    else:
                        role = 'BAT'
                
                # Get historical statistics based on past matches within this series
                # Use the API date for accurate chronological ordering (already assigned above)
                past_stats = self.get_player_past_stats_by_date(player_name, all_players_temp, actual_match_date, series_info.get('series_id', ''))
                
                player_record = {
                    'cricsheet_match_id': match_data['cricsheet_match_id'],
                    'api_match_id': api_match_info.get('match_id', '') or '',
                    'api_series_id': series_info.get('series_id', '') or '',
                    'cricsheet_identifier': player_info.get('cricsheet_identifier', ''),
                    'player_id': player_info['player_id'],
                    'name': player_name,
                    'team': stats['team'],
                    'role': role,
                    'batting_style': player_info.get('batting_style', ''),
                    'bowling_style': player_info.get('bowling_style', ''),
                    'batting_order': batting_order,
                    'balls_faced': stats['balls_faced'],
                    'overs_bowled': round(stats['overs_bowled'], 1),
                    'bowling_phases': str(bowling_phases),
                    'bowling_phases_detail': json.dumps(stats['bowling_phases_detail']),  # Store detailed phase data as JSON
                    'fantasy_points': round(fantasy_points.get(player_name, 0.0), 1),
                    'date': actual_match_date,  # Use API date for accurate chronological ordering
                    'last10_fantasy_scores': json.dumps(past_stats['last_n_fantasy']),
                    'avg_fantasy_points_last5': past_stats['avg_fantasy_5'],
                    'avg_balls_faced_last5': past_stats['avg_balls_5'],
                    'avg_overs_bowled_last5': past_stats['avg_overs_5'],
                    'ownership': None  # New column for ownership data
                }
                
                players_data.append(player_record)
                
                # Add to temporary storage for historical calculations in future matches
                all_players_temp.append(player_record.copy())
            
            # **DNP HANDLING: Create records for players who didn't play (DNP)**
            dnp_players = all_squad_players - players_who_played
            for player_name in dnp_players:
                print(f"Processing DNP player: {player_name}")
                
                # Determine which team this DNP player belongs to
                player_team = None
                for team, players in match_data['players'].items():
                    if player_name in players:
                        player_team = team
                        break
                
                if not player_team:
                    print(f"  Warning: Could not determine team for DNP player {player_name}")
                    continue
                
                # Get player info from API with match context
                match_context = {
                    'league': match_data['league'],
                    'date': match_data['date'],
                    'venue': match_data['venue'],
                    'team': player_team
                }
                player_info = self.get_player_info_from_api(player_name, cricsheet_data, match_context)
                
                # Initialize actual_match_date for DNP player
                actual_match_date = api_match_info.get('date', match_data['date'])
                
                # Get historical statistics for DNP player
                past_stats = self.get_player_past_stats_by_date(player_name, all_players_temp, actual_match_date, series_info.get('series_id', ''))
                
                # Create DNP player record with 0 fantasy points and minimal stats
                dnp_record = {
                    'cricsheet_match_id': match_data['cricsheet_match_id'],
                    'api_match_id': api_match_info.get('match_id', '') or '',
                    'api_series_id': series_info.get('series_id', '') or '',
                    'cricsheet_identifier': player_info.get('cricsheet_identifier', ''),
                    'player_id': player_info['player_id'],
                    'name': player_name,
                    'team': player_team,
                    'role': player_info.get('role', 'BAT'),  # Default role for DNP
                    'batting_style': player_info.get('batting_style', ''),
                    'bowling_style': player_info.get('bowling_style', ''),
                    'batting_order': 13,  # Special value indicating DNP
                    'balls_faced': 0,
                    'overs_bowled': 0.0,
                    'bowling_phases': str([0.00, 0.00, 0.00]),
                    'bowling_phases_detail': json.dumps({'powerplay': 0.0, 'middle': 0.0, 'death': 0.0}),
                    'fantasy_points': 0.0,  # **DNP gets 0 fantasy points**
                    'date': actual_match_date,
                    'last10_fantasy_scores': json.dumps(past_stats['last_n_fantasy']),
                    'avg_fantasy_points_last5': past_stats['avg_fantasy_5'],
                    'avg_balls_faced_last5': past_stats['avg_balls_5'],
                    'avg_overs_bowled_last5': past_stats['avg_overs_5'],
                    'ownership': None
                }
                
                players_data.append(dnp_record)
                
                # Add DNP record to temporary storage for historical calculations
                all_players_temp.append(dnp_record.copy())
        
        # Save persistent player cache after processing
        self.save_player_cache()
        
        # **PROCESSING SUMMARY**
        total_matches_attempted = len(readme_matches)
        valid_matches_processed = len(matches_data)
        invalid_matches_skipped = total_matches_attempted - valid_matches_processed
        
        print(f"\n" + "="*60)
        print(f"📊 PROCESSING SUMMARY:")
        print(f"  Total matches attempted: {total_matches_attempted}")
        print(f"  ✅ Valid matches processed: {valid_matches_processed}")
        print(f"  ❌ Invalid matches skipped: {invalid_matches_skipped}")
        if invalid_matches_skipped > 0:
            print(f"  📝 Invalid matches were logged above with reasons")
        print(f"  👥 Total player records created: {len(players_data)}")
        print("="*60)
        
        return matches_data, players_data
    
    def save_to_csv(self, matches_data: List[Dict], players_data: List[Dict], append: bool = False):
        """Save data to CSV files"""
        matches_path = os.path.join(OUTPUT_BASE, "matches.csv")
        players_path = os.path.join(OUTPUT_BASE, "players.csv")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        
        # Save matches data
        matches_df = pd.DataFrame(matches_data)
        if append and os.path.exists(matches_path):
            existing_matches = pd.read_csv(matches_path)
            matches_df = pd.concat([existing_matches, matches_df], ignore_index=True)
            # Remove duplicates based on cricsheet_match_id
            matches_df = matches_df.drop_duplicates(subset=['cricsheet_match_id'], keep='last')
        
        matches_df.to_csv(matches_path, index=False)
        print(f"Saved {len(matches_df)} matches to {matches_path}")
        
        # Save players data
        players_df = pd.DataFrame(players_data)
        if append and os.path.exists(players_path):
            existing_players = pd.read_csv(players_path)
            players_df = pd.concat([existing_players, players_df], ignore_index=True)
            # Remove duplicates based on cricsheet_match_id + name
            players_df = players_df.drop_duplicates(subset=['cricsheet_match_id', 'name'], keep='last')
        
        players_df.to_csv(players_path, index=False)
        print(f"Saved {len(players_df)} player records to {players_path}")

    def select_best_player_match(self, candidates: List[Dict], search_name: str, match_context: Dict = None) -> Dict:
        """Select the best player match from API candidates using various criteria"""
        if not candidates:
            return {}
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        print(f"    Selecting from {len(candidates)} candidates for {search_name}")
        
        # Extract search components
        search_lower = search_name.lower()
        search_parts = search_lower.split()
        search_initials = [part[0] for part in search_parts if part and part[0].isalpha()]
        search_surname = search_parts[-1] if search_parts else ''
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = 0
            candidate_name = candidate.get('name', '').lower()
            candidate_parts = candidate_name.split()
            candidate_surname = candidate_parts[-1] if candidate_parts else ''
            
            # Exact match gets highest score
            if candidate_name == search_lower:
                score += 1000
            
            # Surname matching (most important for cricket)
            if search_surname and candidate_surname:
                if search_surname == candidate_surname:
                    score += 200
                elif search_surname in candidate_surname or candidate_surname in search_surname:
                    score += 100
            
            # Initial matching for abbreviated names like "J Clark"
            if len(search_parts) >= 2 and len(search_initials) >= 1:
                candidate_initials = [part[0] for part in candidate_parts if part and part[0].isalpha()]
                if search_initials[0] == candidate_initials[0] if candidate_initials else False:
                    score += 150
                    
                    # Special bonus for common cricket name patterns
                    # Joe/Joseph often abbreviated as J, Josh also as J
                    if search_initials[0] == 'j' and len(candidate_parts) >= 1:
                        first_name_lower = candidate_parts[0]
                        if first_name_lower in ['joe', 'joseph', 'josh', 'joshua', 'james', 'john', 'jonathan']:
                            score += 100  # Strong bonus for common J names
                        elif first_name_lower in ['jack', 'jacob', 'jason', 'jamie', 'jake']:
                            score += 50   # Medium bonus for other J names
            
            # Full name part matching
            for search_part in search_parts:
                if len(search_part) > 1:  # Ignore single characters
                    for candidate_part in candidate_parts:
                        if search_part == candidate_part:
                            score += 80
                        elif search_part in candidate_part or candidate_part in search_part:
                            score += 40
            
            # Check alternative names if available
            alt_names = candidate.get('altNames', '').lower()
            if alt_names:
                alt_parts = alt_names.replace(',', ' ').split()
                for search_part in search_parts:
                    if len(search_part) > 1:
                        for alt_part in alt_parts:
                            if search_part == alt_part:
                                score += 60
                            elif search_part in alt_part:
                                score += 30
            
            # Country/league context scoring
            country = candidate.get('country', '').lower()
            if match_context:
                league = match_context.get('league', '').lower()
                
                # English domestic cricket preferences
                if 't20 blast' in league or 'vitality blast' in league:
                    if country == 'england':
                        score += 50
                    elif country in ['australia', 'new zealand', 'south africa']:  # Common overseas players
                        score += 20
                elif 'ipl' in league:
                    if country in ['india', 'australia', 'england', 'south africa', 'west indies']:
                        score += 30
                elif 'bbl' in league:
                    if country == 'australia':
                        score += 50
                    elif country in ['england', 'new zealand', 'south africa']:
                        score += 20
            
            # Gender context (avoid obvious mismatches)
            # Common female first names to avoid for male matches
            female_names = ['dawn', 'karen', 'jennifer', 'laura', 'belinda', 'maggie', 'nell', 'lauren']
            male_names = ['joe', 'joseph', 'josh', 'joshua', 'max', 'maxwell', 'tom', 'thomas', 'ben', 'benjamin']
            
            first_name = candidate_parts[0] if candidate_parts else ''
            if first_name in female_names:
                score -= 100  # Penalty for likely female names in male context
            elif first_name in male_names:
                score += 20   # Bonus for likely male names
            
            # Special handling for common cricket name patterns
            if search_name.count(' ') == 1:  # "J Clark" pattern
                search_first, search_last = search_parts
                if len(search_first) == 1 and len(candidate_parts) >= 2:
                    # Check if first letter matches
                    if candidate_parts[0].startswith(search_first.lower()):
                        score += 100
            
            scored_candidates.append((score, candidate))
            print(f"      {candidate_name} ({country}): score {score}")
        
        # Return the highest scoring candidate
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidate = scored_candidates[0][1]
        print(f"    Selected: {best_candidate.get('name')} (score: {scored_candidates[0][0]})")
        
        return best_candidate
    
    def extract_surname(self, full_name: str) -> str:
        """Extract surname from full name"""
        parts = full_name.strip().split()
        if len(parts) > 1:
            return parts[-1]  # Last part is usually surname
        return ''
    
    def generate_name_variations(self, player_name: str) -> List[str]:
        """Generate common name variations for search"""
        variations = []
        
        # Remove middle initials and try variations
        parts = player_name.split()
        if len(parts) >= 2:
            # First name + Last name (skip middle initials)
            first_name = parts[0]
            last_name = parts[-1]
            variations.append(f"{first_name} {last_name}")
            
            # Try with just initials
            if len(first_name) > 1:
                variations.append(f"{first_name[0]} {last_name}")
            
            # Try full first name variations
            if '.' in first_name:
                # If first name has dots, try without them
                clean_first = first_name.replace('.', '')
                if clean_first:
                    variations.append(f"{clean_first} {last_name}")
        
        return variations

    def get_series_info_from_api(self, league: str, year: int) -> Dict:
        """Get series information from Cricketdata.org API with enhanced search logic"""
        series_key = f"{league}_{year}"
        
        if series_key in self.series_cache:
            return self.series_cache[series_key]
        
        try:
            print(f"  Fetching series info for {league} {year}")
            
            # Define search terms based on league
            search_terms = [league]  # Primary search term
            
            # Add fallback search terms for problematic leagues
            if league.upper() == "WBBL":
                search_terms = ["Big Bash League", "WBBL", "Womens Big Bash"]
            elif league.upper() == "WPL":
                search_terms = ["Premier League", "WPL", "Womens Premier"]
            elif "international league t20" in league.lower():
                search_terms = ["International League T20", "ILT20", league]
            
            # Try each search term
            all_candidates = []
            for search_term in search_terms:
                print(f"    Trying search term: '{search_term}'")
                search_result = self.api_request("series", {"search": search_term})
                
                if search_result.get('data'):
                    all_candidates.extend(search_result['data'])
            
            if all_candidates:
                # Find series matching the year - categorize and prioritize
                men_series = []
                women_series = []
                cross_year_series = []  # For 2024-25 type series
                
                for series in all_candidates:
                    series_name = series.get('name', '').lower()
                    series_year = series.get('startDate', '')
                    
                    if series_year:
                        try:
                            series_year_int = int(series_year[:4])
                            
                            # Normalize names for comparison
                            league_normalized = league.lower().replace("'", "").replace("-", " ").replace("_", " ")
                            series_normalized = series_name.replace("'", "").replace("-", " ").replace("_", " ")
                            
                            # Check if this series matches our target league
                            is_match = False
                            
                            # Direct name matching
                            if league_normalized in series_normalized:
                                is_match = True
                            
                            # Special cases for WBBL and WPL
                            elif league.upper() == "WBBL" and "women" in series_name and "big bash" in series_name:
                                is_match = True
                            elif league.upper() == "WPL" and "women" in series_name and "premier" in series_name:
                                is_match = True
                            
                            if is_match:
                                # Year matching logic
                                if series_year_int == year:
                                    # Direct year match
                                    if 'women' in series_name or 'female' in series_name:
                                        women_series.append(series)
                                        print(f"    Found women's series: {series.get('name', '')} (ID: {series.get('id', '')})")
                                    else:
                                        men_series.append(series)
                                        print(f"    Found men's series: {series.get('name', '')} (ID: {series.get('id', '')})")
                                elif year == 2025 and series_year_int == 2024 and ("2024-25" in series.get('name', '') or "2024/25" in series.get('name', '')):
                                    # Handle cross-year series (2024-25 for 2025 searches)
                                    cross_year_series.append(series)
                                    print(f"    Found cross-year series for 2025: {series.get('name', '')} (ID: {series.get('id', '')})")
                        except ValueError:
                            continue
                
                # Prioritization logic:
                # 1. Men's series for direct year match
                # 2. Women's series for direct year match  
                # 3. Cross-year series (for 2025 searches)
                target_series = None
                
                if men_series:
                    target_series = men_series[0]
                    series_type = "men's"
                elif women_series:
                    target_series = women_series[0] 
                    series_type = "women's"
                elif cross_year_series:
                    target_series = cross_year_series[0]
                    series_type = "cross-year"
                
                if target_series:
                    series_info = {
                        'series_id': target_series.get('id', ''),
                        'series_name': target_series.get('name', ''),
                        'start_date': target_series.get('startDate', ''),
                        'end_date': target_series.get('endDate', '')
                    }
                    print(f"    Selected {series_type} series: {series_info['series_name']} (ID: {series_info['series_id']})")
                    self.series_cache[series_key] = series_info
                    return series_info
            
            print(f"    No series found for {league} {year}")
            
        except Exception as e:
            print(f"    Error fetching series info: {e}")
        
        # Return empty info if not found
        empty_info = {'series_id': '', 'series_name': '', 'start_date': '', 'end_date': ''}
        self.series_cache[series_key] = empty_info
        return empty_info
    
    def get_match_info_from_api(self, series_id: str, team1: str, team2: str, match_date: str) -> Dict:
        """Get match information from Cricketdata.org API with robust timeout handling"""
        if not series_id:
            print(f"    No series_id provided for match lookup")
            return {'match_id': ''}
        
        try:
            print(f"    Fetching match info for {team1} vs {team2} on {match_date} (series: {series_id})")
            
            # Use series_info endpoint since we know it works
            print(f"    Using series_info endpoint...")
            
            # Try with shorter timeout first, then longer if needed
            timeouts = [8, 15, 25]  # Progressive timeout strategy
            
            for timeout in timeouts:
                try:
                    print(f"    Attempting API call with {timeout}s timeout...")
                    
                    url = f"{API_BASE}/series_info"
                    params = {"apikey": self.api_key, "id": series_id}
                    
                    response = self.session.get(url, params=params, timeout=timeout)
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"    Got response from API")
                        
                        if result and result.get('data'):
                            matches_data = result['data']
                            
                            # Extract match list (API uses 'matchList' not 'matches')
                            if 'matchList' in matches_data:
                                matches_list = matches_data['matchList']
                                print(f"    Found {len(matches_list)} matches in series")
                                
                                # Look for our specific match
                                for match in matches_list:
                                    match_date_api = match.get('date', '')
                                    teams_api = match.get('teams', [])
                                    
                                    # Try to match by date first (most reliable)
                                    if match_date_api and match_date_api.startswith(match_date):
                                        print(f"    Found match on target date: {match_date_api}")
                                        
                                        if len(teams_api) >= 2:
                                            # Enhanced team matching using new method
                                            team_match = self.enhanced_team_match(team1, team2, teams_api[0], teams_api[1])
                                            
                                            if team_match:
                                                match_info = {
                                                    'match_id': match.get('id', ''),
                                                    'match_name': match.get('name', ''),
                                                    'date': match.get('date', ''),
                                                    'teams': teams_api
                                                }
                                                print(f"    FOUND MATCH: {match_info['match_name']} (ID: {match_info['match_id']})")
                                                return match_info
                                            else:
                                                print(f"    Date matches but teams don't: {teams_api} vs [{team1}, {team2}]")
                                
                                print(f"    No matching match found for {team1} vs {team2} on {match_date}")
                                
                                # Fallback: try team-only matching if exact date failed
                                print(f"    Trying fallback team-only matching...")
                                for match in matches_list:
                                    teams_api = match.get('teams', [])
                                    if len(teams_api) >= 2:
                                        team_match = self.enhanced_team_match(team1, team2, teams_api[0], teams_api[1])
                                        if team_match:
                                            match_info = {
                                                'match_id': match.get('id', ''),
                                                'match_name': match.get('name', ''),
                                                'date': match.get('date', ''),
                                                'teams': teams_api
                                            }
                                            print(f"    FOUND MATCH (fallback): {match_info['match_name']} (ID: {match_info['match_id']})")
                                            return match_info
                            else:
                                print(f"    No matchList found in response. Available keys: {list(matches_data.keys())}")
                        else:
                            print(f"    No data in API response")
                        
                        return {'match_id': ''}  # Found response but no match
                    
                    elif response.status_code == 429:
                        print(f"    Rate limited, waiting 3 seconds...")
                        time.sleep(3)
                        continue
                    else:
                        print(f"    HTTP {response.status_code}: {response.text[:200]}")
                        return {'match_id': ''}
                    
                except requests.exceptions.Timeout:
                    print(f"    Timeout after {timeout}s - trying longer timeout...")
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"    Request error: {e}")
                    return {'match_id': ''}
            
            # If all timeouts failed
            print(f"    All API attempts timed out for series {series_id}")
            print(f"    This is likely due to large series size or API performance issues")
            
        except Exception as e:
            print(f"    Unexpected error fetching match info: {e}")
            import traceback
            traceback.print_exc()
        
        return {'match_id': ''}
    
    def get_match_details_by_id(self, match_id: str) -> Dict:
        """Get detailed match information by API match ID"""
        if not match_id:
            return {}
        
        try:
            print(f"    Fetching match details for match ID: {match_id}")
            
            # Try with shorter timeout first, then longer if needed
            timeouts = [8, 15, 25]
            
            for timeout in timeouts:
                try:
                    print(f"    Attempting match details API call with {timeout}s timeout...")
                    
                    url = f"{API_BASE}/match_info"
                    params = {"apikey": self.api_key, "id": match_id}
                    
                    response = self.session.get(url, params=params, timeout=timeout)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result and result.get('data'):
                            match_data = result['data']
                            print(f"    Successfully fetched match details")
                            
                            # Extract relevant match information
                            match_info = {
                                'match_id': match_id,
                                'date': match_data.get('date', ''),
                                'venue': match_data.get('venue', ''),
                                'city': match_data.get('city', ''),
                                'country': match_data.get('country', ''),
                                'status': match_data.get('status', ''),
                                'result': match_data.get('result', ''),
                                'team1': match_data.get('team1', ''),
                                'team2': match_data.get('team2', ''),
                                'toss_winner': match_data.get('toss_winner', ''),
                                'toss_decision': match_data.get('toss_decision', ''),
                                'match_winner': match_data.get('match_winner', ''),
                                'series_id': match_data.get('series_id', ''),
                                'series_name': match_data.get('series_name', ''),
                                'format': match_data.get('format', '')
                            }
                            
                            return match_info
                        else:
                            print(f"    No match data found for ID {match_id}")
                            return {}
                    
                    elif response.status_code == 429:
                        print(f"    Rate limited, waiting...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"    API returned status {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"    Timeout with {timeout}s limit")
                    if timeout == timeouts[-1]:  # Last timeout attempt
                        print(f"    All timeouts failed for match ID {match_id}")
                        return {}
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"    Request error: {e}")
                    return {}
            
            return {}
            
        except Exception as e:
            print(f"    Error fetching match details for ID {match_id}: {e}")
            return {}

    def is_match_abandoned(self, cricsheet_data: Dict) -> bool:
        """Check if a match was abandoned/incomplete"""
        outcome = cricsheet_data.get('info', {}).get('outcome', {})
        result = outcome.get('result', '')
        
        # Check for common abandoned match indicators
        if result == 'no result':
            return True
        
        # Additional checks for abandoned matches
        # Check if there are very few deliveries (indicating rain/abandonment)
        innings = cricsheet_data.get('innings', [])
        total_deliveries = 0
        
        for innings_data in innings:
            for over_data in innings_data.get('overs', []):
                total_deliveries += len(over_data.get('deliveries', []))
        
        # If there are very few deliveries (less than 12 balls = 2 overs), likely abandoned
        if total_deliveries < 12:
            return True
        
        return False

    def get_historical_bowling_phases(self, player_name: str, current_match_date: str, all_players_data: List[Dict]) -> List[float]:
        """Get bowling phase proportions from last 5 matches (pre-match feature)"""
        
        # Get player's past bowling matches before current date
        past_matches = []
        for player_record in all_players_data:
            if (player_record['name'] == player_name and 
                player_record['date'] < current_match_date and 
                player_record.get('overs_bowled', 0) > 0):  # Only matches where they bowled
                past_matches.append(player_record)
        
        # Sort by date (oldest first) and take last 5
        past_matches.sort(key=lambda x: x['date'])
        recent_matches = past_matches[-5:] if len(past_matches) >= 5 else past_matches
        
        if not recent_matches:
            # Non-bowler or no bowling history
            print(f"    No bowling history for {player_name} - returning [0.00, 0.00, 0.00]")
            return [0.00, 0.00, 0.00]
        
        # Sum actual overs from detailed phase data
        total_powerplay_overs = 0.0
        total_middle_overs = 0.0
        total_death_overs = 0.0
        
        for match in recent_matches:
            # Parse the stored detailed phase data
            phase_detail_str = match.get('bowling_phases_detail', '{}')
            try:
                # Try to parse as JSON first (new format)
                if phase_detail_str.startswith('{'):
                    import json
                    phase_detail = json.loads(phase_detail_str)
                    total_powerplay_overs += phase_detail.get('powerplay', 0.0)
                    total_middle_overs += phase_detail.get('middle', 0.0)
                    total_death_overs += phase_detail.get('death', 0.0)
                else:
                    # Fallback: if no detailed data available, skip this match
                    print(f"    Skipping match with no detailed phase data: {match.get('date', 'unknown')}")
                    continue
                    
            except (json.JSONDecodeError, TypeError, AttributeError):
                # If parsing fails, skip this match
                print(f"    Skipping match with invalid phase data: {match.get('date', 'unknown')}")
                continue
        
        # Calculate proportions
        total_overs = total_powerplay_overs + total_middle_overs + total_death_overs
        
        if total_overs == 0:
            print(f"    No valid bowling data found for {player_name} - returning [0.00, 0.00, 0.00]")
            return [0.00, 0.00, 0.00]
        
        powerplay_prop = round(total_powerplay_overs / total_overs, 2)
        middle_prop = round(total_middle_overs / total_overs, 2)
        death_prop = round(total_death_overs / total_overs, 2)
        
        # Ensure sum = 1.0 (handle rounding errors)
        total_prop = powerplay_prop + middle_prop + death_prop
        if total_prop > 0 and abs(total_prop - 1.0) > 0.01:
            death_prop = round(1.0 - powerplay_prop - middle_prop, 2)
        
        print(f"    {player_name} bowling phases from {len(recent_matches)} matches: [{powerplay_prop}, {middle_prop}, {death_prop}]")
        return [powerplay_prop, middle_prop, death_prop]

    def get_player_past_stats_by_date(self, player_name: str, all_players_data: List[Dict], current_match_date: str, current_series_id: str, N: int = 10) -> Dict:
        """Get player's past statistics from previous matches in SAME SERIES + YEAR only
        
        CRITICAL FEATURES:
        1. NO DATA LEAKAGE - only includes matches BEFORE current match date
        2. SERIES INDEPENDENCE - only matches from same series_id (same series + same year)
        3. PROGRESSIVE BUILDING - uses existing CSV + current run data
        """
        try:
            # Filter for same player's past matches before current date
            # Handle both YYYY-MM-DD (from API) and DD-MM-YYYY (from CSV) formats
            try:
                current_date = datetime.strptime(current_match_date, '%Y-%m-%d')
            except ValueError:
                current_date = datetime.strptime(current_match_date, '%d-%m-%Y')
            
            # **FIX: COMBINE existing historical data + current run data**
            # BUT exclude current match from all_players_data to prevent data leakage
            all_historical_data = self.existing_player_data + all_players_data
            
            past_matches = []
            for player_record in all_historical_data:
                if (player_record['name'] == player_name):
                    # **SERIES INDEPENDENCE: Only same series + year**
                    match_series_id = str(player_record.get('api_series_id', ''))
                    if match_series_id != current_series_id:
                        continue  # Skip matches from different series/years
                    
                    try:
                        # Try both date formats for robustness
                        player_date_str = player_record['date']
                        try:
                            match_date = datetime.strptime(player_date_str, '%Y-%m-%d')
                        except ValueError:
                            match_date = datetime.strptime(player_date_str, '%d-%m-%Y')
                        
                        # **CRITICAL: Only include matches BEFORE current date to prevent data leakage**
                        if match_date < current_date:
                            # **ADDITIONAL CHECK: Skip invalid matches (no API match ID)**
                            api_match_id = player_record.get('api_match_id', '')
                            if not api_match_id or api_match_id.strip() == '':
                                print(f"    Skipping invalid match (no API match ID) for {player_name} on {player_date_str}")
                                continue
                                
                            past_matches.append(player_record)
                    except ValueError:
                        # Skip records with invalid dates
                        print(f"    Warning: Could not parse date '{player_record.get('date', 'N/A')}' for {player_name}")
                        continue
            
            # Sort by date ascending (oldest first)
            def parse_date_flexible(date_str):
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    return datetime.strptime(date_str, '%d-%m-%Y')
            
            past_matches.sort(key=lambda x: parse_date_flexible(x['date']))
            
            print(f"    Found {len(past_matches)} VALID past matches for {player_name} in SAME SERIES before {current_match_date}")
            
            # Extract statistics from past matches with DNP handling
            fantasy_scores = []
            balls_faced_list = []
            overs_bowled_list = []
            
            for match in past_matches:
                # **DNP LOGIC**: 
                # - If fantasy_points = 0, assume DNP (not in squad) → record as 0
                # - If fantasy_points = 4, player played but got minimal points → record as 4
                fantasy_pts = float(match.get('fantasy_points', 0))
                fantasy_scores.append(fantasy_pts)
                
                balls_faced_list.append(int(match.get('balls_faced', 0)))
                overs_bowled_list.append(float(match.get('overs_bowled', 0)))
            
            # Get last N entries (most recent last)
            last_n_fantasy = fantasy_scores[-N:] if len(fantasy_scores) > 0 else []
            
            # Calculate averages for last 5
            last_5_size = min(5, len(fantasy_scores))
            avg_fantasy_5 = sum(fantasy_scores[-last_5_size:]) / last_5_size if last_5_size > 0 else 0.0
            avg_balls_5 = sum(balls_faced_list[-last_5_size:]) / last_5_size if last_5_size > 0 else 0.0
            avg_overs_5 = sum(overs_bowled_list[-last_5_size:]) / last_5_size if last_5_size > 0 else 0.0
            
            if len(past_matches) > 0:
                print(f"    ✅ Past fantasy scores: {fantasy_scores[-5:]} -> avg: {avg_fantasy_5}")
                print(f"    ✅ Past balls faced: {balls_faced_list[-5:]} -> avg: {avg_balls_5}")
                print(f"    ✅ Past overs bowled: {overs_bowled_list[-5:]} -> avg: {avg_overs_5}")
            
            return {
                'last_n_fantasy': last_n_fantasy,
                'avg_fantasy_5': round(avg_fantasy_5, 1),
                'avg_balls_5': round(avg_balls_5, 1),
                'avg_overs_5': round(avg_overs_5, 2)
            }
            
        except Exception as e:
            print(f"    Error calculating past stats for {player_name}: {e}")
            return {
                'last_n_fantasy': [],
                'avg_fantasy_5': 0.0,
                'avg_balls_5': 0.0,
                'avg_overs_5': 0.0
            }
    
    def clear_cache(self):
        """Clear all caches for a fresh run"""
        self.player_cache.clear()
        self.player_id_cache.clear()
        self.series_cache.clear()
        print("All caches cleared")
    
    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for better matching between Cricsheet and API"""
        if not team_name:
            return ''
        
        # Convert to lowercase and remove common variations
        normalized = team_name.lower().strip()
        
        # Common team name mappings for better matching
        team_mappings = {
            'st kitts and nevis patriots': 'st kitts patriots',
            'antigua and barbuda falcons': 'antigua falcons',
            'guyana amazon warriors': 'guyana warriors',
            'jamaica tallawahs': 'jamaica',
            'barbados royals': 'barbados',
            'st lucia kings': 'st lucia',
            'trinbago knight riders': 'trinbago',
            'trinbago knight riders': 'knight riders'
        }
        
        # Check for direct mapping
        if normalized in team_mappings:
            return team_mappings[normalized]
        
        # Remove common words that might cause mismatches
        words_to_remove = ['cricket', 'club', 'fc', 'cc', 'county']
        words = normalized.split()
        filtered_words = [word for word in words if word not in words_to_remove]
        
        return ' '.join(filtered_words)
    
    def enhanced_team_match(self, cricsheet_team1: str, cricsheet_team2: str, api_team1: str, api_team2: str) -> bool:
        """Enhanced team matching with normalization and fuzzy matching"""
        # Normalize all team names
        cs_team1 = self.normalize_team_name(cricsheet_team1)
        cs_team2 = self.normalize_team_name(cricsheet_team2)
        api_t1 = self.normalize_team_name(api_team1)
        api_t2 = self.normalize_team_name(api_team2)
        
        # Direct exact match (both orders)
        if (cs_team1 == api_t1 and cs_team2 == api_t2) or (cs_team1 == api_t2 and cs_team2 == api_t1):
            return True
        
        # Substring matching (both orders)
        if ((cs_team1 in api_t1 and cs_team2 in api_t2) or 
            (cs_team1 in api_t2 and cs_team2 in api_t1) or
            (api_t1 in cs_team1 and api_t2 in cs_team2) or
            (api_t2 in cs_team1 and api_t1 in cs_team2)):
            return True
        
        # Check key words matching (for cases like "Guyana Amazon Warriors" vs "Guyana Warriors")
        cs_words1 = set(cs_team1.split())
        cs_words2 = set(cs_team2.split())
        api_words1 = set(api_t1.split())
        api_words2 = set(api_t2.split())
        
        # Check if at least 2 words match between team pairs
        match1 = len(cs_words1.intersection(api_words1)) >= 1 and len(cs_words2.intersection(api_words2)) >= 1
        match2 = len(cs_words1.intersection(api_words2)) >= 1 and len(cs_words2.intersection(api_words1)) >= 1
        
        return match1 or match2

def get_available_series():
    """Get list of all available cricket series folders"""
    if not os.path.exists(CRICSHEET_BASE):
        print(f"❌ Cricsheet directory not found: {CRICSHEET_BASE}")
        return []
    
    series_folders = []
    for item in os.listdir(CRICSHEET_BASE):
        item_path = os.path.join(CRICSHEET_BASE, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            series_folders.append(item)
    
    return sorted(series_folders)

def process_all_series_for_year(etl, year, limit=None):
    """Process all available series for a specific year"""
    print(f"🚀 PROCESSING ALL CRICKET SERIES FOR {year}")
    print("=" * 60)
    
    available_series = get_available_series()
    if not available_series:
        print("❌ No cricket series found")
        return False
    
    print(f"📋 Found {len(available_series)} cricket series:")
    for i, series in enumerate(available_series, 1):
        print(f"  {i:2d}. {series}")
    
    successful = 0
    failed = 0
    total_matches = 0
    total_players = 0
    
    print(f"\n🎯 Processing all series for {year}...")
    print("   (Using append mode to combine all data)")
    
    for i, series_name in enumerate(available_series, 1):
        print(f"\n[{i}/{len(available_series)}] Processing: {series_name}")
        print("-" * 40)
        
        series_folder = os.path.join(CRICSHEET_BASE, series_name)
        if not os.path.exists(series_folder):
            print(f"   ⚠️ SKIP: Directory not found")
            continue
        
        try:
            matches_data, players_data = etl.process_series(series_folder, limit, year)
            
            if matches_data:
                # Always use append mode to preserve existing data
                etl.save_to_csv(matches_data, players_data, append=True)
                
                print(f"✅ SUCCESS: {series_name}")
                print(f"   Added {len(matches_data)} matches, {len(players_data)} player records")
                
                successful += 1
                total_matches += len(matches_data)
                total_players += len(players_data)
            else:
                print(f"⚠️ NO DATA: {series_name} (no {year} matches found)")
                
        except Exception as e:
            print(f"❌ FAILED: {series_name}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 FINAL SUMMARY FOR {year}")
    print(f"{'='*60}")
    print(f"✅ Successful series: {successful}")
    print(f"❌ Failed series: {failed}")
    print(f"📈 Total matches processed: {total_matches}")
    print(f"👥 Total player records: {total_players}")
    
    if successful > 0:
        print(f"\n🎉 SUCCESS! Comprehensive {year} cricket dataset created")
        print("   Check matches.csv and players.csv for complete data")
        
        # Quick historical data check
        try:
            import pandas as pd
            players_df = pd.read_csv(os.path.join(OUTPUT_BASE, "players.csv"))
            year_data = players_df[players_df['date'].str.startswith(str(year))]
            historical_records = len(year_data[year_data['last10_fantasy_scores'] != '[]'])
            
            print(f"\n📊 {year} DATASET QUALITY:")
            print(f"   Records with historical data: {historical_records}/{len(year_data)} ({historical_records/len(year_data)*100:.1f}%)")
            
        except Exception as e:
            print(f"   (Could not analyze data quality: {e})")
    
    return successful > 0

def main():
    parser = argparse.ArgumentParser(description='Cricket ETL Pipeline')
    parser.add_argument('--series', type=str, default=None, 
                       help='Series folder name (e.g., "T20 Blast"). If not provided, processes all series.')
    parser.add_argument('--year', type=int, default=None,
                       help='Filter matches by specific year (e.g., 2024). Required when --series is not provided.')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing CSV files (only used with --series)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of matches to process per series (for testing)')
    
    args = parser.parse_args()
    
    # Initialize ETL pipeline
    etl = CricketETL()
    
    # **NEW: Process all series if --series not provided**
    if not args.series:
        if not args.year:
            print("❌ ERROR: --year is required when processing all series")
            print("Usage: python cricket_etl_pipeline.py --year 2024")
            return
        
        # Process all series for the specified year
        success = process_all_series_for_year(etl, args.year, args.limit)
        return
    
    # **EXISTING: Process single series**
    series_folder = os.path.join(CRICSHEET_BASE, args.series)
    
    if not os.path.exists(series_folder):
        print(f"Series folder not found: {series_folder}")
        return
    
    try:
        matches_data, players_data = etl.process_series(series_folder, args.limit, args.year)
        
        if matches_data:
            etl.save_to_csv(matches_data, players_data, append=args.append)
            print(f"Successfully processed {len(matches_data)} matches")
            if args.year:
                print(f"All matches were from year {args.year}")
        else:
            print("No matches processed")
            if args.year:
                print(f"No matches found for year {args.year}")
            
    except Exception as e:
        print(f"Error processing series: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
