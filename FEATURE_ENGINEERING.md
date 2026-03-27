# Feature Engineering Documentation

## Overview

~140 features are engineered per match and stored as individual Parquet files. Features evolved across three training rounds (R1 → R2 → R3), with R3 being the most complete and best-performing set.

---

## Feature Extraction Pipeline

```
Match CSV (from ETL)
      │
      ▼
features.py  ──────────────────────────►  Base player stats
      │
      ▼
enhanced_features.py  ─────────────────►  Rolling form features
      │
      ▼
squad_context_feature_extractor.py ───►  Squad/lineup context
      │
      ▼
enhanced_opportunity_feature_extractor.py  ►  Opportunity estimates
      │
      ▼
venue_interaction_feature_extractor.py  ──►  Player×Venue signals
      │
      ▼
R3_comprehensive_feature_extractor_fixed_v2.py  ►  Final merge → Parquet
```

---

## Feature Categories

### 1. Player Form — Rolling Window Features

Computed over last N matches (5 and 10 match windows):

| Feature | Description |
|---------|-------------|
| `fantasy_score_mean_5` | Mean fantasy score over last 5 matches |
| `fantasy_score_std_5` | Standard deviation of fantasy score (consistency) |
| `fantasy_score_max_5` | Best score in last 5 matches (ceiling potential) |
| `batting_avg_10` | Rolling batting average over last 10 |
| `strike_rate_10` | Rolling batting strike rate |
| `economy_rate_5` | Bowling economy over last 5 |
| `wickets_mean_5` | Mean wickets taken per match |
| `dots_pct_5` | % of balls that were dot balls bowled |

---

### 2. Squad Context Features

These capture a player's *role and opportunity* within their team:

| Feature | Description |
|---------|-------------|
| `batting_position_prob_1` | Probability player bats at position 1 (opener) |
| `batting_position_prob_3` | Probability player bats at #3 |
| `batting_position_modal` | Most common batting position in recent matches |
| `bowling_load_share` | % of team's overs typically bowled by this player |
| `expected_balls_faced` | Expected balls faced based on position × match type |
| `is_regular_bowler` | Binary — does this player bowl regularly? |
| `is_finisher` | Binary — tends to bat lower order in chase situations |

---

### 3. Venue Features

Ground-level historical statistics:

| Feature | Description |
|---------|-------------|
| `venue_avg_first_innings` | Historical first innings average at ground |
| `venue_run_rate` | Average run rate at venue |
| `venue_pitch_type` | Categorical: batting/bowling/neutral/spin |
| `venue_avg_wickets` | Average wickets fallen per innings |
| `venue_boundary_pct` | % of runs scored in boundaries at this ground |
| `venue_matches_played` | Number of T20s played at venue (data quality signal) |

---

### 4. Player × Venue Interaction Features

How *this specific player* performs at *this specific ground*:

| Feature | Description |
|---------|-------------|
| `player_venue_fantasy_avg` | Player's mean fantasy score at this ground |
| `player_venue_sr_ratio` | Strike rate at venue / career strike rate |
| `player_venue_economy_ratio` | Economy at venue / career economy |
| `player_venue_matches` | Number of matches played at this venue |
| `player_venue_best_score` | Player's highest fantasy score at this ground |

> Note: Sparse for players with few appearances at a given ground. Missing values handled by XGBoost natively.

---

### 5. Opportunity Features

Estimated batting/bowling opportunity for the upcoming match:

| Feature | Description |
|---------|-------------|
| `batting_opportunity_score` | Composite: position × form × opponent bowling strength |
| `bowling_opportunity_score` | Composite: conditions × form × opponent batting strength |
| `expected_overs_bowled` | Expected overs this bowler will bowl |
| `captain_ev_multiplier` | Expected fantasy EV if selected as captain (2× multiplier) |

---

### 6. Opponent-Adjusted Features

| Feature | Description |
|---------|-------------|
| `opponent_bowling_strength` | Mean economy + wicket rate of opponent bowling attack |
| `opponent_batting_strength` | Mean fantasy scores of opponent batters in recent form |
| `home_away_flag` | 1 = home team, 0 = away |

---

## Why Parquet?

Features are stored as **one Parquet file per match** rather than one large CSV because:

1. **Read speed** — Parquet columnar format is 5-10× faster to read than CSV for column-subset queries
2. **Compression** — ~6MB per match in Parquet vs much larger CSV equivalent
3. **Schema** — Data types enforced, no silent string/float casting issues
4. **Incremental processing** — New matches can be added without reprocessing the entire dataset

Files follow the naming convention: `match_{match_id}_features.parquet`

---

## Feature Evolution Across Training Rounds

| Round | Features | Notes |
|-------|----------|-------|
| R1 | ~40 | Basic form + position features |
| R2 | ~80 | Added venue data, better rolling windows |
| R3 | ~140 | Full feature set, player×venue interactions, opportunity scores |

R3 is the production feature set used in the final CPL model.

---

## Feature Importance (Top Features)

Based on XGBoost SHAP values from the CPL R3 model:

1. `fantasy_score_mean_5` — Recent form is by far the strongest predictor
2. `batting_opportunity_score` — Will the player actually get to bat/bowl?
3. `player_venue_fantasy_avg` — How has this player done here before?
4. `bowling_load_share` — Regular bowlers are more predictable
5. `venue_avg_first_innings` — Batting-friendly grounds boost batting scores
6. `batting_position_modal` — Openers have more reliable opportunity
7. `fantasy_score_std_5` — Consistency matters; volatile players are risky

See `feature_importance_analysis.py` and `feature_importance_explained.py` for full rankings and interpretations.
