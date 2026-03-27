# System Architecture

## Overview

This system is built as a sequential ML pipeline with five distinct stages. Each stage produces artefacts consumed by the next.

```
[Stage 1] ETL          →  Raw CSVs / Match Records
[Stage 2] Team Gen     →  Simulated Teams + Elite Labels  
[Stage 3] Feature Eng  →  Per-match Parquet files (~140 features)
[Stage 4] Training     →  Trained XGBoost .joblib model
[Stage 5] Inference    →  MGAG — Model-Guided Assisted Generation
```

---

## Stage 1: ETL Pipeline

**Key files:** `cricket_etl_pipeline.py`, `run_enhanced_etl_2024.py`, `database_join/cricket_venue_context.py`

Data is sourced from **Cricsheet** — a public repository of ball-by-ball cricket data in YAML/JSON format. The ETL pipeline:

- Downloads match files by series/year
- Parses ball-by-ball delivery data into player-level aggregates
- Computes per-match fantasy scores using standard fantasy scoring rules (runs, wickets, catches, etc.)
- Outputs one CSV per match or aggregated CSVs by series

A **venue context patch** (`cricket_venue_context.py`) was added in a later iteration to enrich records with ground metadata (city, country, pitch type, historical averages).

---

## Stage 2: Fantasy Team Generation

**Key files:** `parquet_team_generator_ver2.1.py`, `generate_validation_teams.py`

Since there is no labelled dataset of "good" fantasy teams, they are generated synthetically:

1. For each historical match, player pools are loaded from squad data
2. N random teams are generated, each satisfying:
   - Exactly 11 players
   - Valid role distribution (batters, bowlers, all-rounders, wicket-keeper)
   - Budget constraints
   - Captain / Vice-Captain selected
3. Each team's fantasy score is computed
4. Score distribution is calculated per match
5. Teams in the **top 10th percentile** are labelled `elite=1`; rest are `elite=0`

This labelled dataset forms the training corpus.

**Validation teams** are held out separately to evaluate the model's ability to identify elite teams on unseen matches.

---

## Stage 3: Feature Engineering

**Key files:** `features.py`, `enhanced_features.py`, `feature_config.py`, `squad_context_feature_extractor.py`, `enhanced_opportunity_feature_extractor.py`, `venue_interaction_feature_extractor.py`

Features are computed per match and stored as individual **Parquet files** in `R3_Features_output/clean_140_features/`. Using Parquet (vs CSV) was a deliberate choice for:
- Compressed columnar storage (~5-7MB per match vs much larger CSVs)
- Fast read speeds during training
- Schema enforcement

### Feature Categories

**Player Form (rolling window)**
- Last 5/10 match fantasy scores (mean, std, max)
- Rolling batting average, strike rate, economy rate
- Recent wicket-taking form

**Squad Context**
- Batting position probability (likelihood of batting at each position)
- Bowling load share (% of overs bowled in recent matches)
- Expected balls faced based on historical position

**Venue Features**
- Ground average first innings score
- Pitch type classification (batting/bowling/neutral)
- Historical run rate at venue

**Player × Venue Interaction**
- Player's average fantasy score at this specific ground
- Batting SR at venue vs career SR (venue boost factor)
- Wickets per match at venue

**Opportunity Features**
- Expected batting opportunity score (position × form × opponent bowling)
- Expected bowling opportunity score (conditions × form)

**Team-level Context**
- Team batting depth index
- Opponent bowling strength
- Home/away indicator

---

## Stage 4: Model Training

**Key files:** `train_xgboost.py`, `R3_Training/R3_comprehensive_feature_extractor_fixed_v2.py`, `R3_Training/R3_elite_discovery_trainer.py`

### Algorithm
XGBoost Regressor — chosen for:
- Handles tabular sports data well
- Built-in feature importance
- Robust to missing values (common in cricket data — not all players bat/bowl in every match)
- Fast training on moderate dataset sizes

### Training Evolution

| Round | Notes |
|-------|-------|
| R1 | Baseline model, all leagues, simple features |
| R2 | Added venue features, improved ETL |
| R3 | CPL-focused, 140 features, best results |

The decision to focus on **CPL (Caribbean Premier League)** in R3 was deliberate: smaller player pool, consistent venues, and 6-week season format meant cleaner learning signal vs. IPL which has confounding home-ground and DLS factors.

### Model Persistence
Trained models are saved as `.joblib` files (e.g., `cpl_regressor_20250813_121421.joblib`) with a paired filter (`cpl_filter_20250813_121421.joblib`) that pre-screens players before the regressor scores them.

---

## Stage 5: MGAG — Model-Guided Assisted Generation

**Key files:** `MGAG/mgag_orchestrator.py`, `MGAG/mgag_live_orchestrator.py`, `MGAG/simsim.py`, `MGAG/team_validator.py`

MGAG is the inference layer. Rather than directly outputting a team, it uses the trained model to **guide** a team generation search:

1. Load upcoming match squad data
2. Score each player using the XGBoost model (predicted fantasy score)
3. Run constrained team generation (budget, roles, captain rules)
4. Use model scores as weights in the search — higher scored players are preferentially selected
5. Validate generated teams against fantasy platform rules
6. Output ranked team recommendations

The "Assisted Generation" framing reflects that the model doesn't directly produce a team — it assists a constrained selection algorithm to make better choices than random.

---

## Key Design Decisions

**Why Parquet for intermediate data?**
Match feature files are read thousands of times during hyperparameter tuning and cross-validation. Parquet's columnar compression and fast reads made training loops significantly faster than CSV.

**Why generate random teams rather than optimise directly?**
The elite label approach allows the model to learn *what patterns characterise good teams* rather than directly solving the combinatorial optimisation problem. This makes the inference more robust and generalisable across different team compositions.

**Why CPL specifically?**
IPL data is richer but noisier — home-ground effects, pitch curation, and India's extreme weather variation create confounding signals. CPL's smaller, more consistent player pool and Caribbean pitches gave cleaner training signal.

**Why XGBoost over neural nets?**
Dataset size (~200-300 CPL matches × N teams per match) is moderate, not massive. XGBoost handles this well and provides interpretable feature importances. Neural networks would likely overfit with this data volume.
