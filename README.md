# 🏏 Fantasy Cricket Score Predictor — XGBoost ML System

A end-to-end machine learning pipeline that predicts fantasy cricket scores for players, generates **fictional but statistically valid** fantasy teams from historical match data, and uses those teams to train and validate an XGBoost regression model.

Built entirely through **vibe coding** — iterating with AI assistance, no formal CS background. The system evolved from raw data ingestion all the way to a model-guided inference system for generating optimal fantasy teams.

---

## 🧠 What This System Does

```
Public Cricket Data (Cricsheet)
        │
        ▼
   ETL Pipeline  ──────────────────────────────────────────────────────────────►  Raw Match CSVs
   (cricket_etl_pipeline.py)                                                       + Venue Data
        │
        ▼
 Parquet Team Generator  ──────────────────────────────────────────────────►  Simulated Fantasy Teams
 (parquet_team_generator_ver2.1.py)                                             (per past match)
        │
        │   Teams with fantasy score in TOP 10% → "Elite"
        │   All others → "Non-Elite"
        │
        ▼
 Feature Engineering  ──────────────────────────────────────────────────────►  Feature Parquets
 (features.py / enhanced_features.py / squad_context_feature_extractor.py)       (~140 features per match)
        │
        ▼
 XGBoost Training  ─────────────────────────────────────────────────────────►  Trained Model (.joblib)
 (train_xgboost.py / R3_comprehensive_feature_extractor_fixed_v2.py)             CPL-tuned regressor
        │
        ▼
 Model Evaluation & Validation  ────────────────────────────────────────────►  Metrics, Feature Importance
 (validate_model_pipeline.py / feature_importance_analysis.py)
        │
        ▼
 Inference / Team Generation  ──────────────────────────────────────────────►  Recommended Fantasy XI
 (MGAG system / live_team_generator.py)
```

---

## 🗂️ Project Structure

```
app_root/
│
├── 📥 Data Ingestion & ETL
│   ├── cricket_etl_pipeline.py           # Main ETL — pulls Cricsheet ball-by-ball data
│   ├── run_enhanced_etl_2024.py          # Extended ETL with 2024 match data
│   ├── run_all_series_by_year.py         # Batch ETL runner across seasons
│   └── database_join/
│       ├── cricket_etl_pipeline.py       # Core pipeline (also lives here)
│       └── cricket_venue_context.py      # Venue data enrichment patch
│
├── 🧩 Feature Engineering
│   ├── features.py                       # Base feature set
│   ├── enhanced_features.py              # Extended/improved features
│   ├── feature_config.py                 # Feature flags and configuration
│   ├── squad_context.py                  # Squad-level context features
│   ├── squad_context_feature_extractor.py # Extracts squad context per match
│   ├── enhanced_opportunity_feature_extractor.py  # Batting/bowling opportunity features
│   └── venue_interaction_feature_extractor.py     # Player × venue interaction features
│
├── 🏟️ Fantasy Team Generation (Training Data)
│   ├── parquet_team_generator_ver2.1.py  # Generates simulated fantasy teams from parquets
│   ├── generate_validation_teams.py      # Generates held-out validation teams
│   └── live_team_generator.py            # Real-time team generation for inference
│
├── 🤖 Model Training
│   ├── train_xgboost.py                  # Core XGBoost regressor training
│   ├── train_xgboost_fixed.py            # Patched training (feature alignment fixes)
│   ├── train_all_models_comprehensive.py # Multi-league training sweep
│   ├── R3_Training/                      # Round 3 training — best CPL model
│   │   ├── R3_comprehensive_feature_extractor_fixed_v2.py
│   │   ├── R3_elite_discovery_trainer.py
│   │   ├── R3_exact_feature_trainer.py
│   │   └── contextual_cvc_feature_extractor.py
│   └── stage1_xgboost_scorer.py          # Stage 1 scoring pipeline
│
├── 📊 Model Evaluation
│   ├── feature_importance_analysis.py    # SHAP/feature importance plots
│   ├── feature_importance_explained.py   # Human-readable feature explanations
│   ├── validate_model_pipeline.py        # Full pipeline validation
│   ├── investigate_predictions.py        # Prediction deep-dive analysis
│   └── multi_league_elite_rate_calculator.py  # Elite team hit-rate by league
│
├── 🎯 Inference — MGAG (Model-Guided Assisted Generation)
│   ├── MGAG/
│   │   ├── mgag_orchestrator.py          # Main entry point — coordinates full inference pipeline
│   │   ├── mgag_live_orchestrator.py     # Live match version with real-time squad updates
│   │   ├── simsim.py                     # Monte Carlo simulation engine (core search)
│   │   ├── simsim_to_mgag_converter.py   # Converts simulation output to MGAG format
│   │   ├── team_validator.py             # Validates all fantasy platform rules
│   │   ├── real_model_interface.py       # Clean XGBoost ↔ MGAG interface layer
│   │   ├── smart_mgag.py                 # Enhanced version with smarter heuristics
│   │   ├── complete_mgag_pipeline.py     # End-to-end pipeline runner
│   │   ├── optimized_mgag_pipeline.py    # Performance-optimised inference
│   │   ├── match_data_loader.py          # Loads and validates squad data per match
│   │   └── find_cpl_matches.py           # CPL match lookup utility
│   └── enhanced_live_team_generator.py   # Enhanced team generator with model guidance
│
├── 📁 Data Outputs (NOT in repo — generated locally)
│   ├── R3_Features_output/clean_140_features/   # Per-match feature parquets (~6MB each)
│   ├── R3_Training/R3_models/                    # Trained model files (.joblib)
│   └── Live_Matches/                             # Live match data cache
│
└── 📝 Documentation
    ├── README.md                          # This file
    ├── docs/ARCHITECTURE.md               # System architecture deep-dive
    ├── docs/FEATURE_ENGINEERING.md        # All 140 features explained
    ├── docs/HOW_TO_RUN.md                 # Setup and run instructions
    └── docs/RESULTS.md                    # Model results and evaluation
```

---

## 🔬 How the Training Data Was Generated

One of the more creative parts of this project: **there's no labelled "good fantasy team" dataset to download.** So I generated one.

For every historical match in the dataset:
1. Random legal fantasy teams were generated (11 players, valid captain/vice-captain, within budget/role constraints)
2. Each team's fantasy score was computed using standard fantasy scoring rules
3. Teams in the **top 10% of score distribution** for that match were labelled **"Elite"**
4. All others were labelled **"Non-Elite"**

This gave a binary classification signal that could then be used to train a model to predict whether a given team configuration would score in the elite range.

Later iterations focused only on **specific leagues** (primarily CPL — Caribbean Premier League) because the model learned better on consistent pitch conditions, player pools, and venues.

---

## ⚙️ Feature Engineering

~140 features were engineered per match, stored as Parquet files (one per match). Key feature categories:

| Category | Examples |
|----------|---------|
| **Player Form** | Rolling avg runs/wickets, last 5 match scores, strike rate trends |
| **Squad Context** | Batting position probability, bowling load share |
| **Venue Effects** | Ground avg score, pitch type, historical run rates |
| **Player × Venue** | Individual player performance at specific ground |
| **Opportunity Features** | Expected balls faced, overs bowled based on team sheet |
| **Team Context** | Captain multiplier impact, team batting depth |

---

## 🏆 Model Performance

- **Algorithm:** XGBoost Regressor (with filter ensemble)
- **Best League:** CPL (Caribbean Premier League)
- **Target:** Fantasy score prediction per player per match
- **Elite Team Hit Rate:** Tracked via `multi_league_elite_rate_calculator.py`

The model performed best on CPL data — likely due to smaller player pool, consistent venues, and a cleaner squad context signal.

---

## 🚀 How to Run (High Level)

> ⚠️ Full data pipeline requires Cricsheet data download and local path configuration. See `docs/HOW_TO_RUN.md` for details.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ETL (pulls and processes match data)
python cricket_etl_pipeline.py

# 3. Generate training teams
python parquet_team_generator_ver2.1.py

# 4. Extract features
python enhanced_features.py

# 5. Train model
python train_xgboost.py

# 6. Validate
python validate_model_pipeline.py

# 7. Generate teams using trained model
python live_team_generator.py
```

---

## 📦 Dependencies

```
xgboost
pandas
numpy
scikit-learn
pyarrow          # Parquet read/write
fastparquet
joblib           # Model serialisation
requests         # API calls
```

---

---

## 🎯 MGAG — The Inference Loop

The most technically interesting part of this project is not the model — it's how the model's predictions are turned into an actual valid fantasy team.

The core challenge: **you can't just pick the 11 highest-scoring players.** Fantasy cricket has hard constraints — roles (bowlers, batters, all-rounders, keeper), budget caps, and team composition rules. Naively picking by score produces invalid teams constantly.

**MGAG solves this with a Monte Carlo approach:**

1. XGBoost scores every player in the upcoming match squad
2. A filter model removes obvious non-picks
3. `simsim.py` generates thousands of random *but valid* teams, weighting player selection probability by their predicted score
4. Teams are ranked by their total predicted score
5. `team_validator.py` checks every team against platform rules
6. Output: top-ranked valid fantasy XIs with captain/VC recommendations

The key insight: **higher model score → higher inclusion probability in sampled teams → good players naturally dominate the top recommendations**, without ever violating constraints.

See [`docs/MGAG_INFERENCE.md`](docs/MGAG_INFERENCE.md) for the full deep-dive.

## 🧑‍💻 About

Built by **Alok** — marketer, consultant, and amateur ML builder.

Background: Chemical Engineering + MBA in Marketing. Spent 5 years as a professional poker player (where modelling player behaviour under uncertainty became a habit). This project came from curiosity about whether the same probabilistic thinking that works at the poker table could predict fantasy cricket outcomes.

**Vibe coded entirely** — AI-assisted development without a traditional CS background. The system went through multiple training rounds (R2 → R3), each with better feature engineering and cleaner data pipelines.

---

## 📌 Status

This is a **showcase/portfolio project**. The trained CPL model works. Full live inference requires local data setup. The codebase reflects a real iterative ML development process — including dead ends, rewrites, and refinements — which is kind of the point.

---

## 📄 License

MIT — use freely, attribution appreciated.
