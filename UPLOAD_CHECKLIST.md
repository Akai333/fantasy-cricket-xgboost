# 📋 FILES TO UPLOAD TO GITHUB

This is your exact upload checklist. Copy ONLY these files into a new folder, then push to GitHub.

---

## ✅ ROOT LEVEL — `E:\app_root\`

Copy these .py files:

- [ ] `cricket_etl_pipeline.py`          ← Core ETL (most important file)
- [ ] `run_enhanced_etl_2024.py`         ← ETL update
- [ ] `run_all_series_by_year.py`        ← Batch runner
- [ ] `features.py`                      ← Base features
- [ ] `enhanced_features.py`             ← Enhanced features
- [ ] `feature_config.py`               ← Feature config
- [ ] `squad_context.py`                ← Squad context
- [ ] `squad_context_feature_extractor.py`
- [ ] `enhanced_opportunity_feature_extractor.py`
- [ ] `venue_interaction_feature_extractor.py`
- [ ] `parquet_team_generator_ver2.1.py` ← Team generation (key file)
- [ ] `generate_validation_teams.py`
- [ ] `live_team_generator.py`
- [ ] `train_xgboost.py`                ← Model training (key file)
- [ ] `train_xgboost_fixed.py`
- [ ] `train_all_models_comprehensive.py`
- [ ] `stage1_xgboost_scorer.py`
- [ ] `validate_model_pipeline.py`
- [ ] `feature_importance_analysis.py`
- [ ] `feature_importance_explained.py`
- [ ] `investigate_predictions.py`
- [ ] `multi_league_elite_rate_calculator.py`
- [ ] `extract_enhanced_features_and_compare.py`
- [ ] `simsim.py`

---

## ✅ DATABASE_JOIN SUBFOLDER — `E:\app_root\database_join\Scripts\`

- [ ] `cricket_etl_pipeline.py`    (rename to `cricket_etl_pipeline_v2.py` if same filename)
- [ ] `cricket_venue_context.py`

---

## ✅ R3_TRAINING SUBFOLDER — `E:\app_root\R3_Training\`

- [ ] `R3_comprehensive_feature_extractor_fixed_v2.py`  ← Most important R3 file
- [ ] `R3_elite_discovery_trainer.py`
- [ ] `R3_exact_feature_trainer.py`
- [ ] `R3_exact_feature_trainer_fixed.py`
- [ ] `contextual_cvc_feature_extractor.py`
- [ ] `practical_clean_pipeline.py`
- [ ] `run_pipeline_with_checkpoints.py`
- [ ] `validate_training_ready.py`
- [ ] `R3_model_evaluator.py`
- [ ] `R3_venue_interaction_feature_extractor_refined.py`
- [ ] `R3_strategic_team_features_extractor.py`
- [ ] `README.md`  (if it exists in this folder)

---

## ✅ MGAG SUBFOLDER — `E:\app_root\MGAG\`

This is the inference engine — the most important output folder. Be selective: skip the debug_* and test_* files.

**Core orchestration (MUST HAVE):**
- [ ] `mgag_orchestrator.py`              ← Main inference entry point
- [ ] `mgag_live_orchestrator.py`         ← Live match version
- [ ] `correct_mgag_orchestrator.py`      ← Bug-fixed production version
- [ ] `complete_mgag_pipeline.py`         ← End-to-end runner
- [ ] `optimized_mgag_pipeline.py`        ← Optimised version

**Simulation engine (MUST HAVE):**
- [ ] `simsim.py`                         ← Monte Carlo team search
- [ ] `simsim_to_mgag_converter.py`       ← Converts simsim output

**Model interface (MUST HAVE):**
- [ ] `real_model_interface.py`           ← XGBoost ↔ MGAG interface
- [ ] `corrected_model_interface.py`      ← Fixed precision version
- [ ] `smart_mgag.py`                     ← Enhanced heuristics

**Team validation (MUST HAVE):**
- [ ] `team_validator.py`                 ← Platform rule checker

**Data loading:**
- [ ] `match_data_loader.py`              ← Squad data loader
- [ ] `find_cpl_matches.py`               ← CPL match lookup
- [ ] `find_cpl_from_database.py`         ← Database version

**Documentation:**
- [ ] `MGAG.md`                           ← Keep this — good inline docs
- [ ] `LIVE_MATCH_USER_GUIDE.md`          ← User-facing guide

**SKIP these (debug/test files, not needed for showcase):**
- ❌ `debug_*.py` — all debug scripts
- ❌ `test_*.py` — all test scripts  
- ❌ `corrected_results_*.json` — output files, too large
- ❌ `fixed_mgag_results_*.json` — output files

---

## ❌ DO NOT UPLOAD — Exclude These

| What | Why |
|------|-----|
| `.venv/` folder | Virtual environment, huge, not needed |
| `__pycache__/` | Python bytecode cache |
| `R3_Features_output/` | GBs of parquet files |
| `R3_Training/R3_models/` | Large .joblib model files |
| `Live_Matches/` | Raw match data, GBs |
| `Trained model/` | .joblib files too large for GitHub |
| `previous_chat_*.md` | Your chat logs with AI, not relevant |
| `Saved Chats/` | Same |
| `*.joblib` | Model binary files (over 100MB limit) |
| `*.csv` (large ones) | league_80_percent_summary.csv etc. |
| `enhanced_cvc_model_backup_*/` | Backup folder |
| `Cem_app Backup files/` | Backup |
| `UI/`, `UI_ver_2/` | UI experiments not core to the model |

---

## 📁 Suggested Final GitHub Folder Structure

```
fantasy-cricket-predictor/
│
├── README.md                          ← (from docs I created)
├── requirements.txt                   ← (create this)
├── .gitignore                         ← (create this)
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FEATURE_ENGINEERING.md
│   ├── HOW_TO_RUN.md
│   └── RESULTS.md
│
├── etl/
│   ├── cricket_etl_pipeline.py
│   ├── run_enhanced_etl_2024.py
│   ├── run_all_series_by_year.py
│   └── cricket_venue_context.py
│
├── features/
│   ├── features.py
│   ├── enhanced_features.py
│   ├── feature_config.py
│   ├── squad_context.py
│   ├── squad_context_feature_extractor.py
│   ├── enhanced_opportunity_feature_extractor.py
│   └── venue_interaction_feature_extractor.py
│
├── team_generation/
│   ├── parquet_team_generator_ver2.1.py
│   ├── generate_validation_teams.py
│   └── live_team_generator.py
│
├── training/
│   ├── train_xgboost.py
│   ├── train_xgboost_fixed.py
│   ├── train_all_models_comprehensive.py
│   ├── stage1_xgboost_scorer.py
│   └── r3/
│       ├── R3_comprehensive_feature_extractor_fixed_v2.py
│       ├── R3_elite_discovery_trainer.py
│       └── R3_model_evaluator.py
│
├── evaluation/
│   ├── validate_model_pipeline.py
│   ├── feature_importance_analysis.py
│   ├── feature_importance_explained.py
│   └── multi_league_elite_rate_calculator.py
│
└── inference/
    ├── mgag_orchestrator.py
    ├── mgag_live_orchestrator.py
    ├── simsim.py
    ├── team_validator.py
    └── real_model_interface.py
```

---

## 📝 Two More Files to Create

### `requirements.txt` (create this file)
```
xgboost>=1.7.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
pyarrow>=10.0.0
fastparquet>=0.8.0
joblib>=1.2.0
requests>=2.28.0
tqdm>=4.64.0
```

### `.gitignore` (create this file)
```
.venv/
__pycache__/
*.pyc
*.joblib
*.parquet
*.csv
R3_Features_output/
Live_Matches/
Trained model/
R3_Training/R3_models/
previous_chat_*.md
Saved Chats/
enhanced_cvc_model_backup_*/
.vscode/
```
