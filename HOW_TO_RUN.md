# How to Run

> ⚠️ This is a portfolio/showcase project. The full pipeline requires downloading Cricsheet data and configuring local paths. These instructions are for reference — the code demonstrates the approach, not a one-click deployment.

---

## Prerequisites

- Python 3.9+
- ~10GB disk space for full data pipeline
- Windows or Linux (developed on Windows)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fantasy-cricket-predictor
cd fantasy-cricket-predictor

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Dependencies (`requirements.txt`)

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

---

## Running the Pipeline

### Step 1: ETL — Pull and Process Match Data

```bash
python cricket_etl_pipeline.py
```

This pulls ball-by-ball data from Cricsheet, computes per-player fantasy scores, and outputs match CSVs.

To run for a specific year/series:
```bash
python run_all_series_by_year.py --year 2023
```

To add venue context:
```bash
python database_join/cricket_venue_context.py
```

---

### Step 2: Generate Simulated Fantasy Teams

```bash
python parquet_team_generator_ver2.1.py
```

Generates N random valid fantasy teams per match, scores them, and labels elite vs non-elite.

---

### Step 3: Feature Engineering

```bash
python enhanced_features.py
```

Or for the full R3 feature extraction:
```bash
python R3_Training/R3_comprehensive_feature_extractor_fixed_v2.py
```

Output: One Parquet file per match in `R3_Features_output/clean_140_features/`

---

### Step 4: Train the Model

```bash
python train_xgboost.py
```

For CPL-specific training (best results):
```bash
cd R3_Training
python R3_elite_discovery_trainer.py
```

Model saved to `R3_Training/R3_models/` as `.joblib` files.

---

### Step 5: Validate

```bash
python validate_model_pipeline.py
python feature_importance_analysis.py
```

---

### Step 6: Generate Teams (Inference)

```bash
python live_team_generator.py
```

Or via the MGAG orchestrator:
```bash
cd MGAG
python mgag_orchestrator.py
```

---

## Data Notes

- **Source:** [Cricsheet](https://cricsheet.org/) — free, public, ball-by-ball cricket data
- **Format:** YAML/JSON → processed to CSV → features stored as Parquet
- **Leagues covered:** CPL, IPL, BBL, PSL (CPL performs best)
- **Data not included in repo** — too large, but fully reproducible from Cricsheet

---

## Project Layout for Running

```
E:/app_root/              ← Project root (you set this)
├── Live_Matches/         ← Downloaded match data
├── R3_Features_output/   ← Generated feature parquets
│   └── clean_140_features/
├── R3_Training/
│   └── R3_models/        ← Saved model files
└── [all .py files]
```
