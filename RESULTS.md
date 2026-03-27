# Results & Model Performance

## Summary

The XGBoost regressor trained on CPL (Caribbean Premier League) data achieved the best performance across all leagues tested. The model successfully identifies fantasy score patterns from player form, venue context, and squad composition features.

---

## Training Configuration (R3 CPL Model)

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost Regressor |
| League | CPL (Caribbean Premier League) |
| Feature Count | ~140 |
| Training Matches | CPL 2018–2023 |
| Validation | Held-out match set |
| Model files | `cpl_regressor_20250813_121421.joblib` + filter |

---

## Why CPL Performed Best

The model was tested across CPL, IPL, BBL (Big Bash League) and PSL (Pakistan Super League). CPL consistently gave the cleanest results for several reasons:

1. **Small player pool** — CPL teams have ~15-18 players per squad with limited overseas slots. Less variance in player availability than IPL.
2. **Consistent venues** — 5-6 Caribbean grounds with well-established characteristics. Less pitch variation than Indian conditions.
3. **Shorter season** — 6 weeks means recent form features carry more signal (less data drift).
4. **Fewer confounders** — No DLS intervention frequency, no home-crowd effects at the scale seen in IPL.

---

## Elite Team Classification

The core evaluation metric is **Elite Hit Rate** — what % of the model's top-recommended teams fall in the actual top 10% of fantasy scores for that match.

| Approach | Elite Hit Rate (CPL) |
|----------|---------------------|
| Random team selection | ~10% (baseline — by definition) |
| Model-guided selection | Significantly above baseline |

Tracked in detail via `multi_league_elite_rate_calculator.py`.

---

## Feature Importance

Top features by XGBoost importance (CPL R3 model):

| Rank | Feature | Category |
|------|---------|----------|
| 1 | `fantasy_score_mean_5` | Player Form |
| 2 | `batting_opportunity_score` | Opportunity |
| 3 | `player_venue_fantasy_avg` | Player×Venue |
| 4 | `bowling_load_share` | Squad Context |
| 5 | `venue_avg_first_innings` | Venue |
| 6 | `batting_position_modal` | Squad Context |
| 7 | `fantasy_score_std_5` | Player Form |
| 8 | `expected_overs_bowled` | Opportunity |
| 9 | `captain_ev_multiplier` | Team Selection |
| 10 | `strike_rate_10` | Player Form |

Key insight: **Recent form dominates**, but opportunity features (will this player actually bat/bowl?) are nearly as important. A player in great form who bats at #7 in T20 may not get enough balls.

---

## Model Iterations

### Round 1 (R1)
- All-leagues training
- ~40 features (basic form + position)
- Inconsistent across leagues

### Round 2 (R2)
- Added venue features
- Better ETL — fixed data leakage issues where future data was leaking into training features
- Improved rolling window calculations

### Round 3 (R3) — Current Best
- CPL-focused
- 140 features including player×venue interactions
- Two-stage model: filter (classify playable vs. not) + regressor (score prediction)
- Cleaner parquet-based data pipeline
- Best elite hit rate achieved

---

## Known Limitations

1. **Data volume** — CPL has fewer matches than IPL, limiting the training corpus
2. **Squad data** — Player availability (injuries, resting) is hard to predict and not in the feature set
3. **Live pitches** — The model can't see the actual pitch on match day, only historical venue averages
4. **Toss effect** — Winning the toss and choosing to bat/field significantly affects outcomes; this is partially captured but not perfectly
5. **Debutants** — New players have no form history; the model handles this poorly (cold start problem)

---

## Next Steps (If Continued)

- Add toss decision as a feature (it's available pre-match)
- Experiment with player embeddings to handle cold starts
- Try LightGBM or CatBoost as alternatives
- Expand to IPL with more sophisticated home-ground normalisation
- Build a simple web interface for match-day team recommendations
