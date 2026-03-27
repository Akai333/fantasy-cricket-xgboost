# MGAG — Model-Guided Assisted Generation

## What Is MGAG?

MGAG is the inference engine of this system. The name reflects what it actually does: rather than the model *directly outputting* a fantasy team (which isn't possible — teams have hard constraints around roles, budget, and player count), MGAG uses the model's predictions to **guide** a constrained search for the best possible team.

Think of it as: **XGBoost scores players → MGAG assembles a legal team from those scores.**

This is a non-trivial problem. You can't just pick the top 11 players by predicted score — you'd end up with 11 batters, no wicket-keeper, over budget, with an invalid captain selection. MGAG handles all of that.

---

## The Core Problem MGAG Solves

Fantasy cricket has hard constraints:
- Exactly 11 players
- Must include: 1 wicket-keeper, 3-5 batters, 1-4 all-rounders, 3-5 bowlers
- Budget cap (e.g., 100 credits, players priced 7-12 each)
- Captain gets 2× points, Vice-Captain gets 1.5×
- Some platforms limit players from the same team (max 7 from one side)

A naive "sort by predicted score and pick top 11" fails these constraints constantly. MGAG solves this properly.

---

## How MGAG Works

```
Input: Upcoming match squads (Team A + Team B, ~30 players)
         │
         ▼
[Step 1] Score each player using XGBoost model
         → Predicted fantasy score per player
         │
         ▼
[Step 2] Filter: Remove players with very low predicted scores
         (cpl_filter model pre-screens obvious non-picks)
         │
         ▼
[Step 3] Constrained Team Search
         → Generate candidate teams respecting all constraints
         → Weight candidates by sum of player predicted scores
         → Use simulation (simsim.py) to explore the space
         │
         ▼
[Step 4] Captain/Vice-Captain Selection
         → Model identifies highest EV captain candidates
         → Considers 2× multiplier impact on total expected score
         │
         ▼
[Step 5] Validate generated teams (team_validator.py)
         → Checks all fantasy platform rules
         → Rejects invalid teams
         │
         ▼
Output: Ranked list of valid fantasy XI recommendations
```

---

## Key Files in MGAG

| File | Role |
|------|------|
| `mgag_orchestrator.py` | Main entry point — coordinates the full pipeline |
| `mgag_live_orchestrator.py` | Live match version — handles real-time squad updates |
| `simsim.py` | Simulation engine — generates and scores candidate teams |
| `simsim_to_mgag_converter.py` | Converts simsim output to MGAG-compatible format |
| `team_validator.py` | Validates teams against fantasy platform rules |
| `real_model_interface.py` | Clean interface between XGBoost model and MGAG |
| `smart_mgag.py` | Enhanced version with smarter search heuristics |
| `complete_mgag_pipeline.py` | End-to-end pipeline runner |
| `optimized_mgag_pipeline.py` | Performance-optimised version for faster inference |
| `correct_mgag_orchestrator.py` | Bug-fixed version of the core orchestrator |
| `corrected_model_interface.py` | Fixed model interface (precision improvements) |
| `find_cpl_matches.py` | Utility to find and load CPL match data for inference |
| `match_data_loader.py` | Loads and validates squad data for a match |

---

## The Simulation Engine (simsim.py)

`simsim` (short for "simulate simulation") is the core search algorithm inside MGAG. It:

1. Takes the model-scored player pool
2. Generates N random valid teams (satisfying all constraints)
3. Scores each team: `team_score = Σ(player_predicted_scores) + captain_bonus`
4. Returns teams ranked by predicted total score

This is essentially a **Monte Carlo search** guided by model predictions. Rather than exhaustively searching all possible team combinations (which would be computationally expensive — there are millions of valid 11-player combinations from 30 players), simsim samples intelligently by weighting player selection probability by their model score.

Higher model score → higher probability of being included in a sampled team → better teams rise to the top naturally.

---

## Elite vs Non-Elite: The Training Signal

The binary elite/non-elite label (top 10% of fantasy scores = elite) was the key design choice that made training possible:

- **Why not just predict the exact fantasy score?** Because exact scores are very noisy — a batter can score 50 runs or 0 runs, and both are plausible given the same pre-match features. Predicting the exact score is very hard.
- **Why top 10%?** This creates a meaningful signal. A team in the top 10% is one that "got the right players" — the batters who scored big, the bowlers who took wickets. The model learns what *pre-match features* predict these outcomes.
- **At inference time:** MGAG doesn't care about the exact predicted score — it uses the relative ranking of players to guide team selection. Player A predicted higher than Player B → prefer Player A.

---

## MGAG vs Simple Model Output

| Approach | What it does | Why it fails alone |
|----------|-------------|-------------------|
| Raw XGBoost output | Scores each player | Doesn't produce a valid team |
| Greedy top-N selection | Pick 11 highest scorers | Violates role/budget constraints |
| Random team generation | Generates valid teams | No signal — just random |
| **MGAG** | **Model-guided constrained search** | **This is what works** |

The insight: the model and the team generator are **better together** than either alone. The model provides signal; MGAG provides structure.

---

## Live Match Inference

`mgag_live_orchestrator.py` handles the real-world use case: generating a team recommendation for an *upcoming* match.

Inputs required:
- Match ID (to fetch correct squads)
- Both team squad lists with player roles and prices
- Venue information
- Recent form data (pulled from the feature pipeline)

Output:
- Top 3-5 recommended fantasy XIs
- Ranked by predicted total team score
- With captain/vice-captain recommendations for each

---

## Example Inference Flow (CPL Match)

```
Match: Jamaica Tallawahs vs Barbados Royals
Venue: Sabina Park, Kingston

→ Load squads (15 players each side)
→ Score all 30 players via XGBoost (CPL R3 model)

Top scored players (example output):
  1. [Batter A]      Predicted: 68.4   Role: BAT
  2. [All-rounder B] Predicted: 61.2   Role: AR
  3. [Bowler C]      Predicted: 58.9   Role: BOWL
  ...

→ simsim generates 10,000 valid teams
→ Teams ranked by total predicted score
→ Validator checks all constraints
→ Captain recommendation: [Batter A] (highest predicted + 2× multiplier EV)

Top recommended XI:
  Captain: [Batter A]
  Vice-Captain: [All-rounder B]
  [9 more players...]
  Total predicted score: 487.3
```

---

## Why This Architecture?

The MGAG approach was chosen over direct optimisation (e.g., integer linear programming) for a few reasons:

1. **Flexibility** — Fantasy platform rules change. Simulation is easier to update than a formal optimiser.
2. **Stochasticity** — In real fantasy contests, you want *some* variance in your team (not the same team as everyone else). Simulation naturally produces multiple good teams, not just one optimal solution.
3. **Speed** — Monte Carlo simulation with 10K samples runs in seconds. A formal ILP solver with 140+ features would be slower to set up and maintain.
4. **Interpretability** — It's easy to explain: "The model liked these players, so we built teams around them."
