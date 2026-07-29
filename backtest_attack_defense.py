"""Backtest the team-specific attack/defense model against the current
global linear-regression-on-Elo-probability model, on the same
held-out 2024-2026 test period used throughout this project.
"""
import pandas as pd
from goal_model import win_draw_loss as poisson_win_draw_loss
import attack_defense_model as adm

TUNE_SEASONS = {2023, 2024}
TEST_SEASONS = {2025}  # every 2025 team already existed by 2024; a fair test of the method itself


def three_way_brier(p_home, p_draw, p_away, actual_home, actual_draw, actual_away):
    return ((p_home - actual_home) ** 2 + (p_draw - actual_draw) ** 2
             + (p_away - actual_away) ** 2)


def main():
    matches = pd.read_csv("data/nwsl_matches.csv")
    tune_matches = matches[matches["season"].isin(TUNE_SEASONS)]
    test_matches = matches[matches["season"].isin(TEST_SEASONS)].copy()

    model, teams = adm.fit(tune_matches)

    # Compare against the current model's predictions, already computed
    # per-match in predictions.csv (predicted_home_prob + the fitted
    # global goal-regression coefficients from goal_model.py).
    from goal_model import fit_goal_model, predict as global_predict
    preds = pd.read_csv("data/predictions.csv")
    tune_preds = preds[preds["season"].isin(TUNE_SEASONS)]
    home_coef, away_coef = fit_goal_model(tune_preds)
    test_preds = preds[preds["season"].isin(TEST_SEASONS)].set_index(
        ["date", "home_team", "away_team"])

    old_briers, new_briers = [], []
    skipped = 0
    for row in test_matches.itertuples():
        if row.home_team not in teams or row.away_team not in teams:
            skipped += 1
            continue

        if row.home_score > row.away_score:
            actual = (1, 0, 0)
        elif row.home_score < row.away_score:
            actual = (0, 0, 1)
        else:
            actual = (0, 1, 0)

        # New: team-specific attack/defense expected goals -> W/D/L
        lh, la = adm.expected_goals(model, row.home_team, row.away_team)
        p_new = poisson_win_draw_loss(lh, la)
        new_briers.append(three_way_brier(*p_new, *actual))

        # Old: global regression on Elo-implied win probability
        key = (row.date, row.home_team, row.away_team)
        if key in test_preds.index:
            exp_home = test_preds.loc[key, "predicted_home_prob"]
            if hasattr(exp_home, "iloc"):
                exp_home = exp_home.iloc[0]
            p_old = global_predict(exp_home, home_coef, away_coef)
            old_briers.append(three_way_brier(*p_old, *actual))

    print(f"Test matches: {len(test_matches)}, skipped (unseen team): {skipped}")
    print(f"Evaluated -- old: {len(old_briers)}, new: {len(new_briers)}\n")
    print(f"{'Model':45s} {'3-way Brier':>12s}")
    print(f"{'Old: global regression on Elo prob':45s} {sum(old_briers)/len(old_briers):12.4f}")
    print(f"{'New: team-specific attack/defense':45s} {sum(new_briers)/len(new_briers):12.4f}")

    print("\nSample attack/defense ratings (tune-period fit):")
    r = adm.ratings(model, teams).sort_values("attack", ascending=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
