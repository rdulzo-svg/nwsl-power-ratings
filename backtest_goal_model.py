"""Compare the new Poisson goal model against the old constant-draw-rate
approach on the same held-out 2024-2026 test period used to validate
the Elo model itself. Uses the 3-way (multi-class) Brier score: lower
is better, 0 is a perfect prediction.
"""
import pandas as pd
from goal_model import fit_goal_model, predict as poisson_predict

TUNE_SEASONS = {2021, 2022, 2023}
TEST_SEASONS = {2024, 2025, 2026}
OLD_DRAW_RATE = 0.231


def three_way_brier(p_home, p_draw, p_away, actual_home, actual_draw, actual_away):
    return ((p_home - actual_home) ** 2 + (p_draw - actual_draw) ** 2
             + (p_away - actual_away) ** 2)


def old_constant_rate_predict(exp_home: float) -> tuple[float, float, float]:
    p_draw = OLD_DRAW_RATE
    p_home = exp_home - 0.5 * p_draw
    p_away = 1 - p_home - p_draw
    p_home = max(0.0, min(1.0, p_home))
    p_away = max(0.0, min(1.0, p_away))
    p_draw = 1 - p_home - p_away
    return p_home, p_draw, p_away


def main():
    preds = pd.read_csv("data/predictions.csv")
    tune_set = preds[preds["season"].isin(TUNE_SEASONS)]
    test_set = preds[preds["season"].isin(TEST_SEASONS)].copy()

    # Fit the goal model on the SAME tune period used for Elo tuning,
    # so the held-out test set stays genuinely unseen.
    home_coef, away_coef = fit_goal_model(tune_set)

    old_briers, new_briers = [], []
    for row in test_set.itertuples():
        if row.home_score > row.away_score:
            actual = (1, 0, 0)
        elif row.home_score < row.away_score:
            actual = (0, 0, 1)
        else:
            actual = (0, 1, 0)

        p_old = old_constant_rate_predict(row.predicted_home_prob)
        p_new = poisson_predict(row.predicted_home_prob, home_coef, away_coef)

        old_briers.append(three_way_brier(*p_old, *actual))
        new_briers.append(three_way_brier(*p_new, *actual))

    print(f"Held-out test period {sorted(TEST_SEASONS)}, n={len(test_set)}\n")
    print(f"{'Model':40s} {'3-way Brier':>12s}")
    print(f"{'Old: constant 23.1% draw rate':40s} {sum(old_briers)/len(old_briers):12.4f}")
    print(f"{'New: Poisson goal model':40s} {sum(new_briers)/len(new_briers):12.4f}")

    # Does the new model actually produce varying draw probabilities?
    print("\nDraw probability range on test set:")
    draw_probs = [poisson_predict(p, home_coef, away_coef)[1] for p in test_set["predicted_home_prob"]]
    print(f"  min={min(draw_probs):.1%}  max={max(draw_probs):.1%}  "
          f"mean={sum(draw_probs)/len(draw_probs):.1%}")

    print(f"\nFitted home goals: {home_coef[0]:.2f} * prob + {home_coef[1]:.2f}")
    print(f"Fitted away goals: {away_coef[0]:.2f} * prob + {away_coef[1]:.2f}")


if __name__ == "__main__":
    main()
