"""Validate the Elo model against naive baselines using Brier score
(mean squared error between predicted probability and actual outcome,
lower is better). Excludes the first season as a burn-in period since
every team starts at the same arbitrary base rating with zero
informational content."""
import pandas as pd

BURN_IN_SEASON = 2019


def brier_score(predicted, actual):
    return ((predicted - actual) ** 2).mean()


def main():
    preds = pd.read_csv("data/predictions.csv")
    eval_set = preds[preds["season"] > BURN_IN_SEASON].copy()

    print(f"Full dataset: {len(preds)} matches")
    print(f"Evaluation set (excluding {BURN_IN_SEASON} burn-in): {len(eval_set)} matches\n")

    elo_brier = brier_score(eval_set["predicted_home_prob"], eval_set["actual_home_result"])

    coin_flip_brier = brier_score(0.5, eval_set["actual_home_result"])

    historical_home_rate = eval_set["actual_home_result"].mean()
    constant_rate_brier = brier_score(historical_home_rate, eval_set["actual_home_result"])

    print(f"Historical home 'non-loss' rate: {historical_home_rate:.3f}\n")
    print(f"{'Model':40s} {'Brier score':>12s}")
    print(f"{'Elo (team-specific ratings)':40s} {elo_brier:12.4f}")
    print(f"{'Naive: constant 50/50 coin flip':40s} {coin_flip_brier:12.4f}")
    print(f"{'Naive: constant historical home rate':40s} {constant_rate_brier:12.4f}")

    print()
    if elo_brier < constant_rate_brier:
        improvement = (constant_rate_brier - elo_brier) / constant_rate_brier * 100
        print(f"Elo beats the historical-rate baseline by {improvement:.1f}% (lower Brier = better).")
    else:
        print("Elo does NOT beat the historical-rate baseline — team-specific ratings aren't adding value yet.")

    # Accuracy on decisive (non-draw) matches only, using 0.5 as the threshold
    decisive = eval_set[eval_set["home_score"] != eval_set["away_score"]].copy()
    decisive["elo_pick_home"] = decisive["predicted_home_prob"] > 0.5
    decisive["home_won"] = decisive["home_score"] > decisive["away_score"]
    accuracy = (decisive["elo_pick_home"] == decisive["home_won"]).mean()
    naive_home_accuracy = decisive["home_won"].mean()

    print(f"\nOn decisive (non-draw) matches only ({len(decisive)} matches):")
    print(f"  Elo pick-the-winner accuracy:        {accuracy:.1%}")
    print(f"  Naive 'always pick home team' accuracy: {naive_home_accuracy:.1%}")


if __name__ == "__main__":
    main()
