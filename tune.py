"""Grid-search Elo parameters on a tuning period, then report honest
out-of-sample performance on a held-out test period the search never
saw. This avoids fitting hyperparameters directly to the same data
used for the final "did it work" check.
"""
import itertools
import pandas as pd
from elo_model import run_elo

BURN_IN_SEASON = 2019
TUNE_SEASONS = {2021, 2022, 2023}
TEST_SEASONS = {2024, 2025, 2026}

HOME_ADVANTAGE_GRID = [0, 10, 25, 40, 50, 65, 75, 100]
K_FACTOR_GRID = [3, 5, 7, 10, 12, 15, 20, 25, 30, 40]
REGRESSION_GRID = [1.0, 0.85, 0.75, 0.65, 0.5, 0.35, 0.2, 0.0]


def brier_score(predicted, actual):
    return ((predicted - actual) ** 2).mean()


def evaluate(preds: pd.DataFrame, seasons: set) -> float:
    subset = preds[preds["season"].isin(seasons)]
    return brier_score(subset["predicted_home_prob"], subset["actual_home_result"])


def main():
    df = pd.read_csv("data/nwsl_matches.csv")

    results = []
    for home_adv, k, reg in itertools.product(HOME_ADVANTAGE_GRID, K_FACTOR_GRID, REGRESSION_GRID):
        preds, _ = run_elo(df, home_advantage=home_adv, k_factor=k, regression_factor=reg)
        tune_score = evaluate(preds, TUNE_SEASONS)
        results.append((home_adv, k, reg, tune_score))

    results.sort(key=lambda r: r[3])
    best_home_adv, best_k, best_reg, best_tune_score = results[0]

    print(f"Searched {len(results)} parameter combinations.")
    print(f"Best on TUNE period ({sorted(TUNE_SEASONS)}):")
    print(f"  home_advantage={best_home_adv}, k_factor={best_k}, regression_factor={best_reg}")
    print(f"  Tune-period Brier score: {best_tune_score:.4f}\n")

    print("Top 5 combinations on tune period:")
    for home_adv, k, reg, score in results[:5]:
        print(f"  home_adv={home_adv:4d}  k={k:3d}  regression={reg:.2f}  brier={score:.4f}")

    # Now the honest check: run the BEST combo and see how it does on
    # the held-out test seasons it was never optimized against.
    preds, _ = run_elo(df, home_advantage=best_home_adv, k_factor=best_k, regression_factor=best_reg)
    test_score = evaluate(preds, TEST_SEASONS)

    test_subset = preds[preds["season"].isin(TEST_SEASONS)]
    naive_rate = test_subset["actual_home_result"].mean()
    naive_score = brier_score(naive_rate, test_subset["actual_home_result"])
    coinflip_score = brier_score(0.5, test_subset["actual_home_result"])

    # Compare against the untuned defaults from the first pass, on the same held-out set
    default_preds, _ = run_elo(df, home_advantage=65.0, k_factor=20.0, regression_factor=1.0)
    default_test_score = evaluate(default_preds, TEST_SEASONS)

    print(f"\n--- Held-out test period {sorted(TEST_SEASONS)} (never used in tuning) ---")
    print(f"{'Model':45s} {'Brier score':>12s}")
    print(f"{'Tuned Elo':45s} {test_score:12.4f}")
    print(f"{'Untuned Elo (first-pass defaults)':45s} {default_test_score:12.4f}")
    print(f"{'Naive: constant historical home rate':45s} {naive_score:12.4f}")
    print(f"{'Naive: 50/50 coin flip':45s} {coinflip_score:12.4f}")

    decisive = test_subset[test_subset["home_score"] != test_subset["away_score"]].copy()
    decisive["elo_pick_home"] = decisive["predicted_home_prob"] > 0.5
    decisive["home_won"] = decisive["home_score"] > decisive["away_score"]
    tuned_acc = (decisive["elo_pick_home"] == decisive["home_won"]).mean()
    naive_acc = decisive["home_won"].mean()
    print(f"\nPick-the-winner accuracy on held-out decisive matches ({len(decisive)}):")
    print(f"  Tuned Elo:                {tuned_acc:.1%}")
    print(f"  Naive 'always pick home': {naive_acc:.1%}")


if __name__ == "__main__":
    main()
