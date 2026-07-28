"""A standard soccer Elo rating system with home advantage, a
goal-difference margin-of-victory multiplier (the same MOV formula
used by eloratings.net's World Football Elo Ratings), and
season-to-season regression toward the mean (rosters turn over
enough between seasons that carrying a rating forward unadjusted
overstates how much last season's form should count)."""
from collections import defaultdict
import pandas as pd

BASE_RATING = 1500.0


def mov_multiplier(goal_diff: int) -> float:
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


def run_elo(df: pd.DataFrame, home_advantage: float, k_factor: float,
            regression_factor: float = 1.0):
    """regression_factor: fraction of a team's rating (relative to the
    mean) carried into a new season. 1.0 = no regression (full carry-over),
    0.0 = fully reset to the mean every season."""
    ratings = defaultdict(lambda: BASE_RATING)
    predictions = []
    current_season = None

    for row in df.itertuples():
        if current_season is not None and row.season != current_season:
            for team in list(ratings.keys()):
                ratings[team] = BASE_RATING + regression_factor * (ratings[team] - BASE_RATING)
        current_season = row.season

        home, away = row.home_team, row.away_team
        r_home, r_away = ratings[home], ratings[away]

        exp_home = expected_score(r_home + home_advantage, r_away)

        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        goal_diff = row.home_score - row.away_score
        k = k_factor * mov_multiplier(goal_diff)
        delta = k * (actual_home - exp_home)

        new_r_home = r_home + delta
        new_r_away = r_away - delta

        predictions.append({
            "date": row.date,
            "season": row.season,
            "home_team": home,
            "away_team": away,
            "home_score": row.home_score,
            "away_score": row.away_score,
            "pre_rating_home": r_home,
            "pre_rating_away": r_away,
            "post_rating_home": new_r_home,
            "post_rating_away": new_r_away,
            "predicted_home_prob": exp_home,
            "actual_home_result": actual_home,
        })

        ratings[home] = new_r_home
        ratings[away] = new_r_away

    return pd.DataFrame(predictions), dict(ratings)


if __name__ == "__main__":
    df = pd.read_csv("data/nwsl_matches.csv")
    # Tuned via tune.py: grid-searched on 2021-2023, validated against a
    # genuinely held-out 2024-2026 test set (see tune.py for methodology).
    preds, final_ratings = run_elo(df, home_advantage=25.0, k_factor=10.0, regression_factor=0.5)
    preds.to_csv("data/predictions.csv", index=False)

    print("Final ratings (sorted):")
    for team, rating in sorted(final_ratings.items(), key=lambda x: -x[1]):
        print(f"  {team:28s} {rating:7.1f}")
