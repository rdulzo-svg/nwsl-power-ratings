"""Generate win/draw/loss previews for upcoming NWSL matches using
current Elo ratings.

Elo's raw expected score isn't a clean 3-outcome probability (a draw
counts as 0.5 "score" for both teams), so it can't be published
directly as "win probability" without overstating precision. We
decompose it using a Poisson goal-scoring model (see goal_model.py)
fit on real historical scorelines, rather than a single constant
draw rate applied to every match regardless of how mismatched the
teams are — mismatched games really do produce fewer draws than
close ones, and the fitted model reflects that directly.
"""
import requests
import pandas as pd
from elo_model import run_elo, expected_score
from goal_model import fit_goal_model, predict as goal_model_predict

HOME_ADVANTAGE = 25.0
K_FACTOR = 10.0
REGRESSION_FACTOR = 0.5

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.nwsl/scoreboard"

# ESPN's API omits "FC" for this one team; normalize to match our team registry.
NAME_FIXES = {
    "Utah Royals": "Utah Royals FC",
}


def current_ratings() -> dict:
    df = pd.read_csv("data/nwsl_matches.csv")
    _, final_ratings = run_elo(df, home_advantage=HOME_ADVANTAGE, k_factor=K_FACTOR,
                                regression_factor=REGRESSION_FACTOR)
    return final_ratings


def fitted_goal_coefficients():
    preds = pd.read_csv("data/predictions.csv")
    tune_set = preds[preds["season"].isin([2021, 2022, 2023])]
    return fit_goal_model(tune_set)


def pull_upcoming(start: str, end: str) -> list[dict]:
    params = {"dates": f"{start}-{end}", "limit": 1000}
    resp = requests.get(BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        if comp["status"]["type"]["completed"]:
            continue

        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")

        home_name = NAME_FIXES.get(home["team"]["displayName"], home["team"]["displayName"])
        away_name = NAME_FIXES.get(away["team"]["displayName"], away["team"]["displayName"])

        rows.append({
            "date": event["date"],
            "home_team": home_name,
            "away_team": away_name,
        })
    return rows


def main(start: str, end: str):
    ratings = current_ratings()
    home_coef, away_coef = fitted_goal_coefficients()
    matches = pull_upcoming(start, end)

    rows = []
    for m in matches:
        r_home = ratings.get(m["home_team"], 1500.0)
        r_away = ratings.get(m["away_team"], 1500.0)
        exp_home = expected_score(r_home + HOME_ADVANTAGE, r_away)
        p_home, p_draw, p_away = goal_model_predict(exp_home, home_coef, away_coef)
        rows.append({
            "date": m["date"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_win_pct": round(p_home * 100, 1),
            "draw_pct": round(p_draw * 100, 1),
            "away_win_pct": round(p_away * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv("data/upcoming_previews.csv", index=False)

    print(f"{len(df)} upcoming matches, {start} to {end}\n")
    for r in df.itertuples():
        print(f"{r.date[:10]}  {r.home_team:24s} {r.home_win_pct:5.1f}%  "
              f"draw {r.draw_pct:4.1f}%  {r.away_win_pct:5.1f}% {r.away_team}")


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "20260729"
    end = sys.argv[2] if len(sys.argv) > 2 else "20260805"
    main(start, end)
