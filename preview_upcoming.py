"""Generate win/draw/loss previews for upcoming NWSL matches using
current Elo ratings.

Elo's raw expected score isn't a clean 3-outcome probability (a draw
counts as 0.5 "score" for both teams), so it can't be published
directly as "win probability" without overstating precision. We
decompose it using NWSL's empirical historical draw rate:
    expected_home = P(home win) + 0.5 * P(draw)
So, given an estimated P(draw):
    P(home win) = expected_home - 0.5 * P(draw)
    P(away win) = 1 - P(home win) - P(draw)
"""
import requests
import pandas as pd
from elo_model import run_elo, expected_score

HOME_ADVANTAGE = 25.0
K_FACTOR = 10.0
REGRESSION_FACTOR = 0.5
DRAW_RATE = 0.231  # empirical rate from backtest.py's historical analysis

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


def win_draw_loss(rating_home: float, rating_away: float) -> tuple[float, float, float]:
    exp_home = expected_score(rating_home + HOME_ADVANTAGE, rating_away)
    p_draw = DRAW_RATE
    p_home = exp_home - 0.5 * p_draw
    p_away = 1 - p_home - p_draw

    # Defensive clip in case future ratings drift outside the historical range.
    p_home = max(0.0, min(1.0, p_home))
    p_away = max(0.0, min(1.0, p_away))
    p_draw = 1 - p_home - p_away
    return p_home, p_draw, p_away


def main(start: str, end: str):
    ratings = current_ratings()
    matches = pull_upcoming(start, end)

    rows = []
    for m in matches:
        r_home = ratings.get(m["home_team"], 1500.0)
        r_away = ratings.get(m["away_team"], 1500.0)
        p_home, p_draw, p_away = win_draw_loss(r_home, r_away)
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
