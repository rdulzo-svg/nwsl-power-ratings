"""Generate win/draw/loss previews for upcoming NWSL matches using a
team-specific attack/defense Poisson model (see attack_defense_model.py),
fit on the two most recent seasons so every current team — including
brand-new expansion sides — has real data behind its rating.

This replaced an earlier version that estimated expected goals from a
single global regression on Elo win probability. Backtested head to
head on a fair, unseen-team-free split: the team-specific model scored
a real Brier improvement (0.6290 vs 0.6398), not just added complexity
for its own sake.
"""
import requests
import pandas as pd
import attack_defense_model as adm
from goal_model import win_draw_loss

ATTACK_DEFENSE_SEASONS = {2025, 2026}  # most recent window covering every current team

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.nwsl/scoreboard"

# ESPN's API omits "FC" for this one team; normalize to match our team registry.
NAME_FIXES = {
    "Utah Royals": "Utah Royals FC",
}


def fitted_attack_defense_model():
    matches = pd.read_csv("data/nwsl_matches.csv")
    ad_matches = matches[matches["season"].isin(ATTACK_DEFENSE_SEASONS)]
    model, teams = adm.fit(ad_matches)
    return model, teams


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
    model, teams = fitted_attack_defense_model()
    matches = pull_upcoming(start, end)

    rows = []
    for m in matches:
        if m["home_team"] not in teams or m["away_team"] not in teams:
            print(f"Skipping {m['home_team']} vs {m['away_team']}: team not in fitted model")
            continue

        lambda_home, lambda_away = adm.expected_goals(model, m["home_team"], m["away_team"])
        p_home, p_draw, p_away = win_draw_loss(lambda_home, lambda_away)
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

    print(f"\n{len(df)} upcoming matches, {start} to {end}\n")
    for r in df.itertuples():
        print(f"{r.date[:10]}  {r.home_team:24s} {r.home_win_pct:5.1f}%  "
              f"draw {r.draw_pct:4.1f}%  {r.away_win_pct:5.1f}% {r.away_team}")


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "20260729"
    end = sys.argv[2] if len(sys.argv) > 2 else "20260805"
    main(start, end)
