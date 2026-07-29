"""Monte Carlo playoff probability simulation.

Starts from actual current standings (real points, real goal
difference), simulates every remaining regular-season match using the
attack/defense Poisson model, and tallies how often each team finishes
in the top 8 (the real NWSL playoff cutoff) across many simulated
seasons.

Simplification worth being upfront about: real NWSL tiebreakers go
points -> head-to-head -> goal difference -> goals scored. Tracking
head-to-head across thousands of simulated seasons is a lot of extra
bookkeeping for a tiebreaker that rarely swings a top-8 cutoff; this
simulation uses points then goal difference only.
"""
import requests
import numpy as np
import pandas as pd
import attack_defense_model as adm

STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/soccer/usa.nwsl/standings"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.nwsl/scoreboard"

ATTACK_DEFENSE_SEASONS = {2025, 2026}
PLAYOFF_SPOTS = 8
N_SIMULATIONS = 10000

NAME_FIXES = {
    "Utah Royals": "Utah Royals FC",
}


def current_standings(season: int) -> pd.DataFrame:
    resp = requests.get(STANDINGS_URL, params={"season": season}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    entries = data["children"][0]["standings"]["entries"]

    rows = []
    for e in entries:
        stats = {s["name"]: s.get("value", 0.0) for s in e["stats"]}
        name = e["team"]["displayName"]
        name = NAME_FIXES.get(name, name)
        rows.append({
            "team": name,
            "points": stats.get("points", 0.0),
            "goal_diff": stats.get("pointDifferential", 0.0),
        })
    return pd.DataFrame(rows)


def remaining_schedule(start: str, end: str) -> list[dict]:
    resp = requests.get(SCOREBOARD_URL, params={"dates": f"{start}-{end}", "limit": 1000}, timeout=30)
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
        rows.append({"home_team": home_name, "away_team": away_name})
    return rows


def simulate(standings: pd.DataFrame, schedule: list[dict], model, model_teams: list,
             n_sims: int = N_SIMULATIONS, seed: int = 42):
    rng = np.random.default_rng(seed)

    teams = list(standings["team"])
    team_index = {t: i for i, t in enumerate(teams)}

    sim_points = np.tile(standings["points"].values.astype(float), (n_sims, 1))
    sim_gd = np.tile(standings["goal_diff"].values.astype(float), (n_sims, 1))

    skipped = []
    for match in schedule:
        home, away = match["home_team"], match["away_team"]
        if home not in team_index or away not in team_index or home not in model_teams or away not in model_teams:
            skipped.append((home, away))
            continue

        lambda_home, lambda_away = adm.expected_goals(model, home, away)
        hg = rng.poisson(lambda_home, size=n_sims)
        ag = rng.poisson(lambda_away, size=n_sims)

        home_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        away_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

        hi, ai = team_index[home], team_index[away]
        sim_points[:, hi] += home_pts
        sim_points[:, ai] += away_pts
        sim_gd[:, hi] += (hg - ag)
        sim_gd[:, ai] += (ag - hg)

    if skipped:
        print(f"Skipped {len(skipped)} scheduled matches (team missing from standings/model): {skipped[:5]}")

    # Rank each simulated season: points desc, then goal diff desc.
    playoff_counts = np.zeros(len(teams))
    points_sum = np.zeros(len(teams))
    for s in range(n_sims):
        order = np.lexsort((-sim_gd[s], -sim_points[s]))
        playoff_counts[order[:PLAYOFF_SPOTS]] += 1
        points_sum += sim_points[s]

    results = pd.DataFrame({
        "team": teams,
        "current_points": standings["points"].values,
        "current_gd": standings["goal_diff"].values,
        "playoff_pct": (playoff_counts / n_sims) * 100,
        "projected_final_points": points_sum / n_sims,
    })
    return results.sort_values("playoff_pct", ascending=False).reset_index(drop=True)


def main(season: int, start: str, end: str):
    standings = current_standings(season)
    schedule = remaining_schedule(start, end)

    matches = pd.read_csv("data/nwsl_matches.csv")
    ad_matches = matches[matches["season"].isin(ATTACK_DEFENSE_SEASONS)]
    model, model_teams = adm.fit(ad_matches)

    print(f"{len(schedule)} remaining matches, {N_SIMULATIONS} simulations\n")
    results = simulate(standings, schedule, model, model_teams)
    results.to_csv("data/playoff_probabilities.csv", index=False)

    print(f"{'Team':26s} {'Pts':>5s} {'GD':>5s}  {'Playoff%':>9s}  {'Proj. Pts':>10s}")
    for r in results.itertuples():
        print(f"{r.team:26s} {r.current_points:5.0f} {r.current_gd:5.0f}  "
              f"{r.playoff_pct:8.1f}%  {r.projected_final_points:10.1f}")


if __name__ == "__main__":
    main(season=2026, start="20260729", end="20261101")
