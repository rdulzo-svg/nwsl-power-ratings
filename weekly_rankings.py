"""Generate a weekly power-rankings snapshot: current rating, change
over the last 7 days, and current-season record for every team.
"""
import pandas as pd

WEEK_CUTOFF_DAYS = 7


def build_rating_history(preds: pd.DataFrame) -> pd.DataFrame:
    home_side = preds[["date", "home_team", "post_rating_home"]].rename(
        columns={"home_team": "team", "post_rating_home": "rating"})
    away_side = preds[["date", "away_team", "post_rating_away"]].rename(
        columns={"away_team": "team", "post_rating_away": "rating"})
    history = pd.concat([home_side, away_side], ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    return history.sort_values("date")


def rating_as_of(history: pd.DataFrame, team: str, as_of) -> float:
    team_hist = history[(history["team"] == team) & (history["date"] <= as_of)]
    if team_hist.empty:
        return 1500.0
    return team_hist.iloc[-1]["rating"]


def season_record(matches: pd.DataFrame, team: str, season: int) -> str:
    season_matches = matches[matches["season"] == season]
    home = season_matches[season_matches["home_team"] == team]
    away = season_matches[season_matches["away_team"] == team]

    wins = ((home["home_score"] > home["away_score"]).sum()
             + (away["away_score"] > away["home_score"]).sum())
    losses = ((home["home_score"] < home["away_score"]).sum()
               + (away["away_score"] < away["home_score"]).sum())
    draws = ((home["home_score"] == home["away_score"]).sum()
              + (away["away_score"] == away["home_score"]).sum())
    return f"{wins}-{losses}-{draws}"


def main():
    preds = pd.read_csv("data/predictions.csv")
    matches = pd.read_csv("data/nwsl_matches.csv")
    history = build_rating_history(preds)

    as_of = history["date"].max()
    week_ago = as_of - pd.Timedelta(days=WEEK_CUTOFF_DAYS)
    current_season = int(matches["season"].max())

    teams = sorted(history["team"].unique())
    rows = []
    for team in teams:
        current = rating_as_of(history, team, as_of)
        prior = rating_as_of(history, team, week_ago)
        rows.append({
            "team": team,
            "rating": round(current, 1),
            "change": round(current - prior, 1),
            "record": season_record(matches, team, current_season),
        })

    rankings = pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)
    rankings.insert(0, "rank", range(1, len(rankings) + 1))

    rankings.to_csv("data/weekly_rankings.csv", index=False)

    print(f"As of {as_of.date()} (season {current_season})\n")
    print(f"{'#':>2}  {'Team':28s} {'Rating':>7s} {'Chg':>6s}  {'Record':>7s}")
    for r in rankings.itertuples():
        arrow = "+" if r.change > 0 else ("" if r.change == 0 else "")
        print(f"{r.rank:>2}  {r.team:28s} {r.rating:7.1f} {arrow}{r.change:5.1f}  {r.record:>7s}")


if __name__ == "__main__":
    main()
