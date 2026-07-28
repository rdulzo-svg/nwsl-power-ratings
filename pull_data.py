"""Pull historical NWSL match results from ESPN's public site API."""
import time
import requests
import pandas as pd

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.nwsl/scoreboard"

# NWSL regular seasons run roughly March-November.
SEASONS = range(2019, 2027)


def pull_season(year: int) -> list[dict]:
    params = {
        "dates": f"{year}0101-{year}1231",
        "limit": 1000,
    }
    resp = requests.get(BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        status = comp["status"]["type"]
        if not status.get("completed"):
            continue

        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")

        rows.append({
            "date": event["date"][:10],
            "season": year,
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": int(home["score"]),
            "away_score": int(away["score"]),
            "venue": comp.get("venue", {}).get("fullName", ""),
        })
    return rows


def main():
    all_rows = []
    for year in SEASONS:
        rows = pull_season(year)
        print(f"{year}: {len(rows)} completed matches")
        all_rows.extend(rows)
        time.sleep(1)

    df = pd.DataFrame(all_rows)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv("data/nwsl_matches.csv", index=False)
    print(f"\nTotal: {len(df)} matches saved to data/nwsl_matches.csv")


if __name__ == "__main__":
    main()
