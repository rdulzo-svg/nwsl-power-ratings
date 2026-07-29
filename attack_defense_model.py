"""Team-specific attack and defense ratings via a Poisson regression
on goals scored (the standard Maher/Dixon-Coles approach to football
scoring models), instead of a single blended strength number.

Each match contributes two rows: the home team's scoring performance
and the away team's scoring performance, each as a function of who
was doing the scoring (attack) and who they played against (defense),
plus a home-advantage term:

    log(E[goals]) = intercept + attack[scoring_team] + concede[opponent] + home * is_home

"concede[opponent]" is how many goals teams tend to score AGAINST
that opponent — so a strong defense has a very negative value there.
We flip the sign so "defense" is reported on the same "higher is
better" scale as attack.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _stack_matches(matches: pd.DataFrame) -> pd.DataFrame:
    home_rows = matches[["home_team", "away_team", "home_score"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_score": "goals"})
    home_rows["home"] = 1

    away_rows = matches[["away_team", "home_team", "away_score"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_score": "goals"})
    away_rows["home"] = 0

    return pd.concat([home_rows, away_rows], ignore_index=True)


def fit(matches: pd.DataFrame):
    stacked = _stack_matches(matches)
    model = smf.glm(
        formula="goals ~ C(team) + C(opponent) + home",
        data=stacked,
        family=sm.families.Poisson(),
    ).fit()
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return model, teams


def _team_effect(model, teams: list, prefix: str) -> dict:
    """Reads C(team)[T.Name] / C(opponent)[T.Name] coefficients,
    treating the (arbitrary) dropped reference category as 0."""
    effects = {}
    for team in teams:
        key = f"C({prefix})[T.{team}]"
        effects[team] = model.params.get(key, 0.0)
    return effects


def ratings(model, teams: list) -> pd.DataFrame:
    """Centered attack/defense ratings for display: 0 = league average,
    positive = better than average, on a log-goals scale."""
    raw_attack = _team_effect(model, teams, "team")
    raw_concede = _team_effect(model, teams, "opponent")

    mean_attack = np.mean(list(raw_attack.values()))
    mean_concede = np.mean(list(raw_concede.values()))

    rows = []
    for team in teams:
        rows.append({
            "team": team,
            "attack": raw_attack[team] - mean_attack,
            "defense": -(raw_concede[team] - mean_concede),
        })
    return pd.DataFrame(rows)


def expected_goals(model, home_team: str, away_team: str) -> tuple[float, float]:
    intercept = model.params.get("Intercept", 0.0)
    home_coef = model.params.get("home", 0.0)

    attack = _team_effect(model, [home_team, away_team], "team")
    concede = _team_effect(model, [home_team, away_team], "opponent")

    log_lambda_home = intercept + attack[home_team] + concede[away_team] + home_coef
    log_lambda_away = intercept + attack[away_team] + concede[home_team]

    return float(np.exp(log_lambda_home)), float(np.exp(log_lambda_away))
