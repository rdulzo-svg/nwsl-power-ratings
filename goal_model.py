"""Convert Elo's win-probability signal into real win/draw/loss splits
using a Poisson goal-scoring model, instead of a single constant draw
rate. A flat draw rate applied to every match regardless of how
mismatched the teams are doesn't hold up — mismatched games really do
produce fewer draws than close ones, and we have real scorelines to
prove it directly rather than assuming it.

Method: fit expected goals (lambda) for the home and away side as a
linear function of the Elo-implied win probability, using historical
matches. Then, for any matchup, treat home and away goals as
independent Poisson variables and sum the joint scoreline grid to get
P(home win), P(draw), P(away win). This is the standard, well-
established approach used by real football prediction models
(a simplified version of what sites like FiveThirtyEight's soccer
model do), not something invented for this project.
"""
import numpy as np
from scipy.stats import poisson

MAX_GOALS = 8  # truncate the scoreline grid; tail beyond this is negligible


def fit_goal_model(preds):
    """Fit lambda_home and lambda_away as linear functions of
    predicted_home_prob. Returns (home_coef, away_coef), each a
    (slope, intercept) pair."""
    x = preds["predicted_home_prob"].values
    home_coef = np.polyfit(x, preds["home_score"].values, deg=1)
    away_coef = np.polyfit(x, preds["away_score"].values, deg=1)
    return home_coef, away_coef


def expected_goals(predicted_home_prob: float, home_coef, away_coef) -> tuple[float, float]:
    lambda_home = np.polyval(home_coef, predicted_home_prob)
    lambda_away = np.polyval(away_coef, predicted_home_prob)
    # Poisson means can't be negative or zero; floor defensively.
    return max(lambda_home, 0.05), max(lambda_away, 0.05)


def win_draw_loss(lambda_home: float, lambda_away: float) -> tuple[float, float, float]:
    home_pmf = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_home)
    away_pmf = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_away)
    grid = np.outer(home_pmf, away_pmf)  # grid[i, j] = P(home scores i, away scores j)

    p_draw = np.trace(grid)
    p_home = np.sum(np.tril(grid, k=-1))  # i > j
    p_away = np.sum(np.triu(grid, k=1))   # i < j

    total = p_home + p_draw + p_away  # < 1 due to truncation; renormalize
    return p_home / total, p_draw / total, p_away / total


def predict(predicted_home_prob: float, home_coef, away_coef) -> tuple[float, float, float]:
    lh, la = expected_goals(predicted_home_prob, home_coef, away_coef)
    return win_draw_loss(lh, la)
