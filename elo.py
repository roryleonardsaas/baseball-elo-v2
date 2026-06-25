import math
import os
import pickle
import hashlib
from collections import defaultdict
import pandas as pd

DEFAULT_RATING = 1500.0
# Replacement-level floor for "ELO Value": a volume-adjusted rating on the ELO scale.
# ELO Value = REPLACEMENT_ELO + (ELO − REPLACEMENT_ELO) × min(PA / full_workload, 1),
# so a full-workload player keeps their ELO and a part-timer is scaled toward this floor.
REPLACEMENT_ELO = 1360.0
# Times-through-the-order penalty: each time a pitcher cycles through the lineup,
# expected wOBA-against rises (fatigue + hitter familiarity). Raising the expected
# bar by this per turn means a hit given up deep in a start costs the pitcher less,
# and an out earned deep counts more — neutralizing the structural disadvantage.
# Data (2024): ~+0.036 the 2nd time, ~+0.082 the 3rd; 0.04/turn fits well.
TTO_PENALTY = 0.04
# Tuned for wOBA residuals. wOBA outcomes are leptokurtic (rare +1.5 HR jumps),
# so a smaller K than the on-base era keeps the spread in standard ELO territory
# (regulars ~1320-1900, elite ~1800) and keeps expected_woba reasonably calibrated.
BASE_K = 7.0
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Bump when history tuple format changes so stale caches are ignored.
# v9: times-through-the-order penalty added to the expected value.
_CACHE_VERSION = 9


def expected_woba(r_batter: float, r_pitcher: float, league_woba: float) -> float:
    """
    Expected wOBA for this matchup.
    p is the logistic win-probability (0.5 at equal ratings); expected wOBA
    scales linearly so that equal ratings give exactly league_woba, a dominant
    batter approaches 2*league_woba, and a dominated batter approaches 0.
    """
    p = 1 / (1 + 10 ** ((r_pitcher - r_batter) / 400))
    return 2 * league_woba * p


def elo_index(rating: float, league_woba: float, role: str) -> int:
    """
    Convert an ELO rating to a 100-scale index (like wRC+ / FIP-).
    Batter: expected wOBA vs a league-average pitcher, indexed to 100 (higher=better).
    Pitcher: expected wOBA allowed vs a league-average batter, indexed to 100
    (lower=better, comparable to ERA-/FIP-).
    """
    if role == "batter":
        ew = expected_woba(rating, DEFAULT_RATING, league_woba)
    else:
        ew = expected_woba(DEFAULT_RATING, rating, league_woba)
    return round(100 * ew / league_woba)


def compute_park_factors(df: pd.DataFrame, regression: float = 0.5, min_pa: int = 500) -> dict:
    """
    Per-(season, park) wOBA park factors via the home/road method.
    For each team's park: compare wOBA of all PAs there (home games) against
    wOBA of that team's road games. Because the same team plays both sets,
    this largely controls for team quality. PF > 1 = hitter-friendly park.

    Regressed toward 1.0 (default 50%) to damp single-season noise, and falls
    back to 1.0 for parks with too few PAs to estimate reliably.
    """
    park_factors: dict = {}
    for season, sdf in df.groupby("season"):
        for team in set(sdf["home_team"].unique()):
            home = sdf.loc[sdf["home_team"] == team, "woba_value"]
            road = sdf.loc[sdf["away_team"] == team, "woba_value"]
            if len(home) < min_pa or len(road) < min_pa or road.mean() == 0:
                pf = 1.0
            else:
                pf = home.mean() / road.mean()
                pf = 1.0 + (pf - 1.0) * regression
            park_factors[(int(season), team)] = pf
    return park_factors


def _career_factor(pa_count: int) -> float:
    """
    Multiplier on K based on how established a player is.
    New players (0 PA): 1.5x — ratings move fast to find their true level.
    Veterans (500+ PA): 1.0x — rating is well-established, smaller updates.
    Smooth exponential decay between the two.
    """
    return 1.0 + 0.5 * math.exp(-pa_count / 150)


def _leverage_factor(delta_wexp: float, mean_abs_wexp: float) -> float:
    """
    Multiplier on K based on how much the PA mattered win-probability-wise.
    Normalized so average leverage = 1.0. Capped at 2x so a walk-off
    can't dwarf an entire season's worth of other PAs.
    """
    if mean_abs_wexp == 0:
        return 1.0
    return min(abs(delta_wexp) / mean_abs_wexp, 2.0)


def run_elo(
    df: pd.DataFrame,
    league_woba: float = 0.320,
    avg_warmup_pa: int = 50,
    years: tuple = (),
):
    # ── Disk cache ─────────────────────────────────────────────────────────────
    key = f"v{_CACHE_VERSION}-{sorted(years)}-{league_woba}"
    cache_path = os.path.join(CACHE_DIR, f"elo_{hashlib.md5(key.encode()).hexdigest()}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Precompute mean |WPA| for leverage normalisation
    mean_abs_wexp = float(df["delta_home_win_exp"].abs().mean()) if "delta_home_win_exp" in df.columns else 1.0
    has_leverage = "delta_home_win_exp" in df.columns

    # Park factors per (season, park), computed from the data itself
    park_factors = compute_park_factors(df)

    # Variance-match the wOBA residuals to the old on-base scale so the rating
    # spread (and therefore peak/worst/range numbers) stays interpretable.
    woba_std = float(df["woba_value"].std()) or 1.0
    onbase_std = float(df["on_base"].std()) if "on_base" in df.columns else 0.4626
    woba_k_scale = onbase_std / woba_std

    batter_ratings: dict[int, float] = {}
    pitcher_ratings: dict[int, float] = {}
    batter_history: dict[int, list] = defaultdict(list)
    pitcher_history: dict[int, list] = defaultdict(list)

    batter_pa_count: dict[int, int] = defaultdict(int)
    pitcher_pa_count: dict[int, int] = defaultdict(int)
    batter_rating_sum: dict[int, float] = defaultdict(float)
    pitcher_rating_sum: dict[int, float] = defaultdict(float)
    batter_peak: dict[int, float] = {}
    pitcher_peak: dict[int, float] = {}
    batter_worst: dict[int, float] = {}
    pitcher_worst: dict[int, float] = {}
    game_bf: dict = defaultdict(int)  # (game, pitcher) → batters faced so far this game

    for row in df.itertuples(index=False):
        b_id = int(row.batter)
        p_id = int(row.pitcher)
        outcome = float(row.woba_value)  # wOBA value of the PA (0 to ~2.0)
        if outcome != outcome:           # NaN guard
            outcome = 0.0
        date = row.game_date

        r_b = batter_ratings.get(b_id, DEFAULT_RATING)
        r_p = pitcher_ratings.get(p_id, DEFAULT_RATING)

        # Times-through-the-order: raise the expected bar as the pitcher goes deeper.
        # Keyed per (game, pitcher), so a fresh reliever resets to the 1st time through.
        game_bf[(row.game_pk, p_id)] += 1
        times_through = (game_bf[(row.game_pk, p_id)] - 1) // 9
        tto_factor = 1 + TTO_PENALTY * times_through

        # Expected wOBA, park-adjusted and depth-adjusted: in a hitter's park (PF>1)
        # or deep in a start (tto_factor>1) the bar rises, so a given hit earns less
        # and a given out costs more — neutralizing park and fatigue effects.
        pf = park_factors.get((int(row.season), row.home_team), 1.0)
        e = expected_woba(r_b, r_p, league_woba) * pf * tto_factor

        # Dynamic K: leverage × career factor, computed independently per role.
        # woba_k_scale keeps the rating spread comparable to the on-base version.
        lev = _leverage_factor(row.delta_home_win_exp if has_leverage else mean_abs_wexp, mean_abs_wexp)
        k_b = BASE_K * woba_k_scale * _career_factor(batter_pa_count[b_id]) * lev
        k_p = BASE_K * woba_k_scale * _career_factor(pitcher_pa_count[p_id]) * lev

        b_delta = k_b * (outcome - e)
        p_delta = k_p * (outcome - e)

        batter_ratings[b_id] = r_b + b_delta
        pitcher_ratings[p_id] = r_p - p_delta

        # Peak / worst tracking
        if batter_ratings[b_id] > batter_peak.get(b_id, DEFAULT_RATING):
            batter_peak[b_id] = batter_ratings[b_id]
        if batter_ratings[b_id] < batter_worst.get(b_id, DEFAULT_RATING):
            batter_worst[b_id] = batter_ratings[b_id]
        if pitcher_ratings[p_id] > pitcher_peak.get(p_id, DEFAULT_RATING):
            pitcher_peak[p_id] = pitcher_ratings[p_id]
        if pitcher_ratings[p_id] < pitcher_worst.get(p_id, DEFAULT_RATING):
            pitcher_worst[p_id] = pitcher_ratings[p_id]

        batter_pa_count[b_id] += 1
        pitcher_pa_count[p_id] += 1

        if batter_pa_count[b_id] > avg_warmup_pa:
            batter_rating_sum[b_id] += batter_ratings[b_id]
        if pitcher_pa_count[p_id] > avg_warmup_pa:
            pitcher_rating_sum[p_id] += pitcher_ratings[p_id]

        event = row.events
        batter_history[b_id].append((
            batter_pa_count[b_id], date, round(batter_ratings[b_id], 1),
            event, p_id, round(b_delta, 2), round(r_p, 1),
        ))
        pitcher_history[p_id].append((
            pitcher_pa_count[p_id], date, round(pitcher_ratings[p_id], 1),
            event, b_id, round(-p_delta, 2), round(r_b, 1),
        ))

    full_season_pa = 400

    def _avg(rating_sum: dict, pa_count: dict, warmup: int) -> dict[int, float]:
        result = {}
        for pid, total in rating_sum.items():
            post_warmup = pa_count[pid] - warmup
            if post_warmup <= 0:
                result[pid] = DEFAULT_RATING
                continue
            raw_avg = total / post_warmup
            credibility = min(post_warmup / full_season_pa, 1.0)
            result[pid] = round(DEFAULT_RATING + (raw_avg - DEFAULT_RATING) * credibility, 1)
        return result

    batter_avg = _avg(batter_rating_sum, batter_pa_count, avg_warmup_pa)
    pitcher_avg = _avg(pitcher_rating_sum, pitcher_pa_count, avg_warmup_pa)

    # ── Recenter each pool to a PA-weighted mean of 1500 ──────────────────────
    # Dynamic K (different career factors for batter vs pitcher) makes the two
    # pools drift apart. A uniform shift per pool restores both to 1500 without
    # changing any within-pool ordering, and keeps the matchup predictor's
    # expected_score(1500, 1500) == league_obp identity true.
    def _pa_weighted_mean(ratings: dict, pa_count: dict) -> float:
        tw = sum(pa_count.get(p, 0) for p in ratings)
        if tw == 0:
            return DEFAULT_RATING
        return sum(ratings[p] * pa_count.get(p, 0) for p in ratings) / tw

    b_shift = DEFAULT_RATING - _pa_weighted_mean(batter_ratings, batter_pa_count)
    p_shift = DEFAULT_RATING - _pa_weighted_mean(pitcher_ratings, pitcher_pa_count)

    for d in (batter_ratings, batter_avg, batter_peak, batter_worst):
        for pid in d:
            d[pid] = round(d[pid] + b_shift, 1)
    for d in (pitcher_ratings, pitcher_avg, pitcher_peak, pitcher_worst):
        for pid in d:
            d[pid] = round(d[pid] + p_shift, 1)

    # History rating gets its own pool's shift; the stored opponent ELO is the
    # other pool's rating, so it gets the other pool's shift.
    for pid, hist in batter_history.items():
        batter_history[pid] = [
            (n, d, round(r + b_shift, 1), e, o, dl, round(oe + p_shift, 1))
            for (n, d, r, e, o, dl, oe) in hist
        ]
    for pid, hist in pitcher_history.items():
        pitcher_history[pid] = [
            (n, d, round(r + p_shift, 1), e, o, dl, round(oe + b_shift, 1))
            for (n, d, r, e, o, dl, oe) in hist
        ]

    result = (
        batter_ratings, pitcher_ratings,
        batter_avg, pitcher_avg,
        batter_peak, pitcher_peak,
        batter_worst, pitcher_worst,
        batter_history, pitcher_history,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def build_leaderboard(
    ratings: dict[int, float],
    avg_ratings: dict[int, float],
    peak_ratings: dict[int, float],
    worst_ratings: dict[int, float],
    display_names: dict[int, str],
    pa_counts: pd.Series,
    min_pa: int,
    sort_by: str = "End ELO",
    team_filter: set | None = None,
    player_teams: dict[int, str] | None = None,
    extra_columns: dict[str, dict[int, float]] | None = None,
) -> pd.DataFrame:
    extra_columns = extra_columns or {}
    rows = []
    for pid, rating in ratings.items():
        if pa_counts.get(pid, 0) < min_pa:
            continue
        team = (player_teams or {}).get(pid, "")
        if team_filter and team not in team_filter:
            continue
        row = {
            "Name": display_names.get(pid, f"ID:{pid}"),
            "Team": team,
            "End ELO": round(rating, 1),
            "Avg ELO": avg_ratings.get(pid, round(rating, 1)),
            "Peak ELO": round(peak_ratings.get(pid, rating), 1),
            "Worst ELO": round(worst_ratings.get(pid, rating), 1),
            "Range": round(peak_ratings.get(pid, rating) - worst_ratings.get(pid, rating), 1),
            "PA": int(pa_counts.get(pid, 0)),
        }
        for col_name, col_vals in extra_columns.items():
            row[col_name] = col_vals.get(pid)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(sort_by, ascending=False).reset_index(drop=True)
