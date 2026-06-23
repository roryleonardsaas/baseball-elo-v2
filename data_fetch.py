import os
import pandas as pd
from pybaseball import statcast, playerid_reverse_lookup

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Statcast coverage begins 2015; earlier seasons route through Retrosheet.
STATCAST_START = 2015

ON_BASE_EVENTS = {
    "single", "double", "triple", "home_run",
    "walk", "hit_by_pitch", "catcher_interf",
}

PLATE_APPEARANCE_EVENTS = ON_BASE_EVENTS | {
    "strikeout", "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "field_error", "fielders_choice",
    "fielders_choice_out", "strikeout_double_play", "other_out",
    "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play",
}

# Columns to keep — includes fields needed for iterations 2 and 3
KEEP_COLS = [
    "game_date", "game_pk", "at_bat_number",
    "batter", "pitcher",
    "events",
    "home_team", "away_team", "inning_topbot",
    "inning", "bat_score", "fld_score",    # pre-PA game state for leverage (late & close)
    "woba_value",                          # iter 2: actual wOBA weight
    "estimated_woba_using_speedangle",     # iter 3: xwOBA
    "delta_home_win_exp",                  # leverage proxy
]


def _statcast_with_retry(start: str, end: str, max_attempts: int = 5) -> pd.DataFrame:
    import time
    for attempt in range(max_attempts):
        try:
            return statcast(start_dt=start, end_dt=end)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"Download failed ({e}), retrying in {wait}s… (attempt {attempt + 1}/{max_attempts})")
            time.sleep(wait)


def _load_season_pa(year: int) -> pd.DataFrame:
    """Load (or download + cache) one season's plate appearances, without names."""
    pa_cache = os.path.join(CACHE_DIR, f"statcast_{year}.parquet")

    if os.path.exists(pa_cache):
        return pd.read_parquet(pa_cache)

    from datetime import date
    end = min(date(year, 11, 5), date.today()).strftime("%Y-%m-%d")
    raw = _statcast_with_retry(f"{year}-03-20", end)

    # Filter to completed plate appearances only
    df = raw[raw["events"].isin(PLATE_APPEARANCE_EVENTS)][KEEP_COLS].copy()

    # Deduplicate: one row per (game, at-bat). Statcast can return duplicate
    # rows for suspended/replayed games or overlapping date pulls.
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number"])

    df = df.dropna(subset=["batter", "pitcher", "events"])
    df["batter"] = df["batter"].astype(int)
    df["pitcher"] = df["pitcher"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)
    df["on_base"] = df["events"].isin(ON_BASE_EVENTS).astype(int)
    df["season"] = year
    df.to_parquet(pa_cache)
    return df


def fetch_seasons(years: list[int]) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Load one or more seasons, concatenated in strict chronological order.
    With multiple years this powers 'career' mode: ELO carries across seasons
    because run_elo processes the combined frame in date order without resets.
    """
    import retrosheet

    frames = []
    names: dict[int, str] = {}
    for y in sorted(years):
        if y < STATCAST_START:
            df_y, retro_names = retrosheet.parse_season(y)
            names.update(retro_names)
        else:
            df_y = _load_season_pa(y)
        frames.append(df_y)

    df = pd.concat(frames, ignore_index=True)
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year
    # game_pk is int (Statcast) or str game-id (Retrosheet); unify so sorting/dedup is safe
    df["game_pk"] = df["game_pk"].astype(str)
    df = df.sort_values(["game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)

    # Names: Retrosheet ids already resolved above; resolve remaining MLBAM (Statcast) ids
    statcast_ids = [i for i in (set(df["batter"]) | set(df["pitcher"]))
                    if i < retrosheet.RETRO_ID_BASE]
    if statcast_ids:
        names.update(_build_name_lookup(statcast_ids))
    return df, names


def fetch_season(year: int) -> tuple[pd.DataFrame, dict[int, str]]:
    return fetch_seasons([year])


def _build_name_lookup(player_ids: list[int]) -> dict[int, str]:
    cache_path = os.path.join(CACHE_DIR, "player_names.parquet")

    if os.path.exists(cache_path):
        existing = pd.read_parquet(cache_path)
        cached_ids = set(existing["key_mlbam"].astype(int).tolist())
        missing = [pid for pid in player_ids if pid not in cached_ids]
    else:
        existing = pd.DataFrame()
        missing = player_ids

    if missing:
        new_rows = playerid_reverse_lookup(missing, key_type="mlbam")
        combined = pd.concat([existing, new_rows], ignore_index=True) if not existing.empty else new_rows
        combined.to_parquet(cache_path)
    else:
        combined = existing

    lookup: dict[int, str] = {}
    for _, row in combined.iterrows():
        pid = int(row["key_mlbam"])
        name = f"{str(row['name_first']).title()} {str(row['name_last']).title()}"
        lookup[pid] = name
    return lookup
