"""
Retrosheet event-file loader — extends the system back before the Statcast era (pre-2015).

Retrosheet publishes play-by-play "event files" (one per home team per season). This module
downloads, parses, and converts them into the SAME plate-appearance schema that the Statcast
path produces, so the rest of the app (ELO engine, leaderboards, matchup predictor) works
unchanged on historical seasons.

Player IDs: Retrosheet uses string IDs (e.g. "troum001"). We map each to a stable synthetic
integer (>= 9_000_000, far from MLBAM IDs) via a persisted registry, so the engine's int-based
code is untouched and the same player keeps one ID across seasons (career mode works within the
Retrosheet era).

MVP scope: outcomes, wOBA, on-base, inning. Running score (bat_score/fld_score) is set to 0 for
now — a follow-up will add run tracking to power historical leverage/clutch analysis.
"""
import os
import zipfile
import requests
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RAW_DIR = os.path.join(os.path.dirname(__file__), "retrosheet_raw")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

RETRO_ID_BASE = 9_000_000  # synthetic int IDs start here, clear of MLBAM IDs
_REGISTRY_PATH = os.path.join(CACHE_DIR, "retro_id_registry.parquet")

# Fixed wOBA-style weights, matching the scale of Statcast's `woba_value`, so historical and
# modern seasons sit on the same outcome scale. League wOBA still varies by era naturally,
# because outcome *frequencies* differ.
WOBA_WEIGHTS = {
    "single": 0.90, "double": 1.25, "triple": 1.62, "home_run": 2.00,
    "walk": 0.70, "hit_by_pitch": 0.70,
}
ON_BASE = {"single", "double", "triple", "home_run", "walk", "hit_by_pitch"}

# Event-code prefixes that are NOT plate appearances (the batter re-appears with the real result)
_SKIP_PREFIXES = ("NP", "SB", "CS", "PO", "WP", "PB", "BK", "DI", "OA", "FLE")

# Lazily-loaded {retro_id: (first_year, last_year)} for disambiguating shared names
# (e.g. Ken Griffey Sr 1973–1991 vs Jr 1989–2010).
_RETRO_SPANS: dict | None = None


def _retro_year_spans() -> dict:
    global _RETRO_SPANS
    if _RETRO_SPANS is None:
        _RETRO_SPANS = {}
        try:
            from pybaseball import chadwick_register
            reg = chadwick_register()
            for r in reg.itertuples():
                rid = r.key_retro
                if isinstance(rid, str) and rid and pd.notna(r.mlb_played_first):
                    _RETRO_SPANS[rid] = (int(r.mlb_played_first), int(r.mlb_played_last))
        except Exception:
            _RETRO_SPANS = {}
    return _RETRO_SPANS


def _classify(event: str) -> str | None:
    """Map a Retrosheet event code to an outcome category, or None if it isn't a PA."""
    code = event.split(".")[0].split("/")[0].split("+")[0].strip()
    if not code or code.startswith(_SKIP_PREFIXES):
        return None
    if code.startswith("HP"):
        return "hit_by_pitch"
    c = code[0]
    if c == "H":            # H or HR
        return "home_run"
    if c == "I" or code.startswith("IW"):
        return "walk"       # intentional walk
    if c == "W":
        return "walk"
    if c == "S":            # SB already skipped above
        return "single"
    if c == "D":            # DI already skipped above
        return "double"
    if c == "T":
        return "triple"
    if c == "K":
        return "strikeout"
    return "field_out"      # numeric outs, errors, fielder's choice, etc.


def _download_season(year: int) -> str:
    zip_path = os.path.join(RAW_DIR, f"{year}eve.zip")
    if not os.path.exists(zip_path):
        url = f"https://www.retrosheet.org/events/{year}eve.zip"
        # Use requests (certifi CA bundle) — the macOS python.org build lacks system
        # root certs, so urllib's HTTPS verification fails.
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"Retrosheet has no event file for {year} "
                f"(coverage is complete from 1974, partial earlier)."
            )
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
    return zip_path


def _load_registry() -> dict[str, int]:
    if os.path.exists(_REGISTRY_PATH):
        reg = pd.read_parquet(_REGISTRY_PATH)
        return dict(zip(reg["retro_id"], reg["syn_id"].astype(int)))
    return {}


def _save_registry(reg: dict[str, int]) -> None:
    pd.DataFrame({"retro_id": list(reg), "syn_id": list(reg.values())}).to_parquet(_REGISTRY_PATH)


def parse_season(year: int) -> tuple[pd.DataFrame, dict[int, str]]:
    """Return (PA DataFrame in the app's schema, {syn_id: name})."""
    cache_path = os.path.join(CACHE_DIR, f"retro_{year}.parquet")
    names_path = os.path.join(CACHE_DIR, f"retro_names_{year}.parquet")
    if os.path.exists(cache_path) and os.path.exists(names_path):
        ndf = pd.read_parquet(names_path)
        return pd.read_parquet(cache_path), dict(zip(ndf["syn_id"].astype(int), ndf["name"]))

    zip_path = _download_season(year)
    registry = _load_registry()
    retro_names: dict[str, str] = {}

    def syn(retro_id: str) -> int:
        if retro_id not in registry:
            registry[retro_id] = RETRO_ID_BASE + len(registry)
        return registry[retro_id]

    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        event_files = [n for n in zf.namelist() if n.upper().endswith((".EVN", ".EVA"))]
        for fname in event_files:
            text = zf.read(fname).decode("latin-1")
            _parse_file(text, year, syn, retro_names, rows)

    _save_registry(registry)

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)
    df["woba_value"] = df["events"].map(WOBA_WEIGHTS).fillna(0.0)
    df["on_base"] = df["events"].isin(ON_BASE).astype(int)
    df["estimated_woba_using_speedangle"] = pd.NA   # no Statcast xwOBA pre-2015
    df["delta_home_win_exp"] = 0.0                  # leverage deferred (no score tracking yet)
    df["bat_score"] = 0
    df["fld_score"] = 0
    df["season"] = year

    present = set(df["batter"]) | set(df["pitcher"])
    inv = {syn_id: rid for rid, syn_id in registry.items()}  # syn_id -> retro_id
    names = {syn_id: retro_names.get(inv[syn_id], str(syn_id)) for syn_id in present}

    # Disambiguate shared names by active-year span (Griffey Sr vs Jr), falling
    # back to the Retrosheet id if the span is unknown.
    from collections import Counter
    dup_names = {nm for nm, c in Counter(names.values()).items() if c > 1}
    if dup_names:
        spans = _retro_year_spans()
        for syn_id, nm in list(names.items()):
            if nm in dup_names:
                rid = inv[syn_id]
                span = spans.get(rid)
                names[syn_id] = f"{nm} ({span[0]}–{span[1]})" if span else f"{nm} [{rid}]"

    df.to_parquet(cache_path)
    pd.DataFrame({"syn_id": list(names), "name": list(names.values())}).to_parquet(names_path)
    return df, names


def _parse_file(text: str, year: int, syn, retro_names: dict, rows: list) -> None:
    visteam = hometeam = game_date = game_id = None
    cur_pitcher = {0: None, 1: None}
    ab_num = 0

    for line in text.splitlines():
        parts = line.rstrip("\n").split(",")
        rec = parts[0]

        if rec == "id":
            game_id = parts[1]
            cur_pitcher = {0: None, 1: None}
            ab_num = 0
        elif rec == "info":
            if parts[1] == "visteam":
                visteam = parts[2]
            elif parts[1] == "hometeam":
                hometeam = parts[2]
            elif parts[1] == "date":
                game_date = parts[2].replace("/", "-")
        elif rec in ("start", "sub"):
            pid, name, team, _order, pos = parts[1], parts[2].strip('"'), int(parts[3]), parts[4], parts[5]
            retro_names[pid] = name
            if pos == "1":
                cur_pitcher[team] = pid
        elif rec == "play":
            # play,inning,half,batter,count,pitches,event
            inning, half = int(parts[1]), int(parts[2])
            batter, event = parts[3], parts[6]
            cat = _classify(event)
            if cat is None:
                continue
            pitcher = cur_pitcher[1 - half]
            if pitcher is None:
                continue
            ab_num += 1
            rows.append({
                "game_date": game_date,
                "game_pk": game_id,
                "at_bat_number": ab_num,
                "batter": syn(batter),
                "pitcher": syn(pitcher),
                "events": cat,
                "home_team": hometeam,
                "away_team": visteam,
                "inning_topbot": "Top" if half == 0 else "Bot",
                "inning": inning,
            })
