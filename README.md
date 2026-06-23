# Baseball ELO Ratings

An interactive ELO rating system for MLB batters and pitchers, built on pitch-by-pitch
Statcast data. Every plate appearance is a zero-sum contest: the batter and pitcher
exchange rating points based on the outcome, weighted by how much it mattered.

## Features

- **wOBA-weighted outcomes** — a home run moves ratings far more than a walk, using each
  PA's wOBA value rather than a simple on-base/out binary.
- **Park factors** — computed per season from the data itself (home/road method), so a hit
  in Coors counts less than one in a pitcher's park.
- **Dynamic K-factor** — rating updates scale with leverage (win-probability impact) and a
  player's career length (new players move faster), with no manual tuning.
- **Self-calibrating** — each pool is recentered so batters and pitchers both average 1500
  on a plate-appearance-weighted basis, regardless of the league's run environment.
- **Single-season or career** — ratings reset each year, or carry across seasons.
- **Leaderboards** — sortable by End / Avg / Peak / Worst ELO and Range, filterable by team.
- **Cross-year matchup predictor** — pit a 2016 hitter against a 2026 pitcher in any
  season's run environment and see the expected wOBA.
- **Per-PA rating timelines** — chart any player's rating across a season, with off-days
  compressed and each plate appearance hoverable.
- **ELO+ / ELO−** — ratings expressed on a 100-scale (like wRC+ / FIP−), with a validation
  scatter against actual park-adjusted wOBA and the correlation R².
- **Strength of schedule** — who faced the toughest/weakest opponents (a small effect over a
  full season, since you never face your own staff).
- **Clutch & leverage** — high-leverage PAs (late & close) are scored against ELO-expected
  wOBA, crediting production against the elite relievers who pitch those spots. Quantifies
  that hitters face ~70 ELO tougher arms in the clutch.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first load of an uncached season downloads Statcast data (a few minutes) and caches it
locally under `cache/`. Subsequent loads are instant.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — leaderboards, matchup predictor, charts |
| `elo.py` | ELO engine, park factors, leaderboard builder |
| `data_fetch.py` | Statcast download, caching, player-name resolution |

## Data

Statcast data via [pybaseball](https://github.com/jldbc/pybaseball). Coverage begins 2015.

## Roadmap — two directions the scaffolding already supports

This is "Iteration 2" (wOBA + park factors + leverage). The architecture is built so the
next two extensions are mostly drop-in:

### Iteration 3 — expected stats (xwOBA), to strip luck

The hook is already in place: `data_fetch.py` fetches `estimated_woba_using_speedangle`
(Statcast's xwOBA) in `KEEP_COLS`. The change is to use it as the outcome in `run_elo`
instead of `woba_value` (or blend them). xwOBA removes the noise of where balls happened to
land, so the pitcher ratings become defense/luck-independent — the proper apples-to-apples
match for FIP−. Statcast-era only (2015+), since it needs exit velocity and launch angle.

### Historical — pre-2015 via Retrosheet

The ELO engine in `elo.py` is **source-agnostic**: it only needs a DataFrame with
`batter`, `pitcher`, `game_date`, `season`, `woba_value`, `on_base`, `home_team`,
`away_team`, `inning`, `bat_score`, `fld_score`. [Retrosheet](https://www.retrosheet.org)
has play-by-play events back to ~1914 (complete from ~1974). To wire it up: add a Retrosheet
loader alongside `_load_season_pa` (parse event files with the
[Chadwick](https://github.com/chadwickbureau/chadwick) tools / `pychadwick`), map events to
`woba_value` using [year-specific wOBA weights](https://www.fangraphs.com/guts.aspx), and the
rest of the app works unchanged. Iterations 1–2 (OBP, wOBA, park factors, leverage) all run
on Retrosheet; only Iteration 3 (xwOBA) is Statcast-only.
