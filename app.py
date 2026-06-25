import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter

from data_fetch import fetch_seasons
from elo import run_elo, build_leaderboard, expected_woba, compute_park_factors, elo_index

st.set_page_config(page_title="Baseball ELO", layout="wide")
st.title("Baseball ELO Ratings — Iteration 2: wOBA + Park Factors")


@st.cache_data(show_spinner=False)
def cached_fetch(years: tuple[int, ...]):
    return fetch_seasons(list(years))


@st.cache_data(show_spinner=False)
def cached_elo(years: tuple[int, ...], league_woba: float):
    df, _ = cached_fetch(years)
    return run_elo(df, league_woba=league_woba, years=years)


@st.cache_data(show_spinner=False)
def cached_actual_indices(years: tuple[int, ...]):
    """
    Each player's actual park-adjusted seasonal wOBA, indexed to 100.
    Batters: wOBA+ (higher=better). Pitchers: wOBA-against- (lower=better).
    A homemade wRC+ / FIP- analog built on the same wOBA the ELO uses.
    """
    df_y, _ = cached_fetch(years)
    lw = float(df_y["woba_value"].mean())
    pf = compute_park_factors(df_y)
    keys = list(zip(df_y["season"].astype(int), df_y["home_team"]))
    pf_arr = np.array([pf.get(k, 1.0) for k in keys])
    padj = df_y["woba_value"].to_numpy() / pf_arr
    tmp = pd.DataFrame({"batter": df_y["batter"].to_numpy(),
                        "pitcher": df_y["pitcher"].to_numpy(), "padj": padj})
    b_woba = tmp.groupby("batter")["padj"].mean()
    p_woba = tmp.groupby("pitcher")["padj"].mean()
    b_plus = {int(p): round(100 * w / lw) for p, w in b_woba.items()}
    p_minus = {int(p): round(100 * w / lw) for p, w in p_woba.items()}
    return b_plus, p_minus


@st.cache_data(show_spinner=False)
def cached_clutch(years: tuple[int, ...], league_woba: float):
    """
    Per-PA: actual wOBA, ELO-expected wOBA (park-adjusted), the residual, the
    opponent's ELO, and whether the PA was high-leverage (late & close:
    inning >= 7 and within 2 runs). The residual vs expectation is the key —
    it credits production against the tough arms that show up in clutch spots.
    """
    df_y, _ = cached_fetch(years)
    res = cached_elo(years, league_woba)
    br, pr = res[0], res[1]
    pf = compute_park_factors(df_y)
    keys = list(zip(df_y["season"].astype(int), df_y["home_team"]))
    pf_arr = np.array([pf.get(k, 1.0) for k in keys])

    b_elo = df_y["batter"].map(br).fillna(1500).to_numpy(dtype=float)
    p_elo = df_y["pitcher"].map(pr).fillna(1500).to_numpy(dtype=float)
    p_win = 1 / (1 + 10 ** ((p_elo - b_elo) / 400))
    exp = 2 * league_woba * p_win * pf_arr
    actual = df_y["woba_value"].to_numpy(dtype=float)

    inning = pd.to_numeric(df_y.get("inning"), errors="coerce").fillna(1).to_numpy()
    diff = (pd.to_numeric(df_y.get("bat_score"), errors="coerce").fillna(0)
            - pd.to_numeric(df_y.get("fld_score"), errors="coerce").fillna(0)).abs().to_numpy()
    high = (inning >= 7) & (diff <= 2)

    return pd.DataFrame({
        "batter": df_y["batter"].to_numpy(), "pitcher": df_y["pitcher"].to_numpy(),
        "woba": actual, "exp": exp, "resid": actual - exp,
        "b_elo": b_elo, "p_elo": p_elo, "high": high,
    })


@st.cache_data(show_spinner=False)
def season_index_corr(year: int, min_pa_corr: int):
    """Per-season R² between model ELO index and actual park-adjusted wOBA index."""
    df_y, _, lw, res = season_bundle(year)
    br, pr, bavg, pavg = res[0], res[1], res[2], res[3]
    bpa = df_y.groupby("batter").size()
    ppa = df_y.groupby("pitcher").size()
    b_actual, p_actual = cached_actual_indices((year,))

    def _r2(ratings, avg, pa, actual, role):
        pairs = []
        for pid in ratings:
            if pa.get(pid, 0) < min_pa_corr or actual.get(pid) is None:
                continue
            model = elo_index(avg.get(pid, ratings[pid]), lw, role)
            pairs.append((actual[pid], model))
        if len(pairs) < 3:
            return None, len(pairs)
        a, m = zip(*pairs)
        return float(np.corrcoef(a, m)[0, 1]) ** 2, len(pairs)

    b_r2, b_n = _r2(br, bavg, bpa, b_actual, "batter")
    p_r2, p_n = _r2(pr, pavg, ppa, p_actual, "pitcher")
    return b_r2, b_n, p_r2, p_n


def available_cached_years() -> list[int]:
    ys = []
    cache = os.path.join(os.path.dirname(__file__), "cache")
    for pattern in ("statcast_*.parquet", "retro_*.parquet"):
        for f in glob.glob(os.path.join(cache, pattern)):
            base = os.path.basename(f)
            if base.startswith("retro_names"):
                continue
            try:
                y = int(base.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue  # e.g. retro_id_registry.parquet
            if 1900 <= y <= 2100:
                ys.append(y)
    return sorted(set(ys))


@st.cache_data(show_spinner=False)
def season_bundle(year: int):
    """Single-season ratings + names + league wOBA for the matchup predictor."""
    df_y, names_y = cached_fetch((year,))
    lw = round(float(df_y["woba_value"].mean()), 4)
    res = cached_elo((year,), lw)
    return df_y, names_y, lw, res


@st.cache_data(show_spinner=False)
def season_player_labels(year: int, role: str, min_pa_mp: int = 50) -> dict[str, int]:
    """
    Cheap dropdown source: player label -> id for one season+role, using only the
    parsed data (NO ELO run). Lets the matchup picker populate fast; ELO is computed
    only when a matchup is actually requested.
    """
    df_y, names_y = cached_fetch((year,))
    id_col = "batter" if role == "batter" else "pitcher"
    pa = df_y.groupby(id_col).size()
    counts = Counter(names_y.values())
    opts: dict[str, int] = {}
    for pid, n in pa.items():
        if n < min_pa_mp:
            continue
        nm = names_y.get(pid, f"ID:{pid}")
        label = f"{nm} ({pid})" if counts.get(nm, 1) > 1 else nm
        opts[label] = int(pid)
    return opts


@st.cache_data(show_spinner=False)
def season_league_woba(year: int) -> float:
    df_y, _ = cached_fetch((year,))
    return round(float(df_y["woba_value"].mean()), 4)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    from datetime import date as _date
    current_year = _date.today().year
    # 2015+ = Statcast; 1915–2014 = Retrosheet (complete from 1974, partial earlier)
    all_years = list(range(current_year, 1914, -1))

    scope = st.radio(
        "Scope", ["Single Season", "Career"], index=0,
        help="Single Season: ratings reset to 1500 each year. Career: ratings carry across seasons.",
    )
    if scope == "Single Season":
        year = st.selectbox("Season", all_years, index=0)
        selected_years = (year,)
        scope_label = str(year)
    else:
        years_asc = sorted(all_years)
        c1, c2 = st.columns(2)
        start_year = c1.selectbox("From", years_asc, index=max(0, len(years_asc) - 3))
        end_year = c2.selectbox("To", all_years, index=0)
        if end_year < start_year:
            start_year, end_year = end_year, start_year
        selected_years = tuple(range(start_year, end_year + 1))
        scope_label = f"{start_year}–{end_year} career"
        st.caption("Uncached seasons download ~a few min each on first use.")

    min_pa = st.slider("Min PA for leaderboard", 10, 1000, 100, 10)
    sort_by = st.radio("Sort leaderboard by", ["Value", "End ELO", "Avg ELO", "Peak ELO", "Worst ELO", "Range"],
                       help="Value: rate × playing time (credits innings/PA — a workhorse beats an elite low-volume arm). End ELO: current rating. Avg ELO: sustained rate. Peak/Worst: hottest/lowest point. Range: streakiness.")
    innings_weight = st.slider(
        "How much to reward innings / durability", 0, 100, 50, 5,
        help="Higher = volume (innings/PA) counts more in Value; lower = pure rate matters more.",
    )
    # Map 0–100 to a replacement level: more innings weight → lower replacement.
    replacement = 1420 - (innings_weight / 100) * (1420 - 1300)

# ── Load data ─────────────────────────────────────────────────────────────────
_src = ("Retrosheet" if all(y < 2015 for y in selected_years)
        else "Statcast" if all(y >= 2015 for y in selected_years)
        else "Statcast + Retrosheet")
with st.spinner(f"Loading {scope_label} {_src} data (first run: a few minutes per season)…"):
    df, names = cached_fetch(selected_years)

league_woba = round(float(df["woba_value"].mean()), 4)

# ── Team lookup: most recent team for each batter and pitcher ─────────────────
# inning_topbot: 'Top' = away bats, 'Bot' = home bats
df_sorted = df.sort_values(["game_date", "game_pk", "at_bat_number"])
df_sorted["batter_team"] = np.where(df_sorted["inning_topbot"] == "Bot",
                                    df_sorted["home_team"], df_sorted["away_team"])
df_sorted["pitcher_team"] = np.where(df_sorted["inning_topbot"] == "Bot",
                                     df_sorted["away_team"], df_sorted["home_team"])
batter_teams: dict[int, str] = df_sorted.groupby("batter")["batter_team"].last().to_dict()
pitcher_teams: dict[int, str] = df_sorted.groupby("pitcher")["pitcher_team"].last().to_dict()
all_teams = sorted({t for t in set(batter_teams.values()) | set(pitcher_teams.values()) if t})

st.sidebar.markdown("---")
st.sidebar.subheader("Team Filter")
team_filter_raw = st.sidebar.multiselect("Filter leaderboards by team", all_teams,
                                          label_visibility="collapsed")
team_filter = set(team_filter_raw) if team_filter_raw else None

st.success(f"{scope_label} | {len(df):,} plate appearances | {len(names):,} players | League wOBA: {league_woba:.3f}")

# ── Run ELO ───────────────────────────────────────────────────────────────────
(batter_ratings, pitcher_ratings,
 batter_avg, pitcher_avg,
 batter_peak, pitcher_peak,
 batter_worst, pitcher_worst,
 batter_history, pitcher_history) = cached_elo(selected_years, league_woba)

# ── Build display names (append ID only if two players share a name) ──────────
# Built after run_elo so we can cover any IDs missing from the names lookup.
name_counts = Counter(names.values())
def _display(pid: int) -> str:
    name = names.get(pid, f"ID:{pid}")
    return f"{name} ({pid})" if name_counts.get(name, 1) > 1 else name

display_names: dict[int, str] = {
    pid: _display(pid)
    for pid in set(names) | set(batter_ratings) | set(pitcher_ratings)
}

batter_pa = df.groupby("batter").size()
pitcher_pa = df.groupby("pitcher").size()

# Model index (ELO+/ELO-) from Avg ELO, and actual park-adjusted wOBA index
batter_eloplus = {pid: elo_index(batter_avg.get(pid, batter_ratings[pid]), league_woba, "batter")
                  for pid in batter_ratings}
pitcher_elominus = {pid: elo_index(pitcher_avg.get(pid, pitcher_ratings[pid]), league_woba, "pitcher")
                    for pid in pitcher_ratings}
batter_wobaplus, pitcher_wobaminus = cached_actual_indices(selected_years)

# Average opponent strength faced (mean End ELO of the opponents in each PA;
# End ELO is recentered to a PA-weighted 1500, so the measure is well-centered)
bat_strength = batter_ratings
pit_strength = pitcher_ratings
_opp = df[["batter", "pitcher"]].copy()
_opp["opp_for_batter"] = _opp["pitcher"].map(pit_strength)
_opp["opp_for_pitcher"] = _opp["batter"].map(bat_strength)
batter_opp_elo = _opp.groupby("batter")["opp_for_batter"].mean().round(0).to_dict()
pitcher_opp_elo = _opp.groupby("pitcher")["opp_for_pitcher"].mean().round(0).to_dict()

leaderboard = build_leaderboard(
    batter_ratings, batter_avg, batter_peak, batter_worst, display_names, batter_pa,
    min_pa, sort_by, team_filter, batter_teams,
    extra_columns={"ELO+": batter_eloplus, "wOBA+": batter_wobaplus, "Opp ELO": batter_opp_elo},
    replacement=replacement,
)
pitcher_board = build_leaderboard(
    pitcher_ratings, pitcher_avg, pitcher_peak, pitcher_worst, display_names, pitcher_pa,
    min_pa, sort_by, team_filter, pitcher_teams,
    extra_columns={"ELO−": pitcher_elominus, "wOBA-agst−": pitcher_wobaminus, "Opp ELO": pitcher_opp_elo},
    replacement=replacement,
)

# ── Leaderboards ──────────────────────────────────────────────────────────────
st.subheader("Batters")
st.caption("ELO+ = model talent index (from Avg ELO). wOBA+ = actual park-adjusted "
           "seasonal wOBA, indexed to 100 (≈ wRC+). Both: 100 = average, higher = better.")
st.dataframe(leaderboard, use_container_width=True, hide_index=True, height=600)

st.subheader("Pitchers")
st.caption("ELO− = model talent index (from Avg ELO). wOBA-agst− = actual park-adjusted "
           "wOBA allowed, indexed to 100 (≈ ERA−/FIP−). Both: 100 = average, lower = better.")
st.dataframe(pitcher_board, use_container_width=True, hide_index=True, height=600)

# ── Validation: model vs raw, and strength of schedule ────────────────────────
with st.expander("ELO vs raw production — and who faced the toughest competition"):
    st.markdown(
        "**Reading the scatter.** Each dot is a player. The horizontal axis is what they "
        "actually produced — park-adjusted wOBA, indexed so 100 = league average. The vertical "
        "axis is their ELO rating on that same scale. On the diagonal = ELO and raw production "
        "agree. Off the diagonal = the value raw rate stats miss: **leverage** (production in "
        "high win-probability spots), **timing/path** (when in the season it happened), and a "
        "smaller piece, **strength of schedule** (who they faced). The tables below isolate that "
        "last piece — who faced the toughest and weakest competition."
    )
    v_tab_b, v_tab_p = st.tabs(["Batters", "Pitchers"])

    def _scatter(board, model_col, actual_col):
        sub = board.dropna(subset=[model_col, actual_col])
        if len(sub) < 3:
            st.info("Not enough qualified players at this Min PA threshold.")
            return
        r = float(np.corrcoef(sub[model_col], sub[actual_col])[0, 1])
        slope, intercept = np.polyfit(sub[actual_col], sub[model_col], 1)
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=sub[actual_col], y=sub[model_col], mode="markers",
            text=sub["Name"], hovertemplate="%{text}<br>actual %{x}<br>model %{y}<extra></extra>",
            marker=dict(size=6, opacity=0.6),
        ))
        lo = min(sub[model_col].min(), sub[actual_col].min())
        hi = max(sub[model_col].max(), sub[actual_col].max())
        fig_v.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                   line=dict(dash="dash", color="gray"), name="y = x"))
        fig_v.add_trace(go.Scatter(x=[lo, hi], y=[slope * lo + intercept, slope * hi + intercept],
                                   mode="lines", line=dict(color="firebrick"), name="best fit"))
        fig_v.update_layout(
            title=f"r = {r:.3f}   R² = {r**2:.3f}   (n = {len(sub)})",
            xaxis_title=f"Actual {actual_col}", yaxis_title=f"Model {model_col}",
            showlegend=False,
        )
        st.plotly_chart(fig_v, use_container_width=True, key=f"scatter_{model_col}")

    def _schedule_analysis(board, model_col, actual_col, lower_better):
        sub = board.dropna(subset=[model_col, actual_col, "Opp ELO"]).copy()
        if len(sub) < 5:
            return
        # gap = how much ELO credits a player vs their raw line (positive = underrated)
        sub["gap"] = ((sub[actual_col] - sub[model_col]) if lower_better
                      else (sub[model_col] - sub[actual_col]))
        lo, hi = sub["Opp ELO"].min(), sub["Opp ELO"].max()
        r_opp = float(np.corrcoef(sub["Opp ELO"], sub["gap"])[0, 1])
        st.markdown(
            f"**Strength of schedule is real but small.** Average opponent ELO spans only "
            f"**{lo:.0f}–{hi:.0f}** here — over a full season everyone faces near-league-average "
            f"competition (you never face your own staff, and the rest of the slate is fairly "
            f"balanced). It correlates only **r = {r_opp:.2f}** with the ELO-vs-raw gap, so "
            f"opponents explain just a sliver of it; most of the gap is **leverage and timing**. "
            f"Still, at the margins it nudges these players:"
        )
        cols = ["Name", actual_col, model_col, "Opp ELO"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Toughest schedule** — faced the best opponents (raw stats understate them)")
            st.dataframe(sub.sort_values("Opp ELO", ascending=False).head(12)[cols],
                         use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Easiest schedule** — faced the weakest opponents (raw stats flatter them)")
            st.dataframe(sub.sort_values("Opp ELO").head(12)[cols],
                         use_container_width=True, hide_index=True)

    with v_tab_b:
        _scatter(leaderboard, "ELO+", "wOBA+")
        _schedule_analysis(leaderboard, "ELO+", "wOBA+", lower_better=False)
    with v_tab_p:
        _scatter(pitcher_board, "ELO−", "wOBA-agst−")
        _schedule_analysis(pitcher_board, "ELO−", "wOBA-agst−", lower_better=True)

# ── Clutch & leverage ─────────────────────────────────────────────────────────
with st.expander("Clutch & Leverage — is there clutch, once you adjust for who you faced?"):
    st.markdown(
        "Sabermetrics says clutch barely exists — but that may be unfair. **High-leverage spots "
        "(late & close: 7th inning or later, within 2 runs) are when the best relievers pitch.** "
        "So a hitter's raw clutch line is built against tougher arms than usual. Here, each "
        "high-leverage PA is scored against its **ELO-expected** wOBA — which already accounts "
        "for that nasty closer — so **Clutch+** = how much a hitter beat expectation when it "
        "mattered, against the competition they actually faced."
    )
    clutch_df = cached_clutch(selected_years, league_woba)
    min_hi_pa = st.slider("Min high-leverage PAs", 20, 200, 50, 10, key="clutch_minpa")

    c_tab_b, c_tab_p = st.tabs(["Batters", "Pitchers"])

    def _clutch_table(role: str):
        opp_col = "p_elo" if role == "batter" else "b_elo"
        id_col = "batter" if role == "batter" else "pitcher"
        rated = batter_ratings if role == "batter" else pitcher_ratings
        # Premise: are opponents tougher in high leverage?
        opp_hi = clutch_df.loc[clutch_df["high"], opp_col].mean()
        opp_lo = clutch_df.loc[~clutch_df["high"], opp_col].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg opponent ELO — high leverage", f"{opp_hi:.0f}")
        m2.metric("Avg opponent ELO — low leverage", f"{opp_lo:.0f}")
        m3.metric("Clutch competition premium", f"{opp_hi - opp_lo:+.0f}",
                  help="Higher = you face meaningfully tougher arms/bats in the clutch.")

        hi = clutch_df[clutch_df["high"]]
        g = hi.groupby(id_col)
        sign = 1 if role == "batter" else -1  # pitchers: allowing less than expected is good
        tbl = pd.DataFrame({
            "HiLev PA": g.size(),
            "HiLev wOBA": g["woba"].mean().round(3),
            "Opp ELO": g[opp_col].mean().round(0),
            "Exp wOBA": g["exp"].mean().round(3),
            "Clutch+": (sign * g["resid"].mean()).round(3),
        })
        tbl = tbl[tbl["HiLev PA"] >= min_hi_pa]
        tbl.insert(0, "Name", [display_names.get(int(i), f"ID:{i}") for i in tbl.index])
        tbl = tbl.sort_values("Clutch+", ascending=False).reset_index(drop=True)

        st.caption("Clutch+ > 0 = produced better than ELO expected in high-leverage spots "
                   "(beat tough competition when it counted). Higher Opp ELO = faced nastier arms/bats.")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Most clutch** (beat expectation in the clutch)")
            st.dataframe(tbl.head(15), use_container_width=True, hide_index=True)
        with cc2:
            st.markdown("**Least clutch** (fell short when it mattered)")
            st.dataframe(tbl.tail(15).iloc[::-1].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

    with c_tab_b:
        _clutch_table("batter")
    with c_tab_p:
        _clutch_table("pitcher")

# ── Correlation over time ─────────────────────────────────────────────────────
with st.expander("Correlation Over Time — is the model tracking reality more closely?"):
    st.caption(
        "Per-season R² between the model index (ELO+/ELO−) and actual park-adjusted "
        "wOBA, then a linear trend across seasons. The trend's own R² says how well a "
        "straight line describes that year-to-year movement; the slope says the direction. "
        "(Actual = wOBA-based, since FanGraphs OPS+/wRC+ scraping is currently blocked.)"
    )
    corr_years = available_cached_years()
    st.caption(f"Showing all loaded seasons: {', '.join(map(str, corr_years)) or 'none'}. "
               "Load more seasons from the sidebar and they appear here automatically.")
    min_pa_corr = st.slider("Min PA per season for correlation", 100, 600, 300, 50, key="corr_minpa")

    if len(corr_years) >= 2:
        rows = []
        for y in sorted(corr_years):
            b_r2, b_n, p_r2, p_n = season_index_corr(y, min_pa_corr)
            rows.append({"Season": y, "Batter R²": b_r2, "Batter n": b_n,
                         "Pitcher R²": p_r2, "Pitcher n": p_n})
        corr_df = pd.DataFrame(rows)

        def _trend(xs, ys):
            mask = [v is not None for v in ys]
            xs2 = np.array([x for x, m in zip(xs, mask) if m], float)
            ys2 = np.array([v for v, m in zip(ys, mask) if m], float)
            if len(xs2) < 2:
                return None
            slope, intercept = np.polyfit(xs2, ys2, 1)
            pred = slope * xs2 + intercept
            ss_res = ((ys2 - pred) ** 2).sum()
            ss_tot = ((ys2 - ys2.mean()) ** 2).sum()
            trend_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            return slope, intercept, trend_r2, xs2

        fig_t = go.Figure()
        summary = []
        for col, color in [("Batter R²", "#1f77b4"), ("Pitcher R²", "#ff7f0e")]:
            ys = corr_df[col].tolist()
            xs = corr_df["Season"].tolist()
            fig_t.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=col,
                                       line=dict(color=color)))
            tr = _trend(xs, ys)
            if tr:
                slope, intercept, trend_r2, xs2 = tr
                fig_t.add_trace(go.Scatter(
                    x=[xs2.min(), xs2.max()],
                    y=[slope * xs2.min() + intercept, slope * xs2.max() + intercept],
                    mode="lines", line=dict(color=color, dash="dash"),
                    name=f"{col} trend", showlegend=False,
                ))
                direction = "increasing" if slope > 0 else "decreasing"
                summary.append(
                    f"**{col}**: slope = {slope:+.4f}/yr ({direction}), trend R² = {trend_r2:.3f}"
                )
        fig_t.update_layout(xaxis_title="Season", yaxis_title="R² (model vs actual)",
                            yaxis_range=[0, 1])
        st.plotly_chart(fig_t, use_container_width=True, key="corr_over_time")
        for line in summary:
            st.markdown(line)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
    else:
        st.info("Pick at least two seasons to see a trend.")

# ── Matchup predictor (cross-year) ────────────────────────────────────────────
st.subheader("Matchup Predictor")
st.caption(
    "Pick a season for each player, then the player. Ratings are league-relative "
    "(each season recenters to 1500), so cross-year matchups ask: how would these two "
    "relative skill levels fare in the chosen run environment? Avg ELO = sustained "
    "season level (recommended); End/Peak available too."
)

st.caption("Pick a season → the dropdown loads instantly (names only). "
           "Ratings are computed only when you've chosen both players.")
_metric_idx = {"Avg ELO": 2, "End ELO": 0, "Peak ELO": 4}


def _rating_for(year: int, pid: int, role: str, metric: str) -> float:
    """Run (cached) ELO for one season and return the chosen rating for one player."""
    _, _, _, res = season_bundle(year)
    ratings = res[_metric_idx[metric] + (0 if role == "batter" else 1)]
    return ratings.get(pid, 1500.0)


mp_metric = st.radio("Rating to use", ["Avg ELO", "End ELO", "Peak ELO"],
                     horizontal=True, key="mp_metric")

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown("**Batter**")
    b_year = st.selectbox("Batter season", all_years, index=0, key="mp_b_year")
    b_opts = season_player_labels(b_year, "batter")          # cheap: names only, no ELO
    batter_pick = st.selectbox("Batter", [""] + sorted(b_opts), key="mp_batter")
with mc2:
    st.markdown("**Pitcher**")
    p_year = st.selectbox("Pitcher season", all_years, index=0, key="mp_p_year")
    p_opts = season_player_labels(p_year, "pitcher")
    pitcher_pick = st.selectbox("Pitcher", [""] + sorted(p_opts), key="mp_pitcher")
with mc3:
    st.markdown("**Run environment**")
    env_year = st.selectbox("Season (sets league wOBA)", all_years, index=0, key="mp_env_year")
    env_lw = season_league_woba(env_year)                    # cheap: league mean, no ELO
    st.metric(f"{env_year} league wOBA", f"{env_lw:.3f}")

if batter_pick and pitcher_pick:
    b_id, p_id = b_opts[batter_pick], p_opts[pitcher_pick]
    with st.spinner("Computing ratings…"):                   # ELO runs here, only now
        r_b = _rating_for(b_year, b_id, "batter", mp_metric)
        r_p = _rating_for(p_year, p_id, "pitcher", mp_metric)
    exp_woba = expected_woba(r_b, r_p, env_lw)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"Batter {mp_metric}", f"{r_b:.0f}", help=f"{batter_pick}, {b_year}")
    r2.metric(f"Pitcher {mp_metric}", f"{r_p:.0f}", help=f"{pitcher_pick}, {p_year}")
    r3.metric("Expected wOBA", f"{exp_woba:.3f}", help=f"In {env_year} run environment")
    r4.metric(f"vs {env_year} league", f"{exp_woba - env_lw:+.3f}")

    # Real head-to-head only exists if both came from the same season
    if b_year == p_year:
        b_df, _ = cached_fetch((b_year,))
        h2h = b_df[(b_df["batter"] == b_id) & (b_df["pitcher"] == p_id)]
        if len(h2h) > 0:
            with st.expander(f"Real head-to-head, {b_year} ({len(h2h)} PA)"):
                h2h_display = h2h[["game_date", "events", "woba_value"]].copy()
                h2h_display["Result"] = h2h_display["events"].str.replace("_", " ").str.title()
                h2h_display = h2h_display[["game_date", "Result", "woba_value"]].rename(
                    columns={"game_date": "Date", "woba_value": "wOBA"})
                st.dataframe(h2h_display.sort_values("Date", ascending=False),
                             use_container_width=True, hide_index=True)
    else:
        st.caption(f"Hypothetical cross-era matchup ({b_year} batter vs {p_year} pitcher) — no real head-to-head.")

# ── Rating distribution ───────────────────────────────────────────────────────
st.subheader("Rating Distribution")
dist_weighted = st.checkbox(
    "Weight by plate appearances", value=True,
    help="On: each player counts proportional to PAs — both pools center at 1500 "
         "(the calibration). Off: raw player counts, which skew because low-PA "
         "batters and pitchers are distributed differently.",
)

b_vals = np.array([batter_ratings[p] for p in batter_ratings])
b_w = np.array([batter_pa.get(p, 0) for p in batter_ratings])
p_vals = np.array([pitcher_ratings[p] for p in pitcher_ratings])
p_w = np.array([pitcher_pa.get(p, 0) for p in pitcher_ratings])

lo = min(b_vals.min(), p_vals.min())
hi = max(b_vals.max(), p_vals.max())
bins = np.linspace(lo, hi, 61)
centers = (bins[:-1] + bins[1:]) / 2
weights_b = b_w if dist_weighted else None
weights_p = p_w if dist_weighted else None
b_hist, _ = np.histogram(b_vals, bins=bins, weights=weights_b)
p_hist, _ = np.histogram(p_vals, bins=bins, weights=weights_p)

fig = go.Figure()
fig.add_trace(go.Bar(x=centers, y=b_hist, name="Batters", opacity=0.6))
fig.add_trace(go.Bar(x=centers, y=p_hist, name="Pitchers", opacity=0.6))
fig.add_vline(x=1500, line_dash="dash", line_color="gray")
fig.update_layout(barmode="overlay", xaxis_title="ELO Rating",
                  yaxis_title="Plate appearances" if dist_weighted else "Players")
st.plotly_chart(fig, use_container_width=True)

# ── Player rating over time ───────────────────────────────────────────────────
st.subheader("ELO Rating Over Time")

# Display name → player id for the loaded scope (drives the time-series picker).
# With a Team Filter active, show every player on that team regardless of PA.
# Otherwise, filter to players meeting the Min PA bar (drops 1-PA noise).
# Unnamed "ID:" players are always excluded (nothing useful to chart).
def _picker(ratings, pa_counts, teams):
    out = {}
    for pid in ratings:
        if display_names[pid].startswith("ID:"):
            continue
        if team_filter:
            if teams.get(pid) in team_filter:
                out[display_names[pid]] = pid
        elif pa_counts.get(pid, 0) >= min_pa:
            out[display_names[pid]] = pid
    return out

batter_label_to_id = _picker(batter_ratings, batter_pa, batter_teams)
pitcher_label_to_id = _picker(pitcher_ratings, pitcher_pa, pitcher_teams)

def build_compressed_chart(histories: dict[str, list], role_label: str,
                           names: dict[int, str]) -> go.Figure:
    """
    Date-based x-axis with off-days compressed.
    Active days (any selected player has a PA) get 1 unit each.
    Gaps between active days are compressed to 0.2 * actual_gap_days,
    so a 5-day break looks slightly wider than a 1-day gap but doesn't
    dominate. Flat lines carry a player's ELO across their own off-days.
    """
    from collections import defaultdict

    # Union of all active dates across all players
    all_active = sorted({
        date
        for history in histories.values()
        for _, date, *_ in history
    })
    if not all_active:
        return go.Figure()

    # Assign compressed x positions to every active date
    date_to_x: dict = {}
    x = 0.0
    prev = None
    for date in all_active:
        if prev is not None:
            gap = (date - prev).days
            # Compress off-days; cap total gap width at 5 units so a long
            # break (injury, or the offseason in career mode) can't dominate.
            x += min(1.0 + 0.2 * (gap - 1), 5.0)
        date_to_x[date] = x
        prev = date

    # Month-start tick marks. Multi-year spans label the year too.
    multi_year = all_active[0].year != all_active[-1].year
    tick_fmt = "%b %Y" if multi_year else "%b %d"
    tickvals, ticktext = [], []
    seen_months: set = set()
    for date in all_active:
        mk = (date.year, date.month)
        if mk not in seen_months:
            seen_months.add(mk)
            tickvals.append(date_to_x[date])
            ticktext.append(date.strftime(tick_fmt))

    fig = go.Figure()

    for label, history in histories.items():
        by_date: dict = defaultdict(list)
        for pa_num, date, rating, event, opponent_id, delta, opp_elo in history:
            by_date[date].append((pa_num, rating, event, opponent_id, delta, opp_elo))

        xs, ys, hovers = [], [], []
        last_rating = None
        last_x = None

        for date in all_active:
            base_x = date_to_x[date]
            pas = by_date.get(date)

            if pas is None:
                # Player was off — extend flat line to this date's position
                if last_rating is not None:
                    xs.append(base_x)
                    ys.append(last_rating)
                    hovers.append(f"{date.strftime('%b %d')} | did not play")
                continue

            n = len(pas)
            # Flat carry-over to start of this game
            if last_rating is not None and last_x != base_x:
                xs.append(base_x)
                ys.append(last_rating)
                hovers.append(f"{date.strftime('%b %d')} | game start | ELO: {last_rating}")

            for i, (pa_num, rating, event, opponent_id, delta, opp_elo) in enumerate(pas):
                pa_x = base_x + (i / (n - 1) * 0.8 if n > 1 else 0.0)
                opponent = names.get(opponent_id, f"ID:{opponent_id}")
                result = event.replace("_", " ").title()
                opp_label = "vs P" if role_label == "PA" else "vs B"
                sign = "+" if delta >= 0 else ""
                hovers.append(
                    f"{date.strftime('%b %d')} | {role_label} #{pa_num}<br>"
                    f"{opp_label}: {opponent} (ELO {opp_elo})<br>"
                    f"Result: {result}<br>"
                    f"ELO: {rating} ({sign}{delta})"
                )
                xs.append(pa_x)
                ys.append(rating)

            last_rating = pas[-1][1]
            last_x = base_x

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=label, text=hovers, hoverinfo="text+name",
        ))

    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=-45)
    fig.update_layout(yaxis_title="ELO Rating", xaxis_title="Date (off-days compressed)")
    return fig


tab_b, tab_p = st.tabs(["Batters", "Pitchers"])

with tab_b:
    options = sorted(batter_label_to_id)
    selected = st.multiselect("Select batters", options, default=[])
    histories_b = {
        label: batter_history[batter_label_to_id[label]]
        for label in selected
        if batter_label_to_id[label] in batter_history
    }
    st.plotly_chart(build_compressed_chart(histories_b, "PA", names),
                    use_container_width=True, key="batter_timeline")

with tab_p:
    options = sorted(pitcher_label_to_id)
    selected = st.multiselect("Select pitchers", options, default=[])
    histories_p = {
        label: pitcher_history[pitcher_label_to_id[label]]
        for label in selected
        if pitcher_label_to_id[label] in pitcher_history
    }
    st.plotly_chart(build_compressed_chart(histories_p, "BF", names),
                    use_container_width=True, key="pitcher_timeline")

# ── Park factors ──────────────────────────────────────────────────────────────
with st.expander("Park Factors (computed from data)"):
    st.caption(
        "wOBA park factors via the home/road method, regressed 50% toward 1.0. "
        ">1.00 = hitter-friendly (the ELO bar is raised there); <1.00 = pitcher-friendly. "
        "Each PA's expected wOBA is multiplied by its park factor before scoring."
    )
    pf_dict = compute_park_factors(df)
    pf_df = pd.DataFrame(
        [{"Season": s, "Park": t, "Park Factor": round(v, 3)} for (s, t), v in pf_dict.items()]
    ).sort_values(["Season", "Park Factor"], ascending=[True, False]).reset_index(drop=True)
    st.dataframe(pf_df, use_container_width=True, hide_index=True, height=400)

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander("Diagnostics"):
    # PA-weighted mean: the calibration check. Both pools are recentered to
    # 1500 (PA-weighted) at the end of run_elo, so these should read ~1500.0.
    def pa_weighted_mean(ratings: dict, pa_counts: pd.Series) -> float:
        total_w, total_wr = 0.0, 0.0
        for pid, r in ratings.items():
            w = pa_counts.get(pid, 0)
            total_w += w
            total_wr += r * w
        return total_wr / total_w if total_w else 0.0

    b_wmean = pa_weighted_mean(batter_ratings, batter_pa)
    p_wmean = pa_weighted_mean(pitcher_ratings, pitcher_pa)
    c1, c2, c3 = st.columns(3)
    c1.metric("PA-weighted batter ELO", f"{b_wmean:.1f}")
    c2.metric("PA-weighted pitcher ELO", f"{p_wmean:.1f}")
    c3.metric("League wOBA used", f"{league_woba:.4f}")
    st.caption("Both pools are recentered so the PA-weighted mean is 1500, "
               "keeping batters and pitchers on a comparable scale.")
