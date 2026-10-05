"""Overview page: explore the cleaned player-season data and check it for errors.

Run from the project root:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run` puts app/ on sys.path, not the project root; add it so `src` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

st.set_page_config(page_title="NBA Prediction Dashboard", page_icon="🏀", layout="wide")

COLUMN_GROUPS = {
    "Identity": config.ID_COLS + ["TEAM_ID", "TEAM_COUNT"],
    "Traditional": ["AGE", "GP", "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A",
                    "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV", "STL",
                    "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS", "DD2", "TD3"],
    "Advanced": ["OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", "AST_RATIO",
                 "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "USG_PCT",
                 "PACE", "PIE", "POSS"],
    "Bio / draft": ["PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "INTERNATIONAL", "DRAFT_ROUND",
                    "DRAFT_NUMBER", "UNDRAFTED"],
    "Engineered": ["YEARS_IN_DATA", "AGE_SQ", "PTS_DELTA", "HAS_PREV_SEASON", "SEASON_GAMES",
                   "GP_SHARE", "FGA_PER_MIN", "FTA_RATE", "THREE_RATE", "TRADED", "NO_3PA", "NO_FTA"],
    "Flags": [f"OUTLIER_{c}" for c in config.OUTLIER_COLS],
    "Target": [config.TARGET],
}


@st.cache_data
def load_data():
    path = config.PROCESSED_DIR / "player_seasons.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    log_path = config.PROCESSED_DIR / "cleaning_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else None
    return df, log


def dummy_columns(df: pd.DataFrame) -> list:
    """One-hot columns created by DataProcessor.encode (TEAM_<abbr> and DRAFT_<round>)."""
    teams = set(df["TEAM_ABBREVIATION"])
    return [c for c in df.columns if (c.startswith("TEAM_") and c[5:] in teams) or c in ("DRAFT_R2", "DRAFT_UNDRAFTED")]


def quality_checks(df: pd.DataFrame) -> list:
    """Each check returns the rows that violate it; an empty result means the check passes."""
    pct_cols = ["FG_PCT", "FG3_PCT", "FT_PCT", "TS_PCT", "USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT"]
    team_dummies = [c for c in dummy_columns(df) if c.startswith("TEAM_")]
    checks = [
        ("One row per (PLAYER_ID, SEASON)", df[df.duplicated(["PLAYER_ID", "SEASON"], keep=False)]),
        ("No missing values outside TARGET_PTS", df[df.drop(columns=[config.TARGET]).isna().any(axis=1)]),
        ("Percentages within 0-1", df[~df[pct_cols].apply(lambda s: s.between(0, 1)).all(axis=1)]),
        (f"Small-sample filter held (GP >= {config.MIN_GP}, MIN >= {config.MIN_MINUTES:g})",
         df[(df["GP"] < config.MIN_GP) | (df["MIN"] < config.MIN_MINUTES)]),
        ("AGE between 18 and 45", df[~df["AGE"].between(18, 45)]),
        ("Height 65-95 in, weight 150-330 lb",
         df[~df["PLAYER_HEIGHT_INCHES"].between(65, 95) | ~df["PLAYER_WEIGHT"].between(150, 330)]),
        (f"DRAFT_NUMBER between 1 and {config.UNDRAFTED_PICK}", df[~df["DRAFT_NUMBER"].between(1, config.UNDRAFTED_PICK)]),
        ("UNDRAFTED flag matches pick 61",
         df[(df["UNDRAFTED"] == 1) != (df["DRAFT_NUMBER"] == config.UNDRAFTED_PICK)]),
        # Per-game values are rounded to 0.1, so allow a little slack on identities
        ("PTS = 2*FGM + FG3M + FTM (+/-0.5)", df[(df["PTS"] - (2 * df["FGM"] + df["FG3M"] + df["FTM"])).abs() > 0.5]),
        ("REB = OREB + DREB (+/-0.15)", df[(df["REB"] - (df["OREB"] + df["DREB"])).abs() > 0.15]),
        ("GP_SHARE <= 1.1 (traded players can slightly exceed 1)", df[df["GP_SHARE"] > 1.1]),
        ("At most one team dummy per row", df[df[team_dummies].sum(axis=1) > 1]),
        (f"No target in live season {config.LIVE_SEASON}",
         df[(df["SEASON"] == config.LIVE_SEASON) & df[config.TARGET].notna()]),
    ]
    return checks


df, log = load_data()
st.title("🏀 NBA Next-Season Scoring: Data Overview")
if df is None:
    st.error("No processed data found. Run `python -m src.data_processor` from the project root first.")
    st.stop()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
seasons = sorted(df["SEASON"].unique())
sel_seasons = st.sidebar.multiselect("Season", seasons, default=seasons)
teams = sorted(df["TEAM_ABBREVIATION"].unique())
sel_teams = st.sidebar.multiselect("Team", teams, placeholder="All teams")
name_query = st.sidebar.text_input("Player name contains")
min_minutes = st.sidebar.slider("Minimum minutes per game", 10.0, float(df["MIN"].max()), 10.0, 0.5)

view = df[df["SEASON"].isin(sel_seasons) & (df["MIN"] >= min_minutes)]
if sel_teams:
    view = view[view["TEAM_ABBREVIATION"].isin(sel_teams)]
if name_query:
    # regex=False: the search box is user input, treat it as plain text
    view = view[view["PLAYER_NAME"].str.contains(name_query, case=False, regex=False)]

# ------------------------------------------------------------------ headline numbers
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Player-seasons", f"{len(df):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Seasons", len(seasons), help=f"{seasons[0]} to {seasons[-1]} regular seasons")
c4.metric("Unique players", f"{df['PLAYER_ID'].nunique():,}")
c5.metric("Rows with target", f"{df[config.TARGET].notna().sum():,}")
st.caption(f"Filtered view: **{len(view):,}** rows. Target = the player's points per game in the following season.")

tab_data, tab_stats, tab_quality, tab_pipeline = st.tabs(
    ["Data preview", "Summary statistics", "Data quality checks", "Cleaning pipeline"])

# ------------------------------------------------------------------ preview
with tab_data:
    groups = st.multiselect("Column groups", list(COLUMN_GROUPS), default=["Identity", "Traditional", "Target"])
    show_dummies = st.checkbox("Show one-hot columns (TEAM_*, DRAFT_R2, DRAFT_UNDRAFTED)")
    cols = list(dict.fromkeys(c for g in groups for c in COLUMN_GROUPS[g] if c in df.columns))
    if show_dummies:
        cols += dummy_columns(df)
    st.dataframe(view[cols or ["PLAYER_NAME", "SEASON"]], width="stretch", hide_index=True, height=500)
    st.download_button("Download filtered rows (CSV)", view[cols].to_csv(index=False), "player_seasons_filtered.csv")

# ------------------------------------------------------------------ stats
with tab_stats:
    numeric = view.select_dtypes("number").drop(columns=["PLAYER_ID", "TEAM_ID", "SEASON_START"], errors="ignore")
    numeric = numeric.drop(columns=dummy_columns(df))
    st.subheader("describe()")
    st.dataframe(numeric.describe().T.round(3), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Player-seasons per season")
        st.bar_chart(view.groupby("SEASON").size().rename("rows"))
    with right:
        st.subheader("Distribution of a column")
        col = st.selectbox("Column", numeric.columns, index=list(numeric.columns).index("PTS"))
        values = view[col].dropna()
        if values.empty:
            st.info("No rows match the current filters.")
        else:
            hist = pd.cut(values, bins=30).value_counts(sort=False)
            # Numeric midpoints keep bins in order; string labels get sorted alphabetically
            hist.index = pd.Index([round(iv.mid, 3) for iv in hist.index], name=col)
            st.bar_chart(hist.rename("count"))

    st.subheader("Categorical breakdowns")
    a, b, c = st.columns(3)
    a.dataframe(view["DRAFT_ROUND"].value_counts().rename("rows"), width="stretch")
    b.dataframe(view["INTERNATIONAL"].map({0: "USA", 1: "International"}).value_counts().rename("rows"), width="stretch")
    c.dataframe(view["TEAM_ABBREVIATION"].value_counts().rename("rows"), width="stretch", height=250)

# ------------------------------------------------------------------ quality
with tab_quality:
    st.write("Automated checks run on the **full** cleaned table (filters ignored). "
             "Expand a failing check to see the offending rows.")
    for name, bad in quality_checks(df):
        if bad.empty:
            st.success(f"✅ {name}")
        else:
            with st.expander(f"❌ {name}: {len(bad)} rows", expanded=False):
                st.dataframe(bad, width="stretch", hide_index=True)

    st.subheader("Missing values per column")
    na = df.isna().sum()
    na = na[na > 0]
    if na.empty:
        st.write("No missing values.")
    else:
        st.dataframe(na.rename("missing").to_frame(), width="stretch")
        st.caption(f"{config.TARGET} is expected to be missing for players with no qualified next season "
                   f"(retired, injured, under the minutes filter) and for every {config.LIVE_SEASON} row.")

    st.subheader("Flagged outliers (kept on purpose)")
    flag_cols = [f"OUTLIER_{c}" for c in config.OUTLIER_COLS]
    flagged = view[view[flag_cols].any(axis=1)]
    st.caption(f"{len(flagged)} rows in the filtered view are outside 1.5×IQR on {', '.join(config.OUTLIER_COLS)}. "
               "These are real stars or heavy-usage players, not errors.")
    st.dataframe(flagged[config.ID_COLS + config.OUTLIER_COLS + flag_cols].sort_values("PTS", ascending=False),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------------ pipeline
with tab_pipeline:
    st.subheader("Row counts at each cleaning step")
    if log is not None:
        st.dataframe(log, width="stretch", hide_index=True)
    else:
        st.write("cleaning_log.csv not found.")
    st.subheader("Train / validation / test split (by season)")
    split_rows = []
    for name, seasons_ in [("train", config.TRAIN_SEASONS), ("val", config.VAL_SEASONS),
                           ("test", config.TEST_SEASONS), ("live", [config.LIVE_SEASON])]:
        part = df[df["SEASON"].isin(seasons_)]
        n = len(part) if name == "live" else part[config.TARGET].notna().sum()
        split_rows.append({"split": name, "feature seasons": ", ".join(seasons_), "rows": int(n)})
    st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True)
