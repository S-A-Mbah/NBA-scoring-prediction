"""Turn the 30 raw nba_api tables into one model-ready player-season table.

Pipeline (each step logs before/after row counts to data/processed/cleaning_log.csv):
    load_raw -> inspect -> merge -> clean -> engineer_features -> build_target -> encode
    -> split (train-median imputation) -> save

Each output row is one player in season t; TARGET_PTS is that player's PPG in season t+1.
Same-season PTS is never the target: PTS ~= 2*FGM + FG3M + FTM, so it would leak.

Usage:
    python -m src.data_processor
"""
import io

import numpy as np
import pandas as pd

from src import config
from src.collect_data import NBADataCollector

KEYS = ["PLAYER_ID", "SEASON"]

# Columns kept from each raw table. Everything else is dropped on purpose:
#   *_RANK            within-season rank of another column: redundant
#   E_* / sp_work_*   NBA's estimated/internal duplicates of the real ratings
#   *FANTASY*, FP_*   fantasy scoring, a linear mix of columns we already keep
#   NICKNAME          free text with no signal
#   advanced FGM/FGA  season TOTALS (base has per-game), would collide with base columns
#   bio overlaps      GP/PTS/REB/... are identical to base/advanced (checked in the profiling step)
BASE_COLS = KEYS + [
    "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "GP", "W", "L", "W_PCT", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
    "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
    "DD2", "TD3", "TEAM_COUNT",
]
ADVANCED_COLS = KEYS + [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", "AST_RATIO", "OREB_PCT",
    "DREB_PCT", "REB_PCT", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "USG_PCT", "PACE", "PIE", "POSS",
]
BIO_COLS = KEYS + [
    "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "COUNTRY", "COLLEGE", "DRAFT_YEAR",
    "DRAFT_ROUND", "DRAFT_NUMBER",
]


class DataProcessor:
    """Loads, merges, cleans, engineers, splits and saves the NBA player-season data."""

    def __init__(self, raw_dir=config.RAW_DIR, out_dir=config.PROCESSED_DIR):
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.raw: dict = {}
        self.df: pd.DataFrame | None = None
        self.splits: dict = {}
        self.log: list = []          # (step, rows_before, rows_after, note)
        self.medians: dict = {}      # training medians used for imputation
        self.feature_cols: list = []

    def _log(self, step, before, after, note=""):
        self.log.append({"step": step, "rows_before": before, "rows_after": after, "note": note})
        print(f"  {step:<28} {before:>6} -> {after:>6}  {note}")

    # ------------------------------------------------------------------ 1. load / inspect
    def load_raw(self) -> dict:
        self.raw = NBADataCollector(raw_dir=self.raw_dir).load_raw()
        return self.raw

    def inspect(self) -> str:
        """shape / dtypes / describe / missing counts per raw table (printed and saved)."""
        buf = io.StringIO()
        for name, df in self.raw.items():
            buf.write(f"\n{'=' * 30} {name} {df.shape}\n")
            buf.write(f"dtypes:\n{df.dtypes.value_counts().to_string()}\n")
            na = df.isna().sum()
            buf.write(f"missing values:\n{na[na > 0].to_string() if na.any() else 'none'}\n")
            buf.write(f"describe:\n{df.describe().T.to_string()}\n")
        report = buf.getvalue()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "inspection_report.txt").write_text(report, encoding="utf-8")
        return report

    # ------------------------------------------------------------------ 2. merge
    def merge(self) -> pd.DataFrame:
        """Join base + advanced + bio on (PLAYER_ID, SEASON); seasons are already stacked."""
        base = self.raw["base"][BASE_COLS]
        n = len(base)
        # validate="one_to_one" raises if any table has a duplicated player-season
        df = (base
              .merge(self.raw["advanced"][ADVANCED_COLS], on=KEYS, how="inner", validate="one_to_one")
              .merge(self.raw["bio"][BIO_COLS], on=KEYS, how="inner", validate="one_to_one"))
        df["SEASON_START"] = df["SEASON"].str[:4].astype(int)   # "2016-17" -> 2016, sortable
        df = df.sort_values(["PLAYER_ID", "SEASON_START"]).reset_index(drop=True)
        # Career index over ALL appearances, computed before the minutes filter so a player's
        # experience is not understated by short seasons that get filtered out later.
        df["YEARS_IN_DATA"] = df.groupby("PLAYER_ID").cumcount()
        self._log("merge base+adv+bio", n, len(df), f"{df['SEASON'].nunique()} seasons stacked")
        self.df = df
        return df

    # ------------------------------------------------------------------ 3. clean
    @staticmethod
    def _season_games(df: pd.DataFrame) -> pd.Series:
        """Games per team in each season, derived from the data rather than hard-coded.

        2019-20 and 2020-21 were shortened (COVID), so raw GP is not comparable across years.
        The season max GP among non-traded players recovers the schedule (82; 72 in 2020-21;
        74 in 2019-20, when teams played unequal counts). Traded players are excluded because
        they can exceed 82 by switching teams. A per-team median was tried and rejected: few
        players appear in every game, so it understated normal seasons (78.5 for 2021-22).
        """
        return df[df["TEAM_COUNT"] == 1].groupby("SEASON")["GP"].max()

    def clean(self) -> pd.DataFrame:
        df = self.df.copy()

        # 3a. duplicates (merge already validated one row per player-season; this is a guard)
        n = len(df)
        df = df.drop_duplicates()
        assert not df.duplicated(KEYS).any(), "more than one row per (PLAYER_ID, SEASON)"
        self._log("drop exact duplicates", n, len(df),
                  f"traded players arrive as one combined row ({(df['TEAM_COUNT'] > 1).sum()} rows)")
        df["TRADED"] = (df["TEAM_COUNT"] > 1).astype(int)

        # 3b. draft fields. "Undrafted", round/pick 0, and a draft year with no round/pick all
        # describe undrafted players (e.g. T.J. McConnell has round 0, Allonzo Trier has none).
        rnd = df["DRAFT_ROUND"].astype("string").str.strip()
        pick = pd.to_numeric(df["DRAFT_NUMBER"], errors="coerce")
        undrafted = rnd.isna() | rnd.isin(["Undrafted", "0"]) | pick.isna() | (pick == 0)
        df["UNDRAFTED"] = undrafted.astype(int)
        df["DRAFT_NUMBER"] = pick.where(~undrafted, config.UNDRAFTED_PICK).astype(int)
        # Round looks ordinal but "Undrafted" is not a point on that scale -> nominal category
        df["DRAFT_ROUND"] = np.where(undrafted, "UNDRAFTED", "R" + rnd.fillna("").astype(str))
        self._log("fix draft fields", len(df), len(df), f"{int(undrafted.sum())} rows undrafted -> pick {config.UNDRAFTED_PICK}")

        # 3c. free-text bio -> flag. COLLEGE is NaN for internationals AND US prep/G-League
        # players, so COUNTRY is the reliable signal; the free text is then dropped.
        df["INTERNATIONAL"] = (df["COUNTRY"].astype("string").str.strip() != "USA").astype(int)
        df = df.drop(columns=["COUNTRY", "COLLEGE", "DRAFT_YEAR"])
        # Weight: NBA re-lists players each season, so borrow the player's own other seasons first
        df["PLAYER_WEIGHT"] = df.groupby("PLAYER_ID")["PLAYER_WEIGHT"].transform(lambda s: s.ffill().bfill())

        # 3d. 0% vs "no attempts". Per-game attempts are rounded, so a percentage is only
        # treated as missing when BOTH attempts and percentage are 0; 0-for-N stays a real 0%.
        for att, pct, flag in [("FG3A", "FG3_PCT", "NO_3PA"), ("FTA", "FT_PCT", "NO_FTA"), ("FGA", "FG_PCT", None)]:
            none = (df[att] == 0) & (df[pct] == 0)
            df.loc[none, pct] = np.nan
            if flag:
                df[flag] = none.astype(int)

        # 3e. season length (needed for GP_SHARE) from the full table, before filtering
        season_games = self._season_games(df)
        df["SEASON_GAMES"] = df["SEASON"].map(season_games)

        # 3f. small-sample filter
        n = len(df)
        df = df[(df["GP"] >= config.MIN_GP) & (df["MIN"] >= config.MIN_MINUTES)].copy()
        self._log("filter GP>=10 & MIN>=10", n, len(df), f"removed {n - len(df)} small-sample rows")

        # 3g. range validation: the impossible values seen in raw data (TS_PCT 1.5, PIE -4)
        # came from 1-5 game samples; fail loudly if any survive the filter.
        for col, lo, hi in [("TS_PCT", 0, 1), ("EFG_PCT", 0, 1.5), ("USG_PCT", 0, 1), ("PIE", -1, 1)]:
            bad = ~df[col].between(lo, hi)
            assert not bad.any(), f"{bad.sum()} rows with {col} outside [{lo}, {hi}]"

        # 3h. outliers: flag, keep. A 30-PPG season is a real star, not a data error.
        for col in config.OUTLIER_COLS:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[f"OUTLIER_{col}"] = (~df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)).astype(int)
        flagged = df[[f"OUTLIER_{c}" for c in config.OUTLIER_COLS]].any(axis=1).sum()
        self._log("flag IQR outliers (kept)", len(df), len(df), f"{flagged} rows flagged on {config.OUTLIER_COLS}")

        self.df = df.reset_index(drop=True)
        return self.df

    # ------------------------------------------------------------------ 4. features
    @staticmethod
    def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        return (num / den.where(den > 0)).fillna(0.0)

    def engineer_features(self) -> pd.DataFrame:
        df = self.df
        df["AGE_SQ"] = df["AGE"] ** 2      # scoring rises then falls with age
        df["GP_SHARE"] = df["GP"] / df["SEASON_GAMES"]
        df["FGA_PER_MIN"] = self._safe_ratio(df["FGA"], df["MIN"])
        df["FTA_RATE"] = self._safe_ratio(df["FTA"], df["FGA"])
        df["THREE_RATE"] = self._safe_ratio(df["FG3A"], df["FGA"])

        # Trend vs the immediately previous (qualified) season. Matched on SEASON_START - 1,
        # not "previous row", because 132 careers skip a season.
        prev = df[["PLAYER_ID", "SEASON_START", "PTS"]].assign(SEASON_START=lambda d: d["SEASON_START"] + 1)
        df = df.merge(prev.rename(columns={"PTS": "PREV_PTS"}), on=["PLAYER_ID", "SEASON_START"], how="left")
        df["HAS_PREV_SEASON"] = df["PREV_PTS"].notna().astype(int)
        df["PTS_DELTA"] = (df["PTS"] - df["PREV_PTS"]).fillna(0.0)   # 0 for first/after-gap seasons
        self.df = df.drop(columns=["PREV_PTS"])
        self._log("engineer features", len(df), len(self.df), "AGE_SQ, GP_SHARE, rates, PTS_DELTA")
        return self.df

    # ------------------------------------------------------------------ 5. target
    def build_target(self) -> pd.DataFrame:
        """TARGET_PTS = the same player's PPG in season t+1 (NaN if no qualified t+1 season)."""
        df = self.df
        nxt = df[["PLAYER_ID", "SEASON_START", "PTS"]].assign(SEASON_START=lambda d: d["SEASON_START"] - 1)
        df = df.merge(nxt.rename(columns={"PTS": config.TARGET}), on=["PLAYER_ID", "SEASON_START"],
                      how="left", validate="one_to_one")
        has = df[config.TARGET].notna().sum()
        self._log("build t+1 target", len(df), len(df),
                  f"{has} rows with target; {len(df) - has} without (retired/injured/filtered/{config.LIVE_SEASON})")
        self.df = df
        return df

    # ------------------------------------------------------------------ 6. encode
    def encode(self) -> pd.DataFrame:
        """One-hot the nominal categories with drop_first=True (avoids the dummy trap)."""
        df = self.df
        # Fixed, sorted category lists so every split gets identical dummy columns
        team = pd.Categorical(df["TEAM_ABBREVIATION"], categories=sorted(df["TEAM_ABBREVIATION"].unique()))
        rnd = pd.Categorical(df["DRAFT_ROUND"], categories=["R1", "R2", "UNDRAFTED"])
        assert not pd.isna(rnd).any(), f"unexpected DRAFT_ROUND values: {set(df['DRAFT_ROUND']) - {'R1', 'R2', 'UNDRAFTED'}}"
        dummies = pd.concat([
            pd.get_dummies(team, prefix="TEAM", drop_first=True, dtype=int),
            pd.get_dummies(rnd, prefix="DRAFT", drop_first=True, dtype=int),   # baseline = round 1
        ], axis=1)
        dummies.index = df.index
        self.df = pd.concat([df, dummies], axis=1)
        self.feature_cols = config.NUMERIC_FEATURES + list(dummies.columns)
        self._log("one-hot team + draft round", len(df), len(self.df), f"{dummies.shape[1]} dummy columns")
        return self.df

    # ------------------------------------------------------------------ 7. split
    def split(self) -> dict:
        """Time-based split by feature season, then impute with TRAIN medians only."""
        df = self.df
        train_mask = df["SEASON"].isin(config.TRAIN_SEASONS)
        self.medians = df.loc[train_mask, config.IMPUTE_MEDIAN_COLS].median().to_dict()
        df = df.fillna(value=self.medians)   # only touches the IMPUTE_MEDIAN_COLS keys
        self.df = df

        labelled = df[df[config.TARGET].notna()]
        self.splits = {
            "train": labelled[labelled["SEASON"].isin(config.TRAIN_SEASONS)],
            "val": labelled[labelled["SEASON"].isin(config.VAL_SEASONS)],
            "test": labelled[labelled["SEASON"].isin(config.TEST_SEASONS)],
            "live": df[df["SEASON"] == config.LIVE_SEASON],   # no target: 2026-27 is the future
        }
        total = sum(len(self.splits[k]) for k in ("train", "val", "test"))
        for name, part in self.splits.items():
            share = f"{len(part) / total:.0%} of labelled" if name != "live" else "prediction rows"
            self._log(f"split: {name}", len(df), len(part), share)

        # Contract for the modelling phase: features are complete
        for name, part in self.splits.items():
            na = part[self.feature_cols].isna().sum()
            assert not na.any(), f"NaNs in {name} features: {na[na > 0].to_dict()}"
        return self.splits

    # ------------------------------------------------------------------ 8. save
    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.out_dir / "player_seasons.csv", index=False)
        for name, part in self.splits.items():
            part.to_csv(self.out_dir / f"{name}.csv", index=False)
        pd.DataFrame(self.log).to_csv(self.out_dir / "cleaning_log.csv", index=False)
        pd.Series(self.feature_cols, name="feature").to_csv(self.out_dir / "feature_columns.csv", index=False)
        print(f"\nSaved to {self.out_dir}")

    def run(self) -> dict:
        print("Processing NBA player-seasons")
        self.load_raw()
        self.inspect()
        self.merge()
        self.clean()
        self.engineer_features()
        self.build_target()
        self.encode()
        self.split()
        self.save()
        return self.splits


if __name__ == "__main__":
    DataProcessor().run()
