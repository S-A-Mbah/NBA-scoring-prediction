"""Project-wide settings: seasons, paths and constants shared by every module."""
from pathlib import Path

# Paths are resolved from the project root so scripts work from any working directory
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# 2016-17 ... 2025-26 (10 regular seasons)
FIRST_SEASON_START = 2016
LAST_SEASON_START = 2025
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(FIRST_SEASON_START, LAST_SEASON_START + 1)]
SEASON_TYPE = "Regular Season"

# Raw tables pulled per season (name used in data/raw/<table>_<season>.csv)
RAW_TABLES = ["base", "advanced", "bio"]

# API politeness / robustness
REQUEST_TIMEOUT = 60      # seconds per request
REQUEST_DELAY = 2.0       # pause between requests; stats.nba.com throttles bursts
MAX_RETRIES = 3

RANDOM_STATE = 42

# ---------------------------------------------------------------- data processing
# Small-sample filter: a 2-game, 3-minute season is noise and drives most extreme outliers
MIN_GP = 10
MIN_MINUTES = 10.0
UNDRAFTED_PICK = 61          # one past the last pick (60) so pick order stays monotonic
OUTLIER_COLS = ["PTS", "MIN", "USG_PCT"]   # flagged with the IQR rule, never removed

# Time-based split by feature season t (target is season t+1). A random split would put
# the same player's adjacent seasons in train and test and overstate accuracy.
TRAIN_SEASONS = SEASONS[:6]   # 2016-17 .. 2021-22
VAL_SEASONS = SEASONS[6:8]    # 2022-23, 2023-24
TEST_SEASONS = SEASONS[8:9]   # 2024-25 (target 2025-26), used once
LIVE_SEASON = SEASONS[-1]     # 2025-26 -> dashboard predictions for 2026-27

ID_COLS = ["PLAYER_ID", "PLAYER_NAME", "SEASON", "SEASON_START", "TEAM_ABBREVIATION"]
TARGET = "TARGET_PTS"

# Numeric model inputs (team / draft-round dummies are appended at runtime, see DataProcessor)
NUMERIC_FEATURES = [
    # traditional per-game
    "AGE", "GP", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
    "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PFD", "PTS",
    "PLUS_MINUS", "W_PCT",
    # advanced
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", "OREB_PCT", "DREB_PCT",
    "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "USG_PCT", "PACE", "PIE",
    # bio
    "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "DRAFT_NUMBER", "INTERNATIONAL",
    # engineered
    "YEARS_IN_DATA", "AGE_SQ", "PTS_DELTA", "HAS_PREV_SEASON", "GP_SHARE", "FGA_PER_MIN",
    "FTA_RATE", "THREE_RATE", "TRADED", "NO_3PA", "NO_FTA",
]
# Columns whose NaNs are filled with the TRAINING-season median (fit on train only: no leakage)
IMPUTE_MEDIAN_COLS = ["FG3_PCT", "FT_PCT", "FG_PCT", "PLAYER_WEIGHT"]
