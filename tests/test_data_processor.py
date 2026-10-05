"""Behavioural tests for DataProcessor on small synthetic raw tables (no network, no raw CSVs)."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_processor import ADVANCED_COLS, BASE_COLS, BIO_COLS, DataProcessor

S = config.SEASONS  # S[0] = 2016-17 (train), S[6] = 2022-23 (val), S[-1] = 2025-26 (live)


def row(pid, season, **over):
    """One qualified player-season with plausible defaults; override any column."""
    r = {c: 1.0 for c in set(BASE_COLS + ADVANCED_COLS + BIO_COLS)}
    r.update(PLAYER_ID=pid, SEASON=season, PLAYER_NAME=f"P{pid}", TEAM_ID=1, TEAM_ABBREVIATION="BOS",
             AGE=25.0, GP=60, MIN=25.0, PTS=12.0, FGA=10.0, FG_PCT=0.45, FG3A=3.0, FG3_PCT=0.35,
             FTA=2.0, FT_PCT=0.8, TEAM_COUNT=1, TS_PCT=0.55, EFG_PCT=0.5, USG_PCT=0.2, PIE=0.1,
             PLAYER_HEIGHT_INCHES=78, PLAYER_WEIGHT=210.0, COUNTRY="USA", COLLEGE="Duke",
             DRAFT_YEAR="2015", DRAFT_ROUND="1", DRAFT_NUMBER="10")
    r.update(over)
    return r


def make_processor(rows, tmp_path):
    df = pd.DataFrame(rows)
    p = DataProcessor(raw_dir=tmp_path, out_dir=tmp_path)
    p.raw = {"base": df[BASE_COLS], "advanced": df[ADVANCED_COLS], "bio": df[BIO_COLS]}
    return p


def run(p):
    p.merge(); p.clean(); p.engineer_features(); p.build_target(); p.encode(); p.split()
    return p.df.set_index(["PLAYER_ID", "SEASON"])


@pytest.fixture
def rows():
    return [
        # player 1: consecutive seasons, then a gap (S[2] missing) -> S[1] has no target
        row(1, S[0], PTS=10.0), row(1, S[1], PTS=14.0), row(1, S[3], PTS=20.0),
        # player 2: undrafted variants + no 3PA/FTA
        row(2, S[0], DRAFT_ROUND="Undrafted", DRAFT_NUMBER="Undrafted", FG3A=0.0, FG3_PCT=0.0),
        row(2, S[1], DRAFT_ROUND="0", DRAFT_NUMBER="0", FTA=0.0, FT_PCT=0.0),
        row(2, S[2], DRAFT_ROUND=np.nan, DRAFT_NUMBER=np.nan),
        # player 3: real 0-for-N from three (attempts > 0) must stay 0%
        row(3, S[0], FG3A=0.5, FG3_PCT=0.0, COUNTRY="France", COLLEGE=np.nan),
        # player 4: small sample -> filtered out
        row(4, S[0], GP=3, MIN=4.0),
        # validation, test and live rows so every split exists
        row(5, S[6]), row(5, S[7]), row(5, S[8]), row(5, S[9]),
    ]


def test_one_row_per_player_season_and_no_nan_features(rows, tmp_path):
    p = make_processor(rows, tmp_path)
    df = run(p)
    assert df.index.is_unique
    assert not p.df[p.feature_cols].isna().any().any()
    assert (4, S[0]) not in df.index          # GP < 10 removed


def test_target_is_next_season_by_number_not_next_row(rows, tmp_path):
    df = run(make_processor(rows, tmp_path))
    assert df.loc[(1, S[0]), "TARGET_PTS"] == 14.0
    assert np.isnan(df.loc[(1, S[1]), "TARGET_PTS"])   # next row is S[3], a gap
    assert df.loc[(1, S[1]), "PTS_DELTA"] == pytest.approx(4.0)
    assert df.loc[(1, S[3]), ["PTS_DELTA", "HAS_PREV_SEASON"]].tolist() == [0.0, 0]
    assert np.isnan(df.loc[(5, S[9]), "TARGET_PTS"])   # live season has no future


def test_all_undrafted_encodings_map_to_pick_61(rows, tmp_path):
    df = run(make_processor(rows, tmp_path))
    p2 = df.loc[2]
    assert (p2["DRAFT_NUMBER"] == config.UNDRAFTED_PICK).all()
    assert (p2["UNDRAFTED"] == 1).all() and (p2["DRAFT_UNDRAFTED"] == 1).all()
    assert df.loc[(1, S[0]), ["DRAFT_NUMBER", "DRAFT_UNDRAFTED", "DRAFT_R2"]].tolist() == [10, 0, 0]


def test_no_attempts_imputed_but_real_zero_percent_kept(rows, tmp_path):
    p = make_processor(rows, tmp_path)
    df = run(p)
    assert df.loc[(2, S[0]), "NO_3PA"] == 1
    assert df.loc[(2, S[0]), "FG3_PCT"] == p.medians["FG3_PCT"]   # imputed, not 0
    assert df.loc[(2, S[1]), "NO_FTA"] == 1
    assert df.loc[(3, S[0]), ["FG3_PCT", "NO_3PA"]].tolist() == [0.0, 0]
    assert df.loc[(3, S[0]), "INTERNATIONAL"] == 1


def test_imputation_medians_come_from_training_seasons_only(rows, tmp_path):
    # Enough extreme validation rows that a median over ALL rows would become 0.99
    rows += [row(100 + i, S[6], FG3_PCT=0.99) for i in range(12)]
    p = make_processor(rows, tmp_path)
    run(p)
    assert p.medians["FG3_PCT"] == pytest.approx(0.35)


def test_splits_do_not_share_seasons(rows, tmp_path):
    p = make_processor(rows, tmp_path)
    run(p)
    seasons = {k: set(v["SEASON"]) for k, v in p.splits.items()}
    assert seasons["train"].isdisjoint(seasons["val"] | seasons["test"] | seasons["live"])
    assert seasons["val"].isdisjoint(seasons["test"])
