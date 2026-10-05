"""Download raw NBA player stats from stats.nba.com via the nba_api package.

Data source: nba_api (https://github.com/swar/nba_api), a client for the JSON API
behind https://www.nba.com/stats. Cite it in the report.

Writes one CSV per table per season to data/raw/<table>_<season>.csv (never edit
these by hand) and returns the stacked tables as DataFrames for the next phase.

Usage:
    python -m src.collect_data            # download missing seasons only
    python -m src.collect_data --force    # re-download everything
"""
import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerbiostats, leaguedashplayerstats

from src import config


class NBADataCollector:
    """Fetches the base, advanced and bio player tables for each season."""

    def __init__(self, seasons=None, raw_dir=config.RAW_DIR):
        self.seasons = seasons or config.SEASONS
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, table: str, season: str):
        return self.raw_dir / f"{table}_{season}.csv"

    def _request(self, table: str, season: str) -> pd.DataFrame:
        """One API call with retries; each table is a single response for the whole league."""
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                if table == "bio":
                    endpoint = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                        season=season,
                        season_type_all_star=config.SEASON_TYPE,
                        per_mode_simple="PerGame",
                        timeout=config.REQUEST_TIMEOUT,
                    )
                else:
                    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
                        season=season,
                        season_type_all_star=config.SEASON_TYPE,
                        per_mode_detailed="PerGame",
                        measure_type_detailed_defense=table.capitalize(),  # "Base" / "Advanced"
                        timeout=config.REQUEST_TIMEOUT,
                    )
                return endpoint.get_data_frames()[0]
            except Exception as exc:  # network errors / timeouts from stats.nba.com
                if attempt == config.MAX_RETRIES:
                    raise
                wait = config.REQUEST_DELAY * 2 ** attempt
                print(f"    {table} {season} failed ({exc.__class__.__name__}); retry in {wait:.0f}s")
                time.sleep(wait)

    def fetch_season(self, season: str, force: bool = False) -> dict:
        """Fetch all raw tables for one season, saving each response as-is."""
        frames = {}
        for table in config.RAW_TABLES:
            path = self._path(table, season)
            if path.exists() and not force:
                frames[table] = pd.read_csv(path)
                print(f"  {table:<9} {season}: cached ({len(frames[table])} rows)")
                continue
            df = self._request(table, season)
            df["SEASON"] = season
            df.to_csv(path, index=False)
            frames[table] = df
            print(f"  {table:<9} {season}: downloaded {len(df)} rows")
            time.sleep(config.REQUEST_DELAY)
        return frames

    def fetch_all(self, force: bool = False) -> dict:
        """Fetch every season and return {table: DataFrame stacked over all seasons}."""
        per_table = {t: [] for t in config.RAW_TABLES}
        for season in self.seasons:
            print(f"Season {season}")
            for table, df in self.fetch_season(season, force=force).items():
                per_table[table].append(df)
        return {t: pd.concat(dfs, ignore_index=True) for t, dfs in per_table.items()}

    def load_raw(self) -> dict:
        """Load already-downloaded CSVs without touching the API (for the processing phase)."""
        out = {}
        for table in config.RAW_TABLES:
            files = sorted(self.raw_dir.glob(f"{table}_*.csv"))
            if not files:
                raise FileNotFoundError(f"No raw files for '{table}' in {self.raw_dir}; run collect_data first")
            out[table] = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        return out


def summarize(tables: dict) -> None:
    for name, df in tables.items():
        counts = df.groupby("SEASON").size()
        print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} cols, "
              f"{counts.min()}-{counts.max()} players/season")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()
    summarize(NBADataCollector().fetch_all(force=args.force))
