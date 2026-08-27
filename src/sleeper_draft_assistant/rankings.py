from __future__ import annotations

from datetime import date
from pathlib import Path

import nflreadpy
import polars as pl

from .models import Player, normalize_position


POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def load_consensus_board(cache_dir: Path, refresh: bool = False) -> list[Player]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"consensus-{date.today().isoformat()}.parquet"
    if cache_file.exists() and not refresh:
        board = pl.read_parquet(cache_file)
    else:
        try:
            rankings = nflreadpy.load_ff_rankings("draft").filter(
                (pl.col("page_type") == "redraft-overall")
                & pl.col("pos").is_in(POSITIONS)
                & pl.col("ecr").is_not_null()
            )
            player_ids = nflreadpy.load_ff_playerids().select(
                pl.col("fantasypros_id").cast(pl.Int64, strict=False).alias("id"),
                pl.col("sleeper_id").cast(pl.String),
            )
            board = (
                rankings.join(player_ids, on="id", how="left")
                .with_columns(
                    pl.when(pl.col("pos") == "DST")
                    .then(pl.col("tm"))
                    .otherwise(pl.col("sleeper_id"))
                    .alias("sleeper_id"),
                    pl.col("ecr").cast(pl.Float64),
                    pl.col("sd").cast(pl.Float64, strict=False),
                )
                .filter(pl.col("sleeper_id").is_not_null())
                .sort("ecr")
                .unique(subset=["sleeper_id"], keep="first", maintain_order=True)
                .select(
                    "sleeper_id",
                    "player",
                    "pos",
                    "team",
                    "tm",
                    "ecr",
                    "sd",
                    "bye",
                    "scrape_date",
                )
            )
            board.write_parquet(cache_file)
        except Exception:
            previous_caches = sorted(cache_dir.glob("consensus-*.parquet"), reverse=True)
            if not previous_caches:
                raise
            board = pl.read_parquet(previous_caches[0])

    players: list[Player] = []
    for row in board.iter_rows(named=True):
        ecr = float(row["ecr"])
        expert_sd = float(row["sd"]) if row["sd"] is not None else 0.0
        players.append(
            Player(
                sleeper_id=str(row["sleeper_id"]),
                name=str(row["player"]),
                position=normalize_position(str(row["pos"])),
                team=str(row.get("team") or row.get("tm") or "FA"),
                ecr=ecr,
                uncertainty=max(expert_sd, 1.0 + 0.03 * ecr),
                bye=int(row["bye"]) if row.get("bye") is not None else None,
            )
        )
    return players
