# Market Data Backfill and ETL Runbook

## Backfill
- `python scripts/backfill_polygon.py SYMBOLS.txt`
- Progress is checkpointed to `backfill_checkpoint.json` so reruns resume.
- Use `--dry-run` to estimate time/requests without inserting.
- To re-ingest a symbol/day, remove rows from `bars_15m` and rerun with `--symbol` or `--start`/`--end`.

The backfill and nightly jobs honor the Polygon rate limit specified via
`POLY_RPS` and `POLY_BURST` (default 0.08 rps and burst 1 – 5 requests/minute).
Adjust these env vars if your plan changes.

## Nightly ETL
- `python scripts/nightly_etl.py`
- Runs at 20:15 ET by default and heals gaps for last 3 days.

## Rotate API Key
- Update `POLYGON_API_KEY` in environment or `.env` and restart processes.

## Rollback Provider
- Set `provider` to `yahoo` or `db` in config to disable Polygon.
