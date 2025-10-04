# Market Data Backfill and ETL Runbook

## Backfill
- `python scripts/backfill_data.py SYMBOLS.txt`
- Progress is checkpointed to `backfill_checkpoint.json` so reruns resume.
- Use `--dry-run` to estimate time/requests without inserting.
- To re-ingest a symbol/day, remove rows from `bars` and rerun with `--symbol` or `--start`/`--end`.

The backfill and nightly jobs use the Schwab data provider. Ensure
`SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`, `SCHWAB_REDIRECT_URI`,
`SCHWAB_ACCOUNT_ID` and `SCHWAB_REFRESH_TOKEN` are configured and rotate them
as needed.

## Nightly ETL
- `python scripts/nightly_etl.py`
- Runs at 20:15 ET by default and heals gaps for last 3 days.

## Gap Fill
- Use `services.price_store.detect_gaps` to identify missing bars.
- Fetch via `services.data_provider.fetch_bars` and `upsert_bars` to heal.

## Intraday Scan / Paper Trading Source
- Intraday requests call `services.data_provider.fetch_range` which serves
  historical bars from SQLite and only hits Schwab for the most recent ~60 days.
- Any bars fetched from Schwab are persisted via `INSERT OR REPLACE` so repeated
  scans and paper trades are DB-only once the window is warm.
- Daily (and higher) intervals are also persisted for consistency even though
  they continue to source from Schwab directly.

## Rotate API Key
- Rotate Schwab OAuth credentials (client ID/secret and refresh token) and
  update the corresponding environment variables before restarting processes.

## Rollback Provider
- Set `provider` to `yahoo` or `db` in config to disable Schwab usage.

## Favorites alerts
- Alerts only fire when the scan row contains an entry/detection event; stop or
  timeout transitions are ignored.
- Deliveries are deduped by `(favorite_id, bar_time)` and are marked sent only
  after at least one channel succeeds.
- Ensure the environment includes Twilio credentials and `ALERT_SMS_TO` when
  using MMS delivery. Missing configuration leaves MMS disabled without
  affecting email alerts.
- The minute scheduler now triggers the autoscan batch every 15 minutes while
  XNYS is open so alerts dispatch automatically during the trading session.
