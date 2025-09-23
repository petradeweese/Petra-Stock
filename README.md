# Petra Stock

This repository contains a FastAPI application and related tools for stock pattern analysis. It
provides a REST API and scheduled jobs to compute and store trading pattern statistics.

## Overview

The service wraps the `pattern_finder_app` engine and exposes endpoints for running scans against
specified tickers. Results are stored in a local SQLite database and may be queried or archived for
later review. A scheduler can periodically run scans while the market is open.

## Installation

Install the Python dependencies using pip:

```bash
pip install -r requirements.txt
```

## Running

Start the FastAPI app with Uvicorn after installing dependencies:

```bash
uvicorn app:app --reload
```

## Usage

Once the server is running, open `http://localhost:8000` to access the API and web interface. Use
the favorites and scheduler features to configure recurring scans. For development, run the test
suite with:

```bash
pytest
```

## Scanning

During a scan, the application verifies that price data covers at least 95% of the
expected bars in the lookback window. For intraday intervals, expected bars are
derived from actual market sessions: only minutes when the market is open are
counted. Weekends and holidays therefore no longer inflate the coverage
requirement.

## Favorites alerts

Saved favorites now emit real-time alerts whenever the underlying scan detects a
new entry signal. The detection logic runs in the shared row-finalization hook
used by the UI, autoscan batch runner, and background scheduler so alerting
behaves consistently regardless of how the scan was triggered. Each delivery is
deduplicated by `(favorite_id, bar_time)` to avoid spamming repeated signals.

Alerts can be delivered over email using your existing SMTP configuration or via
Twilio-powered MMS when a mobile workflow is preferred. When MMS is enabled the
system attempts to send the same formatted body to every number listed in
`ALERT_SMS_TO`, counting the alert as delivered once any recipient succeeds.

The 15-minute scheduler loop now calls the autoscan batch so alerts fire during
regular market hours without manual intervention. The job runs at :00, :15, :30
and :45 (with a small guard to prevent duplicate triggers) as long as the New
York Stock Exchange is open.

## Environment

Copy `.env.example` to `.env` and adjust:

- `SCHWAB_CLIENT_ID` – Schwab application client ID
- `SCHWAB_CLIENT_SECRET` – Schwab application client secret
- `SCHWAB_REDIRECT_URI` – registered redirect URI for the Schwab app
- `SCHWAB_ACCOUNT_ID` – (optional) account ID to scope requests
- `SCHWAB_REFRESH_TOKEN` – long-lived refresh token used to mint access tokens
- `DATABASE_URL` – database connection string
- `SCHWAB_INCLUDE_PREPOST` – include pre/post market bars when true
- `DB_CACHE_TTL` – in-process DB cache TTL
- `CLAMP_MARKET_CLOSED` – clamp backfills to last market close (default true)
- `BACKFILL_CHUNK_DAYS` – days per backfill slice (default 1)
- `FETCH_RETRY_MAX` – max retry attempts for Schwab fetches (default 4)
- `FETCH_RETRY_BASE_MS` – base backoff in milliseconds (default 300)
- `FETCH_RETRY_CAP_MS` – maximum backoff in milliseconds (default 5000)
- `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` – Twilio
  credentials and sender number for MMS alerts
- `ALERT_SMS_TO` – comma-separated list of phone numbers that should receive
  MMS alerts

The Schwab-backed provider automatically falls back to yfinance if Schwab
returns an error, times out, or yields no data. Logs include the provider used
for each fetch to simplify debugging.

## Backfill and ETL

Populate the database using the backfill script which resumes from checkpoints:

```bash
python scripts/backfill_data.py symbols.txt
```

Nightly ETL can keep data fresh:

```bash
python scripts/nightly_etl.py symbols.txt
```

Run a quick end-to-end smoke test (with optional fallback simulation):

```bash
python scripts/smoke_data.py [--force-fallback]
```

## Gap Fill

Run the nightly ETL to heal recent gaps or fetch missing bars manually using
`detect_gaps` and `services.data_provider.fetch_bars`.

## Rotate API Key

Update the Schwab OAuth credentials (`SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`,
`SCHWAB_REDIRECT_URI`, `SCHWAB_ACCOUNT_ID`, and `SCHWAB_REFRESH_TOKEN`) in your
environment or `.env` file and restart the process.

## Changelog

- Add market-closed clamp, chunked backfill and bounded provider retries.

## Deployment Notes

Set any new environment variables as needed and restart the `petra` service.
