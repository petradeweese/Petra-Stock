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

## Environment

Copy `.env.example` to `.env` and adjust:

- `POLYGON_API_KEY` – API key for Polygon.io
- `DATABASE_URL` – database connection string
- `POLY_RPS` / `POLY_BURST` – Polygon rate limit settings
- `DB_CACHE_TTL` – in-process DB cache TTL
- `POLYGON_INCLUDE_PREPOST` – include pre/post market bars when true
- `CLAMP_MARKET_CLOSED` – clamp backfills to last market close (default true)
- `BACKFILL_CHUNK_DAYS` – days per backfill slice (default 1)
- `FETCH_RETRY_MAX` – max retry attempts for Polygon fetches (default 4)
- `FETCH_RETRY_BASE_MS` – base backoff in milliseconds (default 300)
- `FETCH_RETRY_CAP_MS` – maximum backoff in milliseconds (default 5000)

## Backfill and ETL

Populate the database using the backfill script which resumes from checkpoints:

```bash
python scripts/backfill_polygon.py symbols.txt
```

Nightly ETL can keep data fresh:

```bash
python scripts/nightly_etl.py symbols.txt
```

## Gap Fill

Run the nightly ETL to heal recent gaps or fetch missing bars manually using
`detect_gaps` and `fetch_polygon_prices`.

## Rotate API Key

Update `POLYGON_API_KEY` in your environment or `.env` file and restart the process.

## Changelog

- Add market-closed clamp, chunked backfill and bounded Polygon retries.

## Deployment Notes

Set any new environment variables as needed and restart the `petra` service.
