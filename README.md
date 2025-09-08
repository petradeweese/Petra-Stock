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
