# Petra Stock

Petra Stock provides a web and desktop toolkit for scanning equities for recurring options ROI patterns. The project combines a FastAPI backend with an optional Tkinter GUI to analyze historical price data, apply decision-tree models, and surface candidates that meet user-defined hit-rate and drawdown thresholds.

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API**
   ```bash
   uvicorn app:app --reload
   ```
   Access the scanner at http://localhost:8000/scanner.

## Key Modules

- `app.py` – FastAPI application serving the web scanner, favorites archive, and settings pages.
- `pattern_finder_app.py` – Desktop GUI that mirrors the scanning logic for local, interactive use.
- `indices.py` – Ticker lists for the S&P 100 and Top 150 universes used during scans.
- `templates/` – Jinja2 templates and `static/` assets for the web interface.

## Running Scans

After launching the server, open the `/scanner` page and choose a scan type:

- **Top 150** – scans the tickers defined in `indices.TOP150`.
- **S&P 100** – scans the `indices.SP100` universe.
- **Single Ticker** – analyze a specific ticker symbol.

Adjust parameters such as interval, direction, target and stop percentages, then submit to run the scan. Results include historical ROI metrics and can be saved to favorites. The same scans can be executed via the desktop GUI:

```bash
python pattern_finder_app.py
```
