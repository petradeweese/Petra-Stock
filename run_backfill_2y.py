import datetime as dt
from pathlib import Path
# Load helpers straight from the script file
import importlib.util, sys

spec = importlib.util.spec_from_file_location("backfill_polygon", "scripts/backfill_polygon.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["backfill_polygon"] = mod
spec.loader.exec_module(mod)  # gives us mod.backfill and mod.load_symbols

symbols = mod.load_symbols(Path("symbols.txt"))
end = dt.datetime.now(tz=dt.timezone.utc)
start = end - dt.timedelta(days=365*2)   # ~2 years

# use_checkpoint=False to force going through the whole list now
mod.backfill(symbols, dry_run=False, start=start, end=end, use_checkpoint=True)
