# pattern_finder_app.py
# Options ROI Pattern Finder — Walk-Forward + Regimes + Events (macOS/Windows)
# - ROI regressor tree with walk-forward stability scoring
# - Regime gates (SPY trend, VIX z)
# - Direction-aware: UP/DOWN/BOTH, asymmetric exits
# - Costs: slippage, theta/day, vega penalty (scaled to move) — GUI knob
# - Candlestick + MA-cross + time-of-day features
# - Events CSV guardrail (optional)
# - Favorites (direction-aware) + forward test + Favorites tab
# - S&P100 + Top150 scanners with filters (min Hit%, max DD%) and parallelism
# - Auto-Scanner tab: sweeps gates while locking Target/Stop/Within/Hit%/Lookback/Max DD; save to Favorites
# - Alerts tab: SMTP settings, Send Test Email, Run Now, Start/Stop daily scheduler (Mon–Fri)
# - Diagnostics: shows gated bar count so you can tune filters

import os, json, math, warnings, threading, itertools, time, random
import platform
warnings.filterwarnings("ignore")

# Tkinter is only required for the optional desktop GUI.  The server
# environment used by the web API runs headless and does not ship with the
# tkinter module.  Import it lazily and fall back to stubs when unavailable so
# that the core scanning functions remain importable.
try:  # pragma: no cover - optional dependency
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception:  # ModuleNotFoundError on headless servers
    tk = None  # type: ignore
    ttk = messagebox = filedialog = None  # type: ignore

import pandas as pd
import numpy as np
import yfinance as yf
import httpx

from services.price_utils import normalize_price_df, DataUnavailableError

from email.message import EmailMessage
from datetime import datetime, timedelta

from utils import now_et
import smtplib, ssl, certifi

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit

from multiprocessing import Pool, cpu_count
from indices import SP100, TOP150, TOP250  # Index lists

# Trust store for TLS using certifi when running on macOS
if platform.system() == "Darwin":
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# -----------------------------
# Files & Cache
# -----------------------------
FAV_FILE = "favorites.json"
ALERTS_FILE = "alerts_config.json"    # SMTP + recipients + schedule
ALERTS_SENT_FILE = "alerts_sent.json" # daily dedupe (e.g. {"2025-09-06": ["ENPH_15m_UP", ...]})

# Cache results for 10 minutes to avoid re-downloading the same series.
_CACHE: dict = {}
_CACHE_TTL = 600  # seconds


def _cget(key):
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, val = entry
    if time.time() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return val


def _cset(key, val) -> None:
    _CACHE[key] = (time.time(), val)


# Shared HTTPX client for price downloads.  Using a single client enables
# connection pooling, HTTP/2 and higher concurrency when fetching data for
# many tickers.
_HTTP = httpx.Client(
    http2=True,
    timeout=10.0,
    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32),
)

# -----------------------------
# Utilities
# -----------------------------
def _interval_to_minutes(s: str) -> int:
    if s.endswith("m"): return int(s[:-1])
    if s.endswith("h"): return 60*int(s[:-1])
    if s == "1d": return 24*60
    return 60

def _bars_for_window(window_val: float, window_unit: str, interval: str) -> int:
    mbar = _interval_to_minutes(interval)
    if interval == "1d":
        if window_unit == "Days": return max(1, int(math.ceil(window_val)))
        return max(1, int(math.ceil(window_val/6.5)))
    total = window_val*60 if window_unit=="Hours" else window_val*6.5*60
    return max(1, int(math.ceil(total/mbar)))


def wilson_lb95(hits: int, n: int) -> float:
    """Wilson 95% lower bound for binomial proportion."""
    if n <= 0:
        return 0.0
    z = 1.96
    p = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lb = (center - margin) / denom
    return max(0.0, min(1.0, lb))

def _download_prices(ticker: str, interval: str, lookback_years: float) -> pd.DataFrame:
    def _normalize_ohlcv(records):
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df.index = pd.to_datetime(df["timestamp"], unit="s")
        df = df.drop(columns=["timestamp"])  # already converted to index
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        if "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "Adj Close"})
        return df.dropna(how="all")

    key = ("PX", ticker, interval, round(lookback_years, 2))
    cached = _cget(key)
    if cached is not None:
        return cached.copy()

    intraday_caps_days = {"1m":7,"2m":60,"5m":60,"10m":60,"15m":60,"30m":60,"60m":730,"90m":60}
    if interval in intraday_caps_days:
        period = f"{intraday_caps_days[interval]}d"
        yf_int = "5m" if interval == "10m" else interval
    else:
        period = f"{min(int(round(lookback_years*365)), 1825)}d"
        yf_int = "1d"

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={yf_int}&range={period}"
        "&includePrePost=false&events=div%2Csplit"
    )

    data = None
    for attempt in range(3):
        try:
            resp = _HTTP.get(url)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception:  # pragma: no cover - network failure
            if attempt == 2:
                raise
            time.sleep(0.5 + random.random())

    result = (data or {}).get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"No usable price data for {ticker}")
    result0 = result[0]
    ts = result0.get("timestamp", [])
    quote = result0.get("indicators", {}).get("quote", [{}])[0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])
    adj_raw = result0.get("indicators", {}).get("adjclose", [{}])[0]
    adj_list = adj_raw.get("adjclose", []) if isinstance(adj_raw, dict) else adj_raw

    records = []
    for i, t in enumerate(ts):
        records.append({
            "timestamp": t,
            "open": opens[i] if i < len(opens) else None,
            "high": highs[i] if i < len(highs) else None,
            "low": lows[i] if i < len(lows) else None,
            "close": closes[i] if i < len(closes) else None,
            "volume": volumes[i] if i < len(volumes) else None,
            "adjclose": adj_list[i] if i < len(adj_list) else None,
        })

    df = _normalize_ohlcv(records)

    if interval == "10m" and not df.empty and "Close" in df.columns:
        o = df["Open"].resample("10T").first()
        h = df["High"].resample("10T").max()
        l = df["Low"].resample("10T").min()
        c = df["Close"].resample("10T").last()
        v = df["Volume"].resample("10T").sum() if "Volume" in df.columns else None
        df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
        if v is not None:
            df["Volume"] = v
        df = df.dropna(how="any")

    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No usable price data for {ticker} at interval {interval} (period={period}).")

    df = df.sort_index()
    _cset(key, df.copy())
    return df

def _download_market_refs(interval: str, lookback_years: float):
    import pandas as pd
    import yfinance as yf

    def _normalize_ohlcv(df, ticker=None):
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            sym = ticker if (ticker is not None and ticker in set(lvl0)) else lvl0[0]
            try:
                df = df.xs(sym, axis=1, level=0, drop_level=True)
            except Exception:
                df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={c: c.title() for c in df.columns})
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df

    key = ("MKT", interval, round(lookback_years, 2))
    cached = _cget(key)
    if cached is not None:
        spy, vix = cached
        return spy.copy(), vix.copy()

    intraday_caps_days = {"1m": 7, "2m": 60, "5m": 60, "10m": 60, "15m": 60, "30m": 60, "60m": 730, "90m": 60}
    if interval in intraday_caps_days:
        period = f"{intraday_caps_days[interval]}d"
        yf_int = "5m" if interval == "10m" else interval
    else:
        period = f"{min(int(round(lookback_years * 365)), 1825)}d"
        yf_int = "1d"

    spy = yf.download("SPY", period=period, interval=yf_int,
                      group_by=None, auto_adjust=False, progress=False, threads=True)
    vix = yf.download("^VIX", period=period, interval="1d",
                      group_by=None, auto_adjust=False, progress=False, threads=True)

    spy = _normalize_ohlcv(spy, ticker="SPY").sort_index()
    vix = _normalize_ohlcv(vix, ticker="^VIX").sort_index()

    spy = spy[~spy.index.duplicated(keep="last")]
    vix = vix[~vix.index.duplicated(keep="last")]

    if spy.empty or "Close" not in spy.columns:
        raise RuntimeError("Failed to download SPY reference data.")
    if vix.empty or "Close" not in vix.columns:
        raise RuntimeError("Failed to download VIX reference data.")

    _cset(key, (spy.copy(), vix.copy()))
    return spy, vix

# -----------------------------
# Features
# -----------------------------
def _candles(df, suf="_b"):
    O,H,L,C = df["Open"], df["High"], df["Low"], df["Close"]
    Cp, Op = C.shift(1), O.shift(1)
    rng = (H-L).replace(0,np.nan); body = (C-O).abs()
    upw = (H-np.maximum(C,O)).clip(lower=0); dnw = (np.minimum(C,O)-L).clip(lower=0)
    out = pd.DataFrame(index=df.index)
    out[f"CS_BODY{suf}"] = body/rng
    out[f"CS_UP{suf}"] = upw/rng
    out[f"CS_DN{suf}"] = dnw/rng
    out[f"CS_DIR{suf}"] = np.sign(C-O)
    out[f"CS_GAP{suf}"] = (O/Cp - 1.0)
    doji  = (body <= 0.10*rng) & rng.notna()
    hammer= (dnw >= 2.0*body) & (upw <= body)
    star  = (upw >= 2.0*body) & (dnw <= body)
    bull  = (Cp>Op)&(C>O)&(O<=Cp)&(C>=Op)
    bear  = (Cp<Op)&(C<O)&(O>=Cp)&(C<=Op)
    out[f"CS_DOJI{suf}"]=doji.astype(int)
    out[f"CS_HAMMER{suf}"]=hammer.astype(int)
    out[f"CS_STAR{suf}"]=star.astype(int)
    out[f"CS_BULL{suf}"]=bull.astype(int)
    out[f"CS_BEAR{suf}"]=bear.astype(int)
    return out

def _ma_cross(C, suf="_b"):
    out = pd.DataFrame(index=C.index)
    ema20 = EMAIndicator(C, window=20).ema_indicator()
    ema50 = EMAIndicator(C, window=50).ema_indicator()
    ema200= EMAIndicator(C, window=200).ema_indicator()
    out[f"EMA20_GT50{suf}"] = (ema20>ema50).astype(int)
    out[f"EMA50_GT200{suf}"] = (ema50>ema200).astype(int)
    out[f"DIST_EMA200{suf}"] = (C-ema200)/ema200.replace(0,np.nan)
    out[f"EMA50_SLOPE{suf}"] = ema50.pct_change(5)
    return out

def _tod(df, interval, suf="_b"):
    out = pd.DataFrame(index=df.index)
    if hasattr(df.index, "hour"):
        hr = df.index.hour; mn = df.index.minute
        mins = hr*60+mn
        out[f"TOD_HOUR{suf}"] = hr
        out[f"TOD_MIN{suf}"] = mn
        out[f"TOD_FIRST{suf}"] = ((mins>=570)&(mins<630)).astype(int)   # 9:30–10:30 ET
        out[f"TOD_LAST{suf}"]  = ((mins>=900)&(mins<=960)).astype(int)  # 15:00–16:00
    else:
        out[f"TOD_HOUR{suf}"]=0; out[f"TOD_MIN{suf}"]=0; out[f"TOD_FIRST{suf}"]=0; out[f"TOD_LAST{suf}"]=0
    return out

def build_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    H,L,C,O,V = df["High"], df["Low"], df["Close"], df["Open"], df["Volume"]
    X = pd.DataFrame(index=df.index)
    for w in [5,10,20,50,100,200]: X[f"SMA_{w}_b"] = C.rolling(w).mean()
    for w in [5,8,12,20,26,34,50,100,200]:
        try: X[f"EMA_{w}_b"] = EMAIndicator(C, window=w).ema_indicator()
        except: pass
    for w in [20,50,200]:
        if f"EMA_{w}_b" in X: X[f"DIST_EMA{w}_b"] = (C-X[f"EMA_{w}_b"])/X[f"EMA_{w}_b"].replace(0,np.nan)
    for w in [7,14,21]:
        try: X[f"RSI_{w}_b"] = RSIIndicator(C, window=w).rsi()
        except: pass
    try:
        m = MACD(C, 26, 12, 9)
        X["MACD_b"] = m.macd(); X["MACD_SIG_b"]=m.macd_signal(); X["MACD_HIST_b"]=m.macd_diff()
    except: pass
    prev = C.shift(1)
    tr = pd.concat([(H-L).abs(), (H-prev).abs(), (L-prev).abs()], axis=1).max(axis=1)
    X["ATR_14_b"] = tr.rolling(14).mean()
    X["ATR_Z_b"]  = (X["ATR_14_b"] - X["ATR_14_b"].rolling(100).mean()) / X["ATR_14_b"].rolling(100).std()
    X["RANGE_PCT_b"] = (H-L)/C.replace(0,np.nan)
    for w in [10,20,50]:
        ma = V.rolling(w).mean(); sd = V.rolling(w).std()
        X[f"VOL_MA_{w}_b"]=ma; X[f"VOL_Z_{w}_b"]=(V-ma)/sd.replace(0,np.nan)

    cs,_mc,_td = _candles(df), _ma_cross(C), _tod(df, interval)
    X[cs.columns]=cs; X[_mc.columns]=_mc; X[_td.columns]=_td
    X = X.shift(1).apply(lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
    return X.dropna(axis=1, how="all")

# -----------------------------
# Regime features
# -----------------------------
def build_regime_features(index, interval, lookback_years):
    spy, vix = _download_market_refs(interval, lookback_years)

    spy = spy[~spy.index.duplicated(keep="last")]
    vix = vix[~vix.index.duplicated(keep="last")]

    def _align_daily_to_intraday(daily_close: pd.Series, intraday_index: pd.DatetimeIndex) -> pd.Series:
        s = daily_close.copy()
        s.index = pd.to_datetime(s.index.date)
        s = s.groupby(s.index).last().sort_index()
        intraday_dates = pd.to_datetime(intraday_index.date)
        left = pd.DataFrame({"date": intraday_dates, "_idx": intraday_index})
        right = pd.DataFrame({"date": s.index, "val": s.values})
        out = pd.merge_asof(
            left.sort_values("date"),
            right.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        out = out.set_index("_idx")["val"].sort_index().ffill()
        return out

    spy_close = _align_daily_to_intraday(spy["Close"], index)
    vix_close = _align_daily_to_intraday(vix["Close"], index)

    df = pd.DataFrame(index=index)
    ema20 = spy_close.ewm(span=20, adjust=False).mean()
    ema50 = spy_close.ewm(span=50, adjust=False).mean()
    df["SPY_TREND_UP"] = (ema20 > ema50).astype(int)
    df["SPY_TREND_SLOPE"] = ema50.pct_change(10)

    vma = vix_close.rolling(60).mean()
    vsd = vix_close.rolling(60).std()
    df["VIX_Z"] = (vix_close - vma) / vsd.replace(0, np.nan)
    return df

# -----------------------------
# Events
# -----------------------------
def load_event_dates_from_csv(path: str) -> set:
    if not path or not os.path.exists(path): return set()
    try:
        df = pd.read_csv(path)
        col = "date" if "date" in df.columns else df.columns[0]
        return set(pd.to_datetime(df[col]).dt.date)
    except Exception:
        return set()

def build_event_mask(index: pd.DatetimeIndex, dates: set) -> pd.Series:
    if not dates: return pd.Series(True, index=index)
    return ~pd.to_datetime(index.date).isin(dates)

# -----------------------------
# Payoff model (greeks-ish)
# -----------------------------
def _theta_per_bar(interval: str, theta_per_day: float) -> float:
    if theta_per_day<=0: return 0.0
    return theta_per_day * (_interval_to_minutes(interval)/390.0)

def _dir_return(entry: float, px: float, direction: str) -> float:
    return (px/entry - 1.0) if direction=="UP" else (entry/px - 1.0)

def _per_trade_metrics(prices: pd.Series, start_idx: int, k_bars: int,
                       target_pos: float, stop_pos: float, max_tt: int,
                       interval: str, entry_next_open: bool,
                       delta: float, theta_bar: float, vix_z: float,
                       direction: str, slippage_pct: float, vega_scale: float):
    if entry_next_open:
        if start_idx+1 >= len(prices):
            return {"hit":0,"tt":np.nan,"dd":np.nan,"roi":0.0,"outcome":"timeout"}
        entry = prices.iloc[start_idx+1]; first = start_idx+1
    else:
        entry = prices.iloc[start_idx]; first = start_idx

    end = min(first+k_bars, len(prices)-1)
    if end <= first:
        return {"hit":0,"tt":np.nan,"dd":np.nan,"roi":0.0,"outcome":"timeout"}

    max_fav = -1e9; worst = 0.0; t_hit=None; t_stop=None
    for t in range(first+1, end+1):
        m = _dir_return(entry, prices.iloc[t], direction)
        max_fav = max(max_fav, m); worst = min(worst, m)
        hit_triggered = m >= abs(target_pos)
        stop_triggered = m <= -abs(stop_pos)
        if hit_triggered and t_hit is None:
            t_hit = t-first
        if stop_triggered and t_stop is None:
            t_stop = t-first
        if hit_triggered or stop_triggered:
            break

    hit_first = (t_hit is not None) and (t_stop is None or t_hit <= t_stop)
    bars = t_hit if hit_first else (t_stop if t_stop is not None else (end-first))
    favorable = abs(target_pos) if hit_first else max(0.0, max_fav)
    base = delta * favorable
    time_cost = theta_bar * max(0, bars)
    vega_cost = abs(vix_z) * vega_scale * favorable
    roi = base - time_cost - vega_cost - slippage_pct

    hit = 1 if (hit_first and bars <= max_tt) else 0
    if hit:
        outcome = "hit"
    elif t_stop is not None and not hit_first:
        outcome = "stop"
    else:
        outcome = "timeout"
    return {"hit":hit, "tt":(t_hit if t_hit is not None else np.nan),
            "dd":abs(worst), "roi":roi, "outcome":outcome}

# -----------------------------
# Tree helpers
# -----------------------------
def _paths_to_leaves(tree, feature_names):
    t = tree.tree_; L,R,F,T = t.children_left, t.children_right, t.feature, t.threshold
    out, stk = {}, [(0,[])]
    while stk:
        nid, conds = stk.pop()
        if L[nid]==-1 and R[nid]==-1:
            out[nid]=conds; continue
        f,thr = F[nid], T[nid]
        fname = feature_names[f] if f>=0 else f"f{f}"
        stk.append((L[nid], conds+[(fname,"<=",thr)]))
        stk.append((R[nid], conds+[(fname,">", thr)]))
    return out

def _fmt_rule(conds, default="Always trade"):
    if not conds: return " • " + default
    parts=[]
    for f,op,th in conds:
        sval = f"{th:,.0f}" if abs(th)>=1000 else f"{th:.2f}"
        parts.append(f"{f} {op} {sval}")
    return " • " + "  &  ".join(parts)

# -----------------------------
# Core analysis
# -----------------------------
def analyze_roi_mode(ticker, interval, direction,
                     target_pct, stop_pct,
                     window_value, window_unit,
                     lookback_years, max_tt_bars, min_support,
                     delta_assumed, theta_per_day_pct,
                     atrz_gate, slope_gate_pct,
                     use_regime, regime_trend_only, vix_z_max,
                     event_mask, slippage_bps, vega_scale,
                     ccp_grid=(0.0,0.0005), depth_grid=(4,5,6)):

    raw = _download_prices(ticker, interval, lookback_years)
    raw = normalize_price_df(raw)
    if raw is None:
        raise DataUnavailableError(f"{ticker} no_close_or_empty")
    prices = raw["Close"].copy()
    X = build_features(raw, interval)
    regime = build_regime_features(X.index, interval, lookback_years)
    X = X.join(regime, how="left").ffill()
    if "VIX_Z" in X.columns:
        X["VIX_Z"] = X["VIX_Z"].fillna(0.0)

    common = X.index.intersection(prices.index)
    X, prices = X.loc[common], prices.loc[common]
    if len(X) < 400: raise ValueError(f"{ticker}: not enough aligned data.")

    # Gates
    gate = pd.Series(True, index=X.index)
    if "ATR_Z_b" in X: gate &= (X["ATR_Z_b"] >= atrz_gate)
    if "EMA50_SLOPE_b" in X: gate &= (X["EMA50_SLOPE_b"] >= (slope_gate_pct/100.0))
    if use_regime:
        if regime_trend_only:
            gate &= (X["SPY_TREND_UP"]==1) if direction=="UP" else (X["SPY_TREND_UP"]==0)
        if vix_z_max>0: gate &= (X["VIX_Z"].abs() <= vix_z_max)
    if event_mask is not None and len(event_mask)==len(gate):
        gate &= event_mask.loc[gate.index]
    # Diagnostics
    diag = f"[Diag] Tradable bars after gates: {int(gate.sum())}/{len(gate)}"

    k_bars = _bars_for_window(window_value, window_unit, interval)
    theta_bar = _theta_per_bar(interval, theta_per_day_pct/100.0)
    slippage = slippage_bps/10000.0
    delta = float(delta_assumed)

    N = len(X); i_tr, i_te = int(N*0.6), int(N*0.8)
    X_tr, X_te, X_fw = X.iloc[:i_tr], X.iloc[i_tr:i_te], X.iloc[i_te:]
    G_tr, G_te = gate.loc[X_tr.index], gate.loc[X_te.index]

    idx_pos = {ts:i for i,ts in enumerate(prices.index)}

    def _roi_series(idxs):
        res=[]
        for ts in idxs:
            if not gate.loc[ts]: res.append(0.0); continue
            p = idx_pos.get(ts); 
            if p is None or p >= len(prices)-2: res.append(0.0); continue
            vix_z = (float(X.loc[ts,"VIX_Z"]) if ("VIX_Z" in X and pd.notna(X.loc[ts,"VIX_Z"])) else 0.0)
            m = _per_trade_metrics(prices, p, k_bars,
                                   abs(target_pct/100.0), abs(stop_pct/100.0),
                                   max_tt_bars, interval, False,
                                   delta, theta_bar, vix_z, direction, slippage, vega_scale)
            res.append(m["roi"])
        return pd.Series(res, index=idxs)

    y_tr = _roi_series(X_tr.index)

    def _tscv_score(Xt, yt, model):
        tscv = TimeSeriesSplit(n_splits=3); scr=[]
        for tr,va in tscv.split(Xt):
            model.fit(Xt.iloc[tr], yt.iloc[tr])
            scr.append(float(np.mean(model.predict(Xt.iloc[va]))))
        return float(np.mean(scr)) if scr else -1e9

    best, best_s = None, -1e9
    for d in depth_grid:
        for leaf in [max(20, len(X_tr)//80), max(40, len(X_tr)//50)]:
            for ccp in ccp_grid:
                mdl = DecisionTreeRegressor(max_depth=d, min_samples_leaf=leaf, ccp_alpha=ccp, random_state=42)
                s = _tscv_score(X_tr, y_tr, mdl)
                if s > best_s: best, best_s = mdl, s
    tree = best.fit(X_tr, y_tr)
    feats = list(X.columns); paths = _paths_to_leaves(tree, feats)

    def _eval_slice(Xs, Gs):
        rows_map={}
        for ts, nid in zip(Xs.index, tree.apply(Xs)):
            if not Gs.loc[ts]: continue
            p = idx_pos.get(ts); 
            if p is None or p >= len(prices)-2: continue
            vix_z = (float(X.loc[ts,"VIX_Z"]) if ("VIX_Z" in X and pd.notna(X.loc[ts,"VIX_Z"])) else 0.0)
            m = _per_trade_metrics(prices, p, k_bars,
                                   abs(target_pct/100.0), abs(stop_pct/100.0),
                                   max_tt_bars, interval, False,
                                   delta, theta_bar, vix_z, direction, slippage, vega_scale)
            rows_map.setdefault(nid, []).append(m)
        rows=[]
        for nid,lst in rows_map.items():
            if not lst:
                continue
            supp = len(lst)
            hits = sum(1 for x in lst if x.get("outcome") == "hit")
            stops = sum(1 for x in lst if x.get("outcome") == "stop")
            timeouts = sum(1 for x in lst if x.get("outcome") == "timeout")
            hit_rate = hits / supp if supp else 0.0
            stop_pct = stops / supp if supp else 0.0
            timeout_pct = timeouts / supp if supp else 0.0
            avg_tt = float(np.nanmean([x["tt"] for x in lst]))
            avg_dd = float(np.mean([x["dd"] for x in lst]))
            roi_vals = [x["roi"] for x in lst]
            avg_roi = float(np.mean(roi_vals))
            roi_std = float(np.std(roi_vals)) if len(roi_vals) > 1 else 0.0
            sharpe = avg_roi / roi_std if roi_std > 1e-9 else 0.0
            rows.append({
                "node_id": int(nid),
                "support": supp,
                "hit_rate": hit_rate,
                "hit_lb95": wilson_lb95(hits, supp),
                "stop_pct": stop_pct,
                "timeout_pct": timeout_pct,
                "avg_tt": avg_tt,
                "avg_dd": avg_dd,
                "avg_roi": avg_roi,
                "sharpe": sharpe,
                "rule": _fmt_rule(paths.get(nid, [])),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if "hit_lb95" in df.columns:
            df["hit_lb95"] = df["hit_lb95"].fillna(0.0)
        else:
            df["hit_lb95"] = 0.0
        return df.sort_values(["hit_lb95", "avg_roi", "support"], ascending=[False, False, False]).reset_index(drop=True)

    df_te = _eval_slice(X_te, G_te)

    def _wf_score(splits=3):
        tscv = TimeSeriesSplit(n_splits=splits); vals=[]
        for tr,va in tscv.split(X):
            Xt, Xv = X.iloc[tr], X.iloc[va]
            mdl = DecisionTreeRegressor(**tree.get_params()).fit(Xt, _roi_series(Xt.index))
            vals.append(float(np.mean(mdl.predict(Xv))))
        return float(np.mean(vals)) if vals else 0.0

    stability = _wf_score(3)

    def _filter(df):
        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    "direction",
                    "node_id",
                    "support",
                    "hit_rate",
                    "hit_lb95",
                    "stop_pct",
                    "timeout_pct",
                    "avg_tt",
                    "avg_dd",
                    "avg_roi",
                    "sharpe",
                    "rule",
                    "stability",
                    "diag",
                ]
            )
        df = df[df["support"]>=min_support].copy()
        if df.empty:
            df["diag"]=diag; return df
        df.insert(0,"direction",direction)
        df["stability"]=stability; df["diag"]=diag
        return df

    return {"tree":tree,"features":feats,"paths":paths,"k_bars":k_bars,"prices":prices,
            "idx_pos":{ts:i for i,ts in enumerate(prices.index)},"stability":stability}, _filter(df_te), \
           {"X_forward":X_fw, "gate":gate.loc[X_fw.index]}

# -----------------------------
# Parallel scanner
# -----------------------------
def _scan_worker(args):
    tkr, cfg = args
    try:
        model, df, _ = analyze_roi_mode(
            ticker=tkr, interval=cfg["interval"], direction=cfg["direction"],
            target_pct=cfg["target_pct"], stop_pct=cfg["stop_pct"],
            window_value=cfg["window_value"], window_unit=cfg["window_unit"],
            lookback_years=cfg["lookback_years"], max_tt_bars=cfg["max_tt_bars"],
            min_support=cfg["min_support"], delta_assumed=cfg["delta_assumed"],
            theta_per_day_pct=cfg["theta_per_day_pct"], atrz_gate=cfg["atrz_gate"],
            slope_gate_pct=cfg["slope_gate_pct"], use_regime=cfg["use_regime"],
            regime_trend_only=cfg["regime_trend_only"], vix_z_max=cfg["vix_z_max"],
            event_mask=cfg["event_mask_dict"].get(tkr, cfg["default_event_mask"]),
            slippage_bps=cfg["slippage_bps"], vega_scale=cfg["vega_scale"],
            ccp_grid=(0.0,0.0005), depth_grid=(4,5)
        )
        if df is None or df.empty: return None
        df = df[(df["hit_rate"]*100.0 >= cfg["scan_min_hit"])
                & (df["avg_dd"]*100.0 <= cfg["scan_max_dd"])]
        if df.empty: return None
        r = df.iloc[0]
        return {
            "ticker": tkr,
            "direction": r["direction"],
            "avg_roi": float(r["avg_roi"]),
            "avg_roi_pct": r["avg_roi"] * 100.0,
            "hit_pct": r["hit_rate"] * 100.0,
            "hit_lb95": float(r.get("hit_lb95", 0.0)),
            "support": int(r["support"]),
            "avg_tt": r["avg_tt"],
            "avg_dd_pct": r["avg_dd"] * 100.0,
            "stop_pct": float(r.get("stop_pct", 0.0)),
            "timeout_pct": float(r.get("timeout_pct", 0.0)),
            "stability": r["stability"],
            "sharpe": r.get("sharpe", 0.0),
            "rule": r["rule"],
        }
    except Exception:
        return None

def scan_parallel(tickers, cfg, max_workers=None):
    max_workers = max_workers or max(1, min(cpu_count()-1, 8))
    args = [(t, cfg) for t in tickers]
    out=[]
    with Pool(processes=max_workers) as pool:
        for res in pool.imap_unordered(_scan_worker, args):
            if res is not None: out.append(res)
    if not out:
        return pd.DataFrame(
            columns=[
                "ticker",
                "direction",
                "avg_roi",
                "avg_roi_pct",
                "hit_pct",
                "hit_lb95",
                "support",
                "avg_tt",
                "avg_dd_pct",
                "stop_pct",
                "timeout_pct",
                "stability",
                "sharpe",
                "rule",
            ]
        )
    df = pd.DataFrame(out)
    if "hit_lb95" in df.columns:
        df["hit_lb95"] = df["hit_lb95"].fillna(0.0)
    else:
        df["hit_lb95"] = 0.0
    return df.sort_values(["hit_lb95", "avg_roi", "support"], ascending=[False, False, False]).reset_index(drop=True)
from concurrent.futures import ThreadPoolExecutor, as_completed

def scan_parallel_threaded(tickers, cfg, max_workers=None):
    """
    Threaded variant: avoids spawn-related BrokenPipe issues when running
    many back-to-back scans from the Auto-Scanner.
    """
    tasks = [(t, cfg) for t in tickers]
    out = []
    max_workers = max_workers or min(16, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_scan_worker, args) for args in tasks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res is not None:
                    out.append(res)
            except Exception:
                pass
    if not out:
        return pd.DataFrame(
            columns=[
                "ticker",
                "direction",
                "avg_roi",
                "avg_roi_pct",
                "hit_pct",
                "hit_lb95",
                "support",
                "avg_tt",
                "avg_dd_pct",
                "stop_pct",
                "timeout_pct",
                "stability",
                "sharpe",
                "rule",
            ]
        )
    df = pd.DataFrame(out)
    if "hit_lb95" in df.columns:
        df["hit_lb95"] = df["hit_lb95"].fillna(0.0)
    else:
        df["hit_lb95"] = 0.0
    return df.sort_values(["hit_lb95", "avg_roi", "support"], ascending=[False, False, False]).reset_index(drop=True)

# -----------------------------
# GUI
# -----------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Options ROI Pattern Finder — Walk-Forward + Regimes + Events")
        root.geometry("1440x920")

        # scheduler handle
        self._alert_timer = None

        f = ttk.Frame(root, padding=10); f.pack(fill="x")
        # Top inputs
        ttk.Label(f,text="Ticker:").grid(row=0,column=0,sticky="w")
        self.ticker = tk.StringVar(value="TSLA"); ttk.Entry(f,textvariable=self.ticker,width=10).grid(row=0,column=1,padx=6)

        ttk.Label(f,text="Interval:").grid(row=0,column=2,sticky="w")
        self.interval=tk.StringVar(value="15m")
        ttk.Combobox(f,textvariable=self.interval,values=["5m","10m","15m","30m","1h","1d"],state="readonly",width=8).grid(row=0,column=3,padx=6)

        ttk.Label(f,text="Direction:").grid(row=0,column=4,sticky="w")
        self.direction=tk.StringVar(value="BOTH")
        ttk.Combobox(f,textvariable=self.direction,values=["UP","DOWN","BOTH"],state="readonly",width=8).grid(row=0,column=5,padx=6)

        ttk.Label(f,text="Target %:").grid(row=1,column=0,sticky="w")
        self.target=tk.DoubleVar(value=1.0); ttk.Entry(f,textvariable=self.target,width=8).grid(row=1,column=1,padx=6)

        ttk.Label(f,text="Stop %:").grid(row=1,column=2,sticky="w")
        self.stop=tk.DoubleVar(value=0.5); ttk.Entry(f,textvariable=self.stop,width=8).grid(row=1,column=3,padx=6)

        ttk.Label(f,text="Within:").grid(row=1,column=4,sticky="w")
        self.win_val=tk.DoubleVar(value=4.0); ttk.Entry(f,textvariable=self.win_val,width=8).grid(row=1,column=5,padx=6)
        self.win_unit=tk.StringVar(value="Hours")
        ttk.Combobox(f,textvariable=self.win_unit,values=["Hours","Days"],state="readonly",width=8).grid(row=1,column=6,padx=6)

        ttk.Label(f,text="Max TT (bars):").grid(row=0,column=6,sticky="w")
        self.max_tt=tk.IntVar(value=12); ttk.Entry(f,textvariable=self.max_tt,width=8).grid(row=0,column=7,padx=6)

        ttk.Label(f,text="Lookback (years):").grid(row=0,column=8,sticky="w")
        self.lookback=tk.DoubleVar(value=2.0); ttk.Entry(f,textvariable=self.lookback,width=8).grid(row=0,column=9,padx=6)

        ttk.Label(f,text="Delta (assumed):").grid(row=2,column=0,sticky="w")
        self.delta=tk.DoubleVar(value=0.40); ttk.Entry(f,textvariable=self.delta,width=8).grid(row=2,column=1,padx=6)

        ttk.Label(f,text="Theta per day %:").grid(row=2,column=2,sticky="w")
        self.theta=tk.DoubleVar(value=0.20); ttk.Entry(f,textvariable=self.theta,width=8).grid(row=2,column=3,padx=6)

        ttk.Label(f,text="Slippage (bps):").grid(row=2,column=4,sticky="w")
        self.slip=tk.DoubleVar(value=7); ttk.Entry(f,textvariable=self.slip,width=8).grid(row=2,column=5,padx=6)

        ttk.Label(f,text="Vega scale:").grid(row=2,column=6,sticky="w")
        self.vega_scale=tk.DoubleVar(value=0.03); ttk.Entry(f,textvariable=self.vega_scale,width=8).grid(row=2,column=7,padx=6)

        ttk.Label(f,text="ATR Z ≥").grid(row=3,column=0,sticky="w")
        self.atrz=tk.DoubleVar(value=0.10); ttk.Entry(f,textvariable=self.atrz,width=8).grid(row=3,column=1,padx=6)

        ttk.Label(f,text="EMA50 slope ≥ %").grid(row=3,column=2,sticky="w")
        self.slope=tk.DoubleVar(value=0.02); ttk.Entry(f,textvariable=self.slope,width=8).grid(row=3,column=3,padx=6)

        self.use_regime=tk.BooleanVar(value=False)
        ttk.Checkbutton(f,text="Use Regime Filter",variable=self.use_regime).grid(row=3,column=4,sticky="w")
        self.trend_only=tk.BooleanVar(value=False)
        ttk.Checkbutton(f,text="Match SPY Trend to Direction",variable=self.trend_only).grid(row=3,column=5,sticky="w")

        ttk.Label(f,text="|VIX z| ≤").grid(row=3,column=6,sticky="w")
        self.vix_z=tk.DoubleVar(value=3.0); ttk.Entry(f,textvariable=self.vix_z,width=8).grid(row=3,column=7,padx=6)

        ttk.Label(f,text="Events CSV (optional):").grid(row=4,column=0,sticky="w")
        self.events=tk.StringVar(value="")
        ttk.Entry(f,textvariable=self.events,width=40).grid(row=4,column=1,columnspan=3,sticky="we",padx=6)
        ttk.Button(f,text="Browse…",command=self._pick_events).grid(row=4,column=4,padx=6)

        ttk.Label(f,text="Min support:").grid(row=4,column=5,sticky="w")
        self.min_support=tk.IntVar(value=20); ttk.Entry(f,textvariable=self.min_support,width=8).grid(row=4,column=6,padx=6)

        ttk.Label(f,text="Scanner min Hit%:").grid(row=4,column=7,sticky="w")
        self.scan_min_hit=tk.DoubleVar(value=0.0); ttk.Entry(f,textvariable=self.scan_min_hit,width=8).grid(row=4,column=8,padx=6)

        ttk.Label(f,text="Scanner max DD%:").grid(row=4,column=9,sticky="w")
        self.scan_max_dd=tk.DoubleVar(value=0.50); ttk.Entry(f,textvariable=self.scan_max_dd,width=8).grid(row=4,column=10,padx=6)

        ttk.Button(f,text="Run",command=self.on_run).grid(row=0,column=11,padx=10)
        ttk.Button(f,text="Save Selected Favorite",command=self.save_favorite).grid(row=2,column=11,padx=10)
        ttk.Button(f,text="Forward Test Favorites",command=self.forward_test_favorites).grid(row=3,column=11,padx=10)
        ttk.Button(f,text="Scan S&P100",command=lambda:self.on_scan("sp")).grid(row=5,column=0,pady=6,sticky="w")
        ttk.Button(f,text="Scan Top150",command=lambda:self.on_scan("top")).grid(row=5,column=1,pady=6,sticky="w")

        for c in range(0,12): f.grid_columnconfigure(c, pad=4)

        nb = ttk.Notebook(root); nb.pack(fill="both",expand=True,padx=10,pady=10)
        self.tab_rules=ttk.Frame(nb); self.tab_sp=ttk.Frame(nb); self.tab_top=ttk.Frame(nb)
        nb.add(self.tab_rules,text="Rules (This Ticker)")
        nb.add(self.tab_sp,text="S&P100 Scanner")
        nb.add(self.tab_top,text="Top150 Scanner")

        # --- Auto-Scanner tab ---
        self.tab_auto = ttk.Frame(nb)
        nb.add(self.tab_auto, text="Auto-Scanner")

        auto_cols = (
            "rank",
            "ticker",
            "direction",
            "avg_roi",
            "hit_pct",
            "hit_lb95",
            "support",
            "stop_pct",
            "timeout_pct",
            "avg_tt",
            "avg_dd",
            "stability",
            "atrz",
            "slope",
            "reg",
            "trend",
            "vix",
            "rule",
        )
        self.auto_tree = ttk.Treeview(self.tab_auto, columns=auto_cols, show="headings", height=22)
        for c, w in [
            ("rank", 60),
            ("ticker", 80),
            ("direction", 80),
            ("avg_roi", 100),
            ("hit_pct", 100),
            ("hit_lb95", 110),
            ("support", 80),
            ("stop_pct", 110),
            ("timeout_pct", 110),
            ("avg_tt", 80),
            ("avg_dd", 110),
            ("stability", 90),
            ("atrz", 70),
            ("slope", 70),
            ("reg", 60),
            ("trend", 70),
            ("vix", 70),
            ("rule", 820),
        ]:
            self.auto_tree.heading(c, text=c.upper()); self.auto_tree.column(c, width=w, anchor="w")
        self.auto_tree.pack(fill="both", expand=True, padx=6, pady=(6,0))

        auto_btns = ttk.Frame(self.tab_auto); auto_btns.pack(anchor="w", padx=6, pady=6)
        ttk.Button(auto_btns, text="Run Auto Top150", command=self.on_auto_scan_top150).pack(side="left", padx=4)
        ttk.Button(auto_btns, text="Save Selected to Favorites", command=self.save_autoscan_to_favorites).pack(side="left", padx=8)

        # storage for autoscan results and row->favorite mapping
        self.auto_results = None
        self._auto_item_to_fav = {}

        # --- Favorites tab ---
        self.tab_favs = ttk.Frame(nb)
        nb.add(self.tab_favs, text="Favorites")

        fav_cols = ("rank","ticker","direction","interval","target","stop","window","lookback","support_min","rule")
        self.fav_tree = ttk.Treeview(self.tab_favs, columns=fav_cols, show="headings", height=16)
        fav_widths = [60,80,80,80,70,70,120,90,100,900]
        fav_headers = ["RANK","TICKER","DIRECTION","INTERVAL","TARGET%","STOP%","WITHIN","LOOKBACK_YRS","MIN_SUPPORT","RULE"]
        for c,w,h in zip(fav_cols, fav_widths, fav_headers):
            self.fav_tree.heading(c, text=h); self.fav_tree.column(c, width=w, anchor="w")
        self.fav_tree.pack(fill="both", expand=True, padx=6, pady=(6,0))

        res_cols = ("rank","ticker","direction","now","signals_today","last_signal_time","note")
        self.fav_res = ttk.Treeview(self.tab_favs, columns=res_cols, show="headings", height=8)
        for c,w,h in [
            ("rank",60,"RANK"),("ticker",80,"TICKER"),("direction",80,"DIR"),
            ("now",80,"NOW?"),("signals_today",120,"SIGNALS_TODAY"),
            ("last_signal_time",180,"LAST_SIGNAL_TIME"),("note",900,"NOTE")
        ]:
            self.fav_res.heading(c, text=h); self.fav_res.column(c, width=w, anchor="w")
        self.fav_res.pack(fill="both", expand=True, padx=6, pady=(6,0))

        fav_btns = ttk.Frame(self.tab_favs); fav_btns.pack(anchor="w", padx=6, pady=6)
        ttk.Button(fav_btns, text="Refresh Favorites", command=self.show_favorites_tab).pack(side="left", padx=4)
        ttk.Button(fav_btns, text="Delete Selected", command=self.delete_selected_favorite).pack(side="left", padx=4)
        ttk.Button(fav_btns, text="Run Favorites Today", command=self.run_favorites_today).pack(side="left", padx=12)

        # --- Alerts tab ---
        self.tab_alerts = ttk.Frame(nb)
        nb.add(self.tab_alerts, text="Alerts")

        a = ttk.Frame(self.tab_alerts, padding=8); a.pack(fill="x")
        ttk.Label(a, text="From (Gmail):").grid(row=0, column=0, sticky="w")
        self.smtp_user = tk.StringVar(value="")
        ttk.Entry(a, textvariable=self.smtp_user, width=32).grid(row=0, column=1, padx=6)

        ttk.Label(a, text="App password:").grid(row=0, column=2, sticky="w")
        self.smtp_pass = tk.StringVar(value="")
        ttk.Entry(a, textvariable=self.smtp_pass, width=22, show="•").grid(row=0, column=3, padx=6)

        ttk.Label(a, text="To (comma-separated):").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.smtp_to = tk.StringVar(value="")
        ttk.Entry(a, textvariable=self.smtp_to, width=70).grid(row=1, column=1, columnspan=3, sticky="we", padx=6, pady=(6,0))

        ttk.Label(a, text="Daily time (HH:MM, local):").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.alert_time = tk.StringVar(value="09:28")
        ttk.Entry(a, textvariable=self.alert_time, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=(6,0))

        self.alert_weekdays = tk.BooleanVar(value=True)
        ttk.Checkbutton(a, text="Weekdays only (Mon–Fri)", variable=self.alert_weekdays).grid(row=2, column=2, sticky="w")

        btns = ttk.Frame(self.tab_alerts, padding=(8,0)); btns.pack(anchor="w")
        ttk.Button(btns, text="Save Alert Settings", command=self._alerts_save_from_ui).pack(side="left", padx=4)
        ttk.Button(btns, text="Send Test Email", command=self._send_test_email).pack(side="left", padx=4)
        ttk.Button(btns, text="Run Favorites & Email Now", command=self._alerts_run_once).pack(side="left", padx=12)
        ttk.Button(btns, text="Start Daily", command=self._start_alert_scheduler).pack(side="left", padx=4)
        ttk.Button(btns, text="Stop Daily", command=self._stop_alert_scheduler).pack(side="left", padx=4)

        self.alert_status = tk.StringVar(value="Alerts idle.")
        ttk.Label(self.tab_alerts, textvariable=self.alert_status).pack(anchor="w", padx=12, pady=6)

        cols=(
            "rank",
            "direction",
            "avg_roi",
            "hit_pct",
            "hit_lb95",
            "support",
            "stop_pct",
            "timeout_pct",
            "avg_tt",
            "avg_dd",
            "stability",
            "rule",
            "node_id",
            "diag",
        )
        self.tree=ttk.Treeview(self.tab_rules,columns=cols,show="headings",height=24)
        widths=[60,80,100,100,110,80,110,110,80,110,90,840,80,250]
        for (c,w) in zip(cols,widths):
            self.tree.heading(c,text=c.upper()); self.tree.column(c,width=w,anchor="w")
        self.tree.pack(fill="both",expand=True)

        scan_cols=(
            "rank",
            "ticker",
            "direction",
            "avg_roi",
            "hit_pct",
            "hit_lb95",
            "support",
            "stop_pct",
            "timeout_pct",
            "avg_tt",
            "avg_dd",
            "stability",
            "rule",
        )
        self.scan_sp=ttk.Treeview(self.tab_sp,columns=scan_cols,show="headings",height=24)
        self.scan_top=ttk.Treeview(self.tab_top,columns=scan_cols,show="headings",height=24)
        for tv in [self.scan_sp,self.scan_top]:
            for c,w in [
                ("rank",60),
                ("ticker",80),
                ("direction",80),
                ("avg_roi",100),
                ("hit_pct",100),
                ("hit_lb95",110),
                ("support",80),
                ("stop_pct",110),
                ("timeout_pct",110),
                ("avg_tt",80),
                ("avg_dd",110),
                ("stability",90),
                ("rule",820),
            ]:
                tv.heading(c,text=c.upper()); tv.column(c,width=w,anchor="w")
            tv.pack(fill="both",expand=True)
        ttk.Button(self.tab_sp,text="Export CSV",command=lambda:self.export_scan_csv("sp")).pack(anchor="w",padx=6,pady=6)
        ttk.Button(self.tab_top,text="Export CSV",command=lambda:self.export_scan_csv("top")).pack(anchor="w",padx=6,pady=6)

        self.status=tk.StringVar(value="Ready."); ttk.Label(root,textvariable=self.status).pack(anchor="w",padx=12,pady=(0,10))

        self.last_context=None; self.last_sp=None; self.last_top=None

        # Auto-refresh when switching tabs
        def _on_tab_changed(evt):
            try:
                tab_text = nb.tab(nb.select(), "text")
                if tab_text == "Favorites":
                    self.show_favorites_tab()
                elif tab_text == "Alerts":
                    self._alerts_load_to_ui()
            except:
                pass
        nb.bind("<<NotebookTabChanged>>", _on_tab_changed)

        # Load alerts config on boot
        self._alerts_load_to_ui()
        self.show_favorites_tab()

    # ---------- Favorites tab helpers ----------

    def show_favorites_tab(self):
        favs = self._load_favs()
        for r in self.fav_tree.get_children(): self.fav_tree.delete(r)
        for i, f in enumerate(favs, start=1):
            self.fav_tree.insert(
                "", "end",
                values=(
                    i,
                    f.get("ticker", "?"),
                    f.get("direction", "UP"),
                    f.get("interval", "15m"),
                    f.get("target_pct", 1.0),
                    f.get("stop_pct", 0.5),
                    f"{f.get('window_value', 4)} {f.get('window_unit', 'Hours')}",
                    f.get("lookback_years", 0.2),
                    f.get("min_support", 20),
                    f.get("rule", ""),
                ),
            )
        self.status.set(f"{len(favs)} favorite(s) loaded.")

    def delete_selected_favorite(self):
        sel = self.fav_tree.selection()
        if not sel:
            messagebox.showinfo("Favorites", "Select rows to delete.")
            return
        idxs = sorted([int(self.fav_tree.item(iid, "values")[0]) - 1 for iid in sel], reverse=True)
        favs = self._load_favs()
        for ix in idxs:
            if 0 <= ix < len(favs):
                favs.pop(ix)
        self._save_favs(favs)
        self.show_favorites_tab()
        self.status.set("Deleted selected favorite(s).")

    def _parse_rule(self, s: str):
        s = (s or "").replace("•", "").strip()
        if not s: return []
        parts = [p.strip() for p in s.split("&")]
        out = []
        for p in parts:
            if "<=" in p:
                f, thr = p.split("<=", 1); op = "<="
            elif ">" in p:
                f, thr = p.split(">", 1); op = ">"
            else:
                continue
            try:
                val = float(str(thr).strip().replace(",", ""))
                out.append((f.strip(), op, val))
            except:
                pass
        return out

    def _build_gates(self, X: pd.DataFrame, fav: dict) -> pd.Series:
        gate = pd.Series(True, index=X.index)
        if "ATR_Z_b" in X.columns:
            gate &= (X["ATR_Z_b"] >= float(fav.get("atrz", 0.0)))
        if "EMA50_SLOPE_b" in X.columns:
            gate &= (X["EMA50_SLOPE_b"] >= float(fav.get("slope", 0.0)) / 100.0)
        if bool(fav.get("use_regime", False)):
            if bool(fav.get("trend_only", False)) and "SPY_TREND_UP" in X.columns:
                want_up = (fav.get("direction", "UP") == "UP")
                gate &= (X["SPY_TREND_UP"] == (1 if want_up else 0))
            vz = float(fav.get("vix_z_max", 0.0))
            if vz > 0 and "VIX_Z" in X.columns:
                gate &= (X["VIX_Z"].abs() <= vz)
        return gate

    # ---------- Favorites evaluation (used by Alerts & UI) ----------
    def _evaluate_favorites_today(self):
        """
        Evaluate every saved favorite on fresh data and return:
        [{'ticker','direction','fav','now', 'signals_today','last_ts'}]
        """
        rows = []
        favs = self._load_favs()
        if not favs:
            return rows

        groups = {}
        for f in favs:
            key = (f.get("ticker", "?").upper(), f.get("interval", "15m"))
            groups.setdefault(key, []).append(f)

        today = pd.Timestamp.now().date()

        for (tkr, iv), fav_list in groups.items():
            lookback = float(max(0.08, min(0.5, fav_list[0].get("lookback_years", 0.2))))
            try:
                raw = _download_prices(tkr, iv, lookback)
                raw = raw[~raw.index.duplicated(keep="last")]
                if raw.empty:
                    for f in fav_list:
                        rows.append({"ticker": tkr, "direction": f.get("direction", "UP"),
                                     "fav": f, "now": False, "signals_today": 0, "last_ts": None})
                    continue
                Xb = build_features(raw, iv)
                reg = build_regime_features(Xb.index, iv, lookback)
                X = Xb.join(reg, how="left").ffill()
                if X.empty:
                    for f in fav_list:
                        rows.append({"ticker": tkr, "direction": f.get("direction", "UP"),
                                     "fav": f, "now": False, "signals_today": 0, "last_ts": None})
                    continue
                for f in fav_list:
                    conds = self._parse_rule(f.get("rule", ""))
                    mask = pd.Series(True, index=X.index)
                    for feat, op, thr in conds:
                        if feat not in X.columns:
                            mask &= False
                        else:
                            mask &= (X[feat] <= thr) if op == "<=" else (X[feat] > thr)
                    G = self._build_gates(X, f)
                    sel = (mask & G)
                    sel_today = sel[pd.to_datetime(sel.index.date) == today]
                    signals_today = int(sel_today.sum())
                    last_ts = sel[sel].index.max() if sel.any() else None
                    now_flag = bool(len(sel) and sel.iloc[-1])
                    rows.append({
                        "ticker": tkr,
                        "direction": f.get("direction", "UP"),
                        "fav": f,
                        "now": now_flag,
                        "signals_today": signals_today,
                        "last_ts": None if last_ts is None else str(last_ts),
                    })
            except Exception:
                for f in fav_list:
                    rows.append({"ticker": tkr, "direction": f.get("direction", "UP"),
                                 "fav": f, "now": False, "signals_today": 0, "last_ts": None})
        return rows

    def run_favorites_today(self):
        try:
            for r in self.fav_res.get_children():
                self.fav_res.delete(r)
            rows = self._evaluate_favorites_today()
            rank = 1
            for r in rows:
                note = ""
                if r["now"] and r["signals_today"] == 0:
                    note = "Signal just fired on latest bar."
                elif not r["now"] and r["signals_today"] == 0:
                    note = "No signal today yet."
                self.fav_res.insert(
                    "", "end",
                    values=(
                        rank,
                        r["ticker"],
                        r["direction"],
                        "YES" if r["now"] else "NO",
                        r["signals_today"],
                        "" if r["last_ts"] is None else r["last_ts"],
                        note,
                    ),
                )
                rank += 1
            self.status.set("Favorites check complete.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Favorites", f"{type(e).__name__}: {e}")
            self.status.set("Favorites run error.")

    def _pick_events(self):
        p=filedialog.askopenfilename(title="Select events CSV",filetypes=[("CSV","*.csv"),("All files","*.*")])
        if p: self.events.set(p)

    def _params(self):
        return dict(
            ticker=self.ticker.get().strip().upper(),
            interval=self.interval.get(),
            direction=self.direction.get(),
            target_pct=float(self.target.get()),
            stop_pct=float(self.stop.get()),
            window_value=float(self.win_val.get()),
            window_unit=self.win_unit.get(),
            lookback_years=float(self.lookback.get()),
            max_tt_bars=int(self.max_tt.get()),
            min_support=int(self.min_support.get()),
            delta_assumed=float(self.delta.get()),
            theta_per_day_pct=float(self.theta.get()),
            atrz_gate=float(self.atrz.get()),
            slope_gate_pct=float(self.slope.get()),
            use_regime=bool(self.use_regime.get()),
            regime_trend_only=bool(self.trend_only.get()),
            vix_z_max=float(self.vix_z.get()),
            slippage_bps=float(self.slip.get()),
            vega_scale=float(self.vega_scale.get())
        )

    def on_run(self):
        try:
            p = self._params()
            self.status.set("Running...")
            self.root.update_idletasks()

            px = _download_prices(p["ticker"], p["interval"], p["lookback_years"])
            ev_dates = load_event_dates_from_csv(self.events.get())
            ev_mask = build_event_mask(px.index, ev_dates)

            rules_all = []
            last_model = None
            last_forward = None
            dirs = ["UP", "DOWN"] if p["direction"] == "BOTH" else [p["direction"]]

            for d in dirs:
                model, df, fwd = analyze_roi_mode(
                    ticker=p["ticker"], interval=p["interval"], direction=d,
                    target_pct=p["target_pct"], stop_pct=p["stop_pct"],
                    window_value=p["window_value"], window_unit=p["window_unit"],
                    lookback_years=p["lookback_years"], max_tt_bars=p["max_tt_bars"],
                    min_support=p["min_support"], delta_assumed=p["delta_assumed"],
                    theta_per_day_pct=p["theta_per_day_pct"], atrz_gate=p["atrz_gate"],
                    slope_gate_pct=p["slope_gate_pct"], use_regime=p["use_regime"],
                    regime_trend_only=p["regime_trend_only"], vix_z_max=p["vix_z_max"],
                    event_mask=ev_mask,
                    slippage_bps=p["slippage_bps"], vega_scale=p["vega_scale"],
                )
                last_model, last_forward = model, fwd
                rules_all.append(df)

            out = pd.concat(rules_all, axis=0) if rules_all else pd.DataFrame()
            if not out.empty:
                if "hit_lb95" in out.columns:
                    out["hit_lb95"] = out["hit_lb95"].fillna(0.0)
                else:
                    out["hit_lb95"] = 0.0
                out = out.sort_values(
                    ["hit_lb95", "avg_roi", "support"],
                    ascending=[False, False, False]
                ).reset_index(drop=True)

            self._fill_rules(out)
            self.last_context = (last_model, out, last_forward, p)
            self.status.set(f"{p['ticker']}: {len(out)} rule(s).")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Run error", f"{type(e).__name__}: {e}")
            self.status.set("Error.")

    def _fill_rules(self, df):
        for r in self.tree.get_children(): self.tree.delete(r)
        if df is None or df.empty: return
        fmt = lambda x: f"{(0.0 if x is None or pd.isna(x) else float(x))*100:5.2f}%"
        def _frac(row, candidates):
            for col, scale in candidates:
                val = row.get(col)
                if val is not None and not pd.isna(val):
                    return float(val) / scale
            return 0.0
        for i,row in df.iterrows():
            row_vals = {
                "avg_roi": fmt(_frac(row, [("avg_roi", 1.0), ("avg_roi_pct", 100.0)])),
                "hit_pct": fmt(_frac(row, [("hit_rate", 1.0), ("hit_pct", 100.0)])),
                "hit_lb95": fmt(_frac(row, [("hit_lb95", 1.0)])),
                "stop_pct": fmt(_frac(row, [("stop_pct", 1.0)])),
                "timeout_pct": fmt(_frac(row, [("timeout_pct", 1.0)])),
                "avg_dd": fmt(_frac(row, [("avg_dd", 1.0), ("avg_dd_pct", 100.0)])),
            }
            stab_val = row.get("stability", 0)
            try:
                stability = f"{float(stab_val):.3f}"
            except (TypeError, ValueError):
                stability = str(stab_val) if stab_val is not None else "0.000"
            support_val = row.get("support", 0)
            support = 0 if support_val is None or pd.isna(support_val) else int(support_val)
            avg_tt_val = row.get("avg_tt")
            avg_tt = "—" if avg_tt_val is None or pd.isna(avg_tt_val) else f"{avg_tt_val:.1f}"
            avg_dd = row_vals["avg_dd"]
            node_val = row.get("node_id")
            node_id = "" if node_val is None or pd.isna(node_val) else int(node_val)
            self.tree.insert(
                "",
                "end",
                values=(
                    i + 1,
                    row.get("direction", "UP"),
                    row_vals["avg_roi"],
                    row_vals["hit_pct"],
                    row_vals["hit_lb95"],
                    support,
                    row_vals["stop_pct"],
                    row_vals["timeout_pct"],
                    avg_tt,
                    avg_dd,
                    stability,
                    row.get("rule", ""),
                    node_id,
                    row.get("diag", ""),
                ),
            )

    # Favorites persistence
    def _load_favs(self):
        if os.path.exists(FAV_FILE):
            try:
                with open(FAV_FILE,"r",encoding="utf-8") as f: return json.load(f)
            except: return []
        return []
    def _save_favs(self, favs):
        with open(FAV_FILE,"w",encoding="utf-8") as f: json.dump(favs,f,indent=2)

    def save_favorite(self):
        if self.last_context is None:
            messagebox.showinfo("Favorites","Run first, then select rows."); return
        model, df, fwd, p = self.last_context
        sel=self.tree.selection()
        if not sel:
            messagebox.showinfo("Favorites","Select one or more rows to save."); return
        favs=self._load_favs()
        columns = list(self.tree["columns"])
        try:
            node_idx = columns.index("node_id")
        except ValueError:
            messagebox.showerror("Favorites", "Unable to locate node_id column.")
            return
        for iid in sel:
            vals=self.tree.item(iid,"values")
            if node_idx >= len(vals):
                continue
            node_val = vals[node_idx]
            if node_val in ("", None):
                continue
            try:
                node_id=int(node_val)
            except (TypeError, ValueError):
                continue
            row=df[df["node_id"]==node_id]
            if row.empty: continue
            favs.append({
                "ticker":p["ticker"],"interval":p["interval"],
                "direction":row.iloc[0].get("direction",p["direction"]),
                "node_id":node_id,"rule":row.iloc[0]["rule"],
                "target_pct":p["target_pct"],"stop_pct":p["stop_pct"],
                "window_value":p["window_value"],"window_unit":p["window_unit"],
                "lookback_years":p["lookback_years"],"max_tt_bars":p["max_tt_bars"],
                "min_support":p["min_support"],"delta":p["delta_assumed"],
                "theta_day":p["theta_per_day_pct"],"atrz":p["atrz_gate"],
                "slope":p["slope_gate_pct"],"use_regime":p["use_regime"],
                "trend_only":p["regime_trend_only"],"vix_z_max":p["vix_z_max"],
                "slippage_bps":p["slippage_bps"],"vega_scale":p["vega_scale"],
                # store reference avg_dd for info in emails
                "ref_avg_dd": float(row.iloc[0]["avg_dd"]) if not pd.isna(row.iloc[0]["avg_dd"]) else None
            })
        self._save_favs(favs); self.status.set(f"Saved {len(sel)} favorite(s).")

    def forward_test_favorites(self):
        if self.last_context is None:
            messagebox.showinfo("Forward Test","Run first."); return
        model, df, fwd, p = self.last_context
        favs=self._load_favs()
        if not favs:
            messagebox.showinfo("Forward Test","No favorites saved."); return
        Xf, gatef = fwd["X_forward"], fwd["gate"]
        prices, paths, k_bars = model["prices"], model["paths"], model["k_bars"]
        idx_pos = model["idx_pos"]; theta_bar = _theta_per_bar(p["interval"], p["theta_per_day_pct"]/100.0)
        rows=[]
        for fav in favs:
            if fav["ticker"]!=p["ticker"] or fav["interval"]!=p["interval"]: continue
            conds=paths.get(fav["node_id"],[]); mask=pd.Series(True,index=Xf.index)
            for f,op,th in conds:
                if f not in Xf.columns: mask&=False; break
                mask &= (Xf[f]<=th) if op=="<=" else (Xf[f]>th)
            mask &= gatef
            trades=[]
            for ts in Xf.index[mask]:
                pos=idx_pos.get(ts); 
                if pos is None or pos >= len(prices)-2: continue
                vix_z=float(Xf.loc[ts,"VIX_Z"]) if "VIX_Z" in Xf.columns else 0.0
                m=_per_trade_metrics(prices,pos,k_bars,
                    abs(fav["target_pct"]/100.0),abs(fav["stop_pct"]/100.0),
                    fav["max_tt_bars"],p["interval"],False,
                    fav["delta"],theta_bar,vix_z,fav.get("direction","UP"),
                    fav.get("slippage_bps",7)/10000.0,fav.get("vega_scale",0.03))
                trades.append(m)
            if trades:
                supp = len(trades)
                hits = sum(1 for t in trades if t.get("outcome") == "hit")
                stops = sum(1 for t in trades if t.get("outcome") == "stop")
                timeouts = sum(1 for t in trades if t.get("outcome") == "timeout")
                rows.append({
                    "direction": fav.get("direction", "UP"),
                    "node_id": fav["node_id"],
                    "support": supp,
                    "hit_rate": hits / supp if supp else 0.0,
                    "hit_lb95": wilson_lb95(hits, supp),
                    "stop_pct": stops / supp if supp else 0.0,
                    "timeout_pct": timeouts / supp if supp else 0.0,
                    "avg_tt": float(np.nanmean([x["tt"] for x in trades])),
                    "avg_dd": float(np.mean([x["dd"] for x in trades])),
                    "avg_roi": float(np.mean([x["roi"] for x in trades])),
                    "rule": fav["rule"],
                    "stability": "—",
                    "diag": "forward",
                })
        out = pd.DataFrame(rows)
        if not out.empty:
            if "hit_lb95" in out.columns:
                out["hit_lb95"] = out["hit_lb95"].fillna(0.0)
            else:
                out["hit_lb95"] = 0.0
            out = out.sort_values(["hit_lb95", "avg_roi", "support"], ascending=[False, False, False]).reset_index(drop=True)
        self._fill_rules(out)

    # ---------- Scanner ----------
    def on_scan(self, which="sp"):
        try:
            p=self._params(); label = 'S&P100' if which=='sp' else 'Top250' if which=='top250' else 'Top150'
            self.status.set(f"Scanning {label}…"); self.root.update_idletasks()
            if which=="sp": tickers=SP100
            elif which=="top250": tickers=TOP250
            else: tickers=TOP150
            ev_dates=load_event_dates_from_csv(self.events.get())
            ev_masks={}
            for t in tickers:
                try: px=_download_prices(t,p["interval"],p["lookback_years"]); ev_masks[t]=build_event_mask(px.index,ev_dates)
                except Exception: ev_masks[t]=pd.Series(True,index=pd.DatetimeIndex([]))
            dirs=["UP","DOWN"] if p["direction"]=="BOTH" else [p["direction"]]
            out_all=[]
            for d in dirs:
                cfg=dict(interval=p["interval"],direction=d,target_pct=p["target_pct"],stop_pct=p["stop_pct"],
                         window_value=p["window_value"],window_unit=p["window_unit"],lookback_years=p["lookback_years"],
                         max_tt_bars=p["max_tt_bars"],min_support=p["min_support"],delta_assumed=p["delta_assumed"],
                         theta_per_day_pct=p["theta_per_day_pct"],atrz_gate=p["atrz_gate"],slope_gate_pct=p["slope_gate_pct"],
                         use_regime=p["use_regime"],regime_trend_only=p["regime_trend_only"],vix_z_max=p["vix_z_max"],
                         event_mask_dict=ev_masks,default_event_mask=pd.Series(True,index=pd.DatetimeIndex([])),
                         slippage_bps=p["slippage_bps"],vega_scale=p["vega_scale"],
                         scan_min_hit=float(self.scan_min_hit.get()),scan_max_dd=float(self.scan_max_dd.get())*100.0)
                out_all.append(scan_parallel(tickers,cfg))
            if out_all:
                out = pd.concat(out_all, axis=0)
                if "hit_lb95" in out.columns:
                    out["hit_lb95"] = out["hit_lb95"].fillna(0.0)
                else:
                    out["hit_lb95"] = 0.0
                out = out.sort_values(["hit_lb95", "avg_roi", "support"], ascending=[False, False, False]).reset_index(drop=True)
            else:
                out = pd.DataFrame()
            tv=self.scan_sp if which=="sp" else self.scan_top
            for r in tv.get_children(): tv.delete(r)
            if out is not None and not out.empty:
                fmt = lambda x: f"{(0.0 if x is None or pd.isna(x) else float(x))*100:5.2f}%"
                def _frac(row, candidates):
                    for col, scale in candidates:
                        val = row.get(col)
                        if val is not None and not pd.isna(val):
                            return float(val) / scale
                    return 0.0

                for i,row in out.iterrows():
                    row_vals = {
                        "avg_roi": fmt(_frac(row, [("avg_roi", 1.0), ("avg_roi_pct", 100.0)])),
                        "hit_pct": fmt(_frac(row, [("hit_rate", 1.0), ("hit_pct", 100.0)])),
                        "hit_lb95": fmt(_frac(row, [("hit_lb95", 1.0)])),
                        "stop_pct": fmt(_frac(row, [("stop_pct", 1.0)])),
                        "timeout_pct": fmt(_frac(row, [("timeout_pct", 1.0)])),
                        "avg_dd": fmt(_frac(row, [("avg_dd", 1.0), ("avg_dd_pct", 100.0)])),
                    }
                    support_val = row.get("support", 0)
                    support = 0 if support_val is None or pd.isna(support_val) else int(support_val)
                    avg_tt_val = row.get("avg_tt")
                    avg_tt = "—" if avg_tt_val is None or pd.isna(avg_tt_val) else f"{avg_tt_val:.1f}"
                    stab_val = row.get("stability", 0)
                    try:
                        stability = f"{float(stab_val):.3f}"
                    except (TypeError, ValueError):
                        stability = str(stab_val) if stab_val is not None else "0.000"
                    tv.insert(
                        "",
                        "end",
                        values=(
                            i + 1,
                            row.get("ticker", ""),
                            row.get("direction", "UP"),
                            row_vals["avg_roi"],
                            row_vals["hit_pct"],
                            row_vals["hit_lb95"],
                            support,
                            row_vals["stop_pct"],
                            row_vals["timeout_pct"],
                            avg_tt,
                            row_vals["avg_dd"],
                            stability,
                            row.get("rule", ""),
                        ),
                    )
            if which=="sp": self.last_sp=out
            else: self.last_top=out
            self.status.set(f"Scanner done: {len(out)} rows.")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Scan error", str(e))
            self.status.set("Scan error.")

    def export_scan_csv(self, which="sp"):
        df=self.last_sp if which=="sp" else self.last_top
        if df is None or df.empty:
            messagebox.showinfo("Export","No results to export."); return
        p=filedialog.asksaveasfilename(title="Save CSV",defaultextension=".csv",filetypes=[("CSV","*.csv")])
        if not p: return
        df.to_csv(p,index=False); messagebox.showinfo("Export",f"Saved to {p}")

    # ---------- Auto-Scanner ----------
    def on_auto_scan_top150(self):
        """
        Sweep a compact grid around current gates while LOCKING:
        Target, Stop, Within, Scanner Hit%, Lookback, Max DD.
        """
        try:
            # clear UI + mapping
            for r in self.auto_tree.get_children():
                self.auto_tree.delete(r)
            self._auto_item_to_fav.clear()

            p = self._params()
            locked = dict(
                interval=p["interval"],
                target_pct=p["target_pct"],
                stop_pct=p["stop_pct"],
                window_value=p["window_value"],
                window_unit=p["window_unit"],
                lookback_years=p["lookback_years"],
                max_tt_bars=p["max_tt_bars"],
                min_support=p["min_support"],
                delta_assumed=p["delta_assumed"],
                theta_per_day_pct=p["theta_per_day_pct"],
                slippage_bps=p["slippage_bps"],
                vega_scale=p["vega_scale"],
                scan_min_hit=float(self.scan_min_hit.get()),
                scan_max_dd=float(self.scan_max_dd.get()) * 100.0,
            )

            def _uniq(vals):
                out = sorted(set(round(float(x), 4) for x in vals))
                return [x for x in out if not (isinstance(x, (int, float)) and np.isnan(x))]

            atrz0  = float(p["atrz_gate"])
            slope0 = float(p["slope_gate_pct"])
            vix0   = float(p["vix_z_max"])

            atrz_list  = _uniq([atrz0 - 0.10, atrz0, atrz0 + 0.10])
            slope_list = _uniq([max(0.0, slope0 - 0.02), slope0, slope0 + 0.02])
            vix_list   = _uniq([max(0.0, vix0 - 1.0), vix0, min(4.0, vix0 + 1.0)])

            dirs = ["UP", "DOWN"] if self.direction.get() == "BOTH" else [self.direction.get()]
            rank = 1

            for direction in dirs:
                for atrz in atrz_list:
                    for slope in slope_list:
                        # ----- Pass 1: no-regime -----
                        base_cfg = dict(
                            interval=locked["interval"],
                            direction=direction,
                            target_pct=locked["target_pct"],
                            stop_pct=locked["stop_pct"],
                            window_value=locked["window_value"],
                            window_unit=locked["window_unit"],
                            lookback_years=locked["lookback_years"],
                            max_tt_bars=locked["max_tt_bars"],
                            min_support=locked["min_support"],
                            delta_assumed=locked["delta_assumed"],
                            theta_per_day_pct=locked["theta_per_day_pct"],
                            atrz_gate=atrz,
                            slope_gate_pct=slope,
                            use_regime=False,
                            regime_trend_only=False,
                            vix_z_max=0.0,
                            event_mask_dict={},
                            default_event_mask=pd.Series(True, index=pd.DatetimeIndex([])),
                            slippage_bps=locked["slippage_bps"],
                            vega_scale=locked["vega_scale"],
                            scan_min_hit=locked["scan_min_hit"],
                            scan_max_dd=locked["scan_max_dd"],
                        )

                        df = scan_parallel_threaded(TOP150, base_cfg, max_workers=16)
                        rank = self._auto_insert_results(df, direction, atrz, slope, False, False, 0.0, locked, rank)

                        # ----- Pass 2: regime ON (vary trend + VIX cap) -----
                        for trend_only in (False, True):
                            for vix_cap in vix_list:
                                cfg2 = dict(base_cfg)
                                cfg2.update(use_regime=True, regime_trend_only=trend_only, vix_z_max=vix_cap)
                                df = scan_parallel_threaded(TOP150, cfg2, max_workers=16)
                                rank = self._auto_insert_results(df, direction, atrz, slope, True, trend_only, vix_cap, locked, rank)

            self.status.set("Auto-Scanner completed.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Auto-Scanner", f"{type(e).__name__}: {e}")
            self.status.set("Auto-Scanner error.")

    def _auto_insert_results(self, df, direction, atrz, slope, use_reg, trend_only, vix_cap, locked, rank):
        """Insert autoscan rows and cache favorite payloads; return next rank."""
        if df is None or df.empty:
            return rank

        fmt = lambda x: f"{(0.0 if x is None or pd.isna(x) else float(x))*100:5.2f}%"
        def _frac(row, candidates):
            for col, scale in candidates:
                val = row.get(col)
                if val is not None and not pd.isna(val):
                    return float(val) / scale
            return 0.0

        for _, row in df.iterrows():
            row_vals = {
                "avg_roi": fmt(_frac(row, [("avg_roi", 1.0), ("avg_roi_pct", 100.0)])),
                "hit_pct": fmt(_frac(row, [("hit_rate", 1.0), ("hit_pct", 100.0)])),
                "hit_lb95": fmt(_frac(row, [("hit_lb95", 1.0)])),
                "stop_pct": fmt(_frac(row, [("stop_pct", 1.0)])),
                "timeout_pct": fmt(_frac(row, [("timeout_pct", 1.0)])),
                "avg_dd": fmt(_frac(row, [("avg_dd", 1.0), ("avg_dd_pct", 100.0)])),
            }
            support_val = row.get("support", 0)
            support = 0 if support_val is None or pd.isna(support_val) else int(support_val)
            avg_tt_val = row.get("avg_tt")
            avg_tt = "—" if avg_tt_val is None or pd.isna(avg_tt_val) else f"{avg_tt_val:.1f}"
            stab_val = row.get("stability", 0)
            try:
                stability = f"{float(stab_val):.3f}"
            except (TypeError, ValueError):
                stability = str(stab_val) if stab_val is not None else "0.000"
            iid = self.auto_tree.insert(
                "",
                "end",
                values=(
                    rank,
                    row.get("ticker", ""),
                    row.get("direction", direction),
                    row_vals["avg_roi"],
                    row_vals["hit_pct"],
                    row_vals["hit_lb95"],
                    support,
                    row_vals["stop_pct"],
                    row_vals["timeout_pct"],
                    avg_tt,
                    row_vals["avg_dd"],
                    stability,
                    f"{atrz:.2f}",
                    f"{slope:.2f}",
                    "Y" if use_reg else "N",
                    "Y" if trend_only else "N",
                    f"{vix_cap:.1f}",
                    row.get("rule", ""),
                ),
            )
            self._auto_item_to_fav[iid] = {
                "ticker": row["ticker"],
                "interval": locked["interval"],
                "direction": row.get("direction", direction),
                "node_id": -1,  # rule-only favorite
                "rule": row["rule"],
                "target_pct": locked["target_pct"],
                "stop_pct": locked["stop_pct"],
                "window_value": locked["window_value"],
                "window_unit": locked["window_unit"],
                "lookback_years": locked["lookback_years"],
                "max_tt_bars": locked["max_tt_bars"],
                "min_support": locked["min_support"],
                "delta": locked["delta_assumed"],
                "theta_day": locked["theta_per_day_pct"],
                "atrz": atrz,
                "slope": slope,
                "use_regime": use_reg,
                "trend_only": trend_only,
                "vix_z_max": vix_cap,
                "slippage_bps": locked["slippage_bps"],
                "vega_scale": locked["vega_scale"],
                "ref_avg_dd": float(row["avg_dd_pct"]) / 100.0 if pd.notna(row["avg_dd_pct"]) else None,
            }
            rank += 1

        return rank

    def save_autoscan_to_favorites(self):
        sel = self.auto_tree.selection()
        if not sel:
            messagebox.showinfo("Auto-Scanner", "Select one or more rows to save.")
            return
        favs = self._load_favs()
        added = 0
        for iid in sel:
            fav = self._auto_item_to_fav.get(iid)
            if fav:
                favs.append(fav)
                added += 1
        self._save_favs(favs)
        self.status.set(f"Saved {added} auto-scan result(s) to favorites.")
        self.show_favorites_tab()

    # ---------- Alerts helpers ----------
    def _alerts_load(self):
        if os.path.exists(ALERTS_FILE):
            try:
                with open(ALERTS_FILE,"r",encoding="utf-8") as f: return json.load(f)
            except: return {}
        return {}

    def _alerts_save(self, cfg):
        with open(ALERTS_FILE,"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2)

    def _alerts_load_to_ui(self):
        cfg = self._alerts_load()
        self.smtp_user.set(cfg.get("smtp_user",""))
        self.smtp_pass.set(cfg.get("smtp_pass",""))  # leave blank if none saved
        self.smtp_to.set(",".join(cfg.get("recipients",[])))
        self.alert_time.set(cfg.get("time","09:28"))
        self.alert_weekdays.set(bool(cfg.get("weekdays_only", True)))

    def _alerts_get_config_from_ui(self):
        recips = [x.strip() for x in self.smtp_to.get().split(",") if x.strip()]
        return {
            "smtp_user": self.smtp_user.get().strip(),
            "smtp_pass": self.smtp_pass.get().strip(),
            "recipients": recips,
            "time": self.alert_time.get().strip(),
            "weekdays_only": bool(self.alert_weekdays.get())
        }

    def _alerts_save_from_ui(self):
        cfg = self._alerts_get_config_from_ui()
        self._alerts_save(cfg)
        self.alert_status.set("Alert settings saved.")

    def _send_email(self, cfg, subject, body):
        if not cfg.get("smtp_user") or not cfg.get("smtp_pass"):
            raise RuntimeError("SMTP user/password missing.")
        if not cfg.get("recipients"):
            raise RuntimeError("Recipient list is empty.")

        # Build the message
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = cfg["smtp_user"].strip()
        msg["To"] = ", ".join(cfg["recipients"])
        msg.set_content(body)

        # Trust store for TLS using certifi to provide a consistent CA bundle
        ctx = ssl.create_default_context(cafile=certifi.where())

        # Gmail app passwords are shown with spaces — strip them out
        user = cfg["smtp_user"].strip()
        pwd = cfg["smtp_pass"].replace(" ", "").strip()

        # Try implicit TLS first, then fall back to STARTTLS
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
                server.login(user, pwd)
                server.send_message(msg)
        except ssl.SSLError:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.login(user, pwd)
                server.send_message(msg)

    def _compose_alert_email(self, rows):
        lines = []
        hdr = f"PatternFinder Alerts — {now_et().strftime('%Y-%m-%d %H:%M')}"
        lines.append(hdr)
        lines.append("="*len(hdr))
        any_yes = False
        for r in rows:
            if not r["now"]: continue
            f = r["fav"]
            stop = f.get("stop_pct", 0.5)
            ref_dd = f.get("ref_avg_dd")
            dd_part = f" | Ref Avg DD: {ref_dd*100:.2f}% " if isinstance(ref_dd,(int,float)) else ""
            lines.append(
                f"{r['ticker']} {r['direction']} {f.get('interval','15m')}  "
                f"NOW ✅  | Target: {f.get('target_pct',1.0):.2f}%  | Stop (DD cap): {stop:.2f}%{dd_part}| Rule:{f.get('rule','')}"
            )
            any_yes = True
        if not any_yes:
            lines.append("No favorites are signaling YES on the latest bar.")
        # Summary of activity today
        lines.append("")
        lines.append("Today’s signals count:")
        for r in rows:
            lines.append(f" - {r['ticker']} {r['direction']} {r['fav'].get('interval','15m')}: {r['signals_today']} | last: {'' if r['last_ts'] is None else r['last_ts']}")
        lines.append("")
        lines.append("— Sent by PatternFinder")
        return "\n".join(lines), any_yes

    def _send_test_email(self):
        cfg = self._alerts_get_config_from_ui()
        try:
            self._send_email(cfg, "PatternFinder: Test Email", "If you can read this, SMTP login worked. ✅")
            self.alert_status.set("Test email sent.")
            messagebox.showinfo("Alerts", "Test email sent.")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Alerts", f"Test email failed: {e}")
            self.alert_status.set("Test email failed.")

    def _load_sent_map(self):
        if os.path.exists(ALERTS_SENT_FILE):
            try:
                with open(ALERTS_SENT_FILE,"r",encoding="utf-8") as f:
                    return json.load(f)
            except: return {}
        return {}

    def _save_sent_map(self, m):
        with open(ALERTS_SENT_FILE,"w",encoding="utf-8") as f:
            json.dump(m,f,indent=2)

    def _alerts_run_once(self):
        try:
            cfg = self._alerts_get_config_from_ui()
            rows = self._evaluate_favorites_today()

            # Dedupe: don't email same (ticker,interval,dir) more than once per day
            sent = self._load_sent_map()
            today = now_et().strftime("%Y-%m-%d")
            already = set(sent.get(today, []))
            filtered = []
            for r in rows:
                key = f"{r['ticker']}_{r['fav'].get('interval','15m')}_{r['direction']}"
                if r["now"] and key not in already:
                    filtered.append(r)

            body, any_yes = self._compose_alert_email(rows if filtered else rows)
            if any_yes:
                self._send_email(cfg, "PatternFinder: YES signal", body)
                # update dedupe set
                for r in rows:
                    if r["now"]:
                        key = f"{r['ticker']}_{r['fav'].get('interval','15m')}_{r['direction']}"
                        already.add(key)
                sent[today] = sorted(already)
                self._save_sent_map(sent)
                self.alert_status.set("Alert email sent.")
            else:
                # optional: still send summary? We'll only log locally.
                self.alert_status.set("No YES signals. (No email sent.)")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Alerts", f"Run-now failed: {e}")
            self.alert_status.set("Run-now failed.")

    def _seconds_until(self, hhmm:str):
        try:
            hh, mm = [int(x) for x in hhmm.split(":")]
        except:
            hh, mm = 9, 28
        now = now_et()
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        # If weekdays only, skip to Monday if weekend
        if self.alert_weekdays.get():
            while target.weekday() >= 5:  # 5=Sat, 6=Sun
                target += timedelta(days=1)
        return max(1, int((target - now).total_seconds()))

    def _tick_and_reschedule(self):
        self._alerts_run_once()
        self._start_alert_scheduler()

    def _start_alert_scheduler(self):
        self._stop_alert_scheduler()
        secs = self._seconds_until(self.alert_time.get().strip())
        self._alert_timer = threading.Timer(secs, self._tick_and_reschedule)
        self._alert_timer.daemon = True
        self._alert_timer.start()
        self.alert_status.set(f"Daily alerts scheduled in {secs//60} min {secs%60}s.")

    def _stop_alert_scheduler(self):
        try:
            if self._alert_timer is not None:
                self._alert_timer.cancel()
        except:
            pass
        self._alert_timer = None
        self.alert_status.set("Daily alerts stopped.")

# -----------------------------
# Main entry point
# -----------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    if platform.system() == "Darwin":
        # macOS requires 'spawn' for multiprocessing with Tk apps
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()
