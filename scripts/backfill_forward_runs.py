#!/usr/bin/env python3
"""One-off task to backfill ``forward_runs`` from historical forward tests."""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import db
from services.favorites import canonical_direction
from services.forward_runs import (
    forward_rule_hash,
    log_forward_entry,
    log_forward_exit,
)
from services.telemetry import log as log_telemetry

LOGGER = logging.getLogger(__name__)
FORWARD_SLIPPAGE = 0.0008


@dataclass
class BackfillStats:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
        }


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        if number != number:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        stamp = raw
    else:
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    stamp = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _interval_minutes(value: Any) -> int:
    try:
        label = str(value or "").strip().lower()
    except Exception:
        return 0
    if not label:
        return 0
    try:
        if label.endswith("m"):
            return int(float(label[:-1]))
        if label.endswith("h"):
            return int(float(label[:-1]) * 60)
        if label.endswith("d"):
            return int(float(label[:-1]) * 60 * 24)
        return int(float(label))
    except (TypeError, ValueError):
        return 0


def _estimate_entry_ts(created_at: Any, interval: Any) -> str | None:
    created_stamp = _parse_timestamp(created_at)
    if created_stamp is None:
        return None
    interval_minutes = _interval_minutes(interval)
    if interval_minutes:
        try:
            created_stamp = created_stamp + timedelta(minutes=interval_minutes)
        except Exception:
            pass
    return created_stamp.isoformat().replace("+00:00", "Z")


def _estimate_exit_ts(entry_iso: str | None, row: sqlite3.Row) -> str | None:
    base_stamp = _parse_timestamp(entry_iso)
    if base_stamp is None:
        base_stamp = _parse_timestamp(row["created_at"])
    if base_stamp is None:
        return None
    exit_reason = (row["exit_reason"] or "").strip().lower()
    if exit_reason == "target":
        elapsed = _coerce_float(row["time_to_hit"])
    elif exit_reason == "stop":
        elapsed = _coerce_float(row["time_to_stop"])
    else:
        elapsed = None
    interval_minutes = _interval_minutes(row["interval"])
    if elapsed is not None:
        try:
            base_stamp = base_stamp + timedelta(minutes=float(elapsed))
            return base_stamp.isoformat().replace("+00:00", "Z")
        except Exception:
            pass
    bars = _coerce_int(row["bars_to_exit"])
    if bars and interval_minutes:
        try:
            exit_stamp = base_stamp + timedelta(minutes=bars * interval_minutes)
            return exit_stamp.isoformat().replace("+00:00", "Z")
        except Exception:
            pass
    fallback = _parse_timestamp(row["updated_at"] or row["last_run_at"])
    if fallback is None:
        return None
    return fallback.isoformat().replace("+00:00", "Z")


def _estimate_exit_price(entry_px: float | None, roi_pct: Any, direction: Any) -> float | None:
    entry_price = _coerce_float(entry_px)
    roi_value = _coerce_float(roi_pct)
    if entry_price is None or roi_value is None:
        return None
    side = 1 if canonical_direction(direction) == "UP" else -1
    entry_fill = entry_price * (1 + FORWARD_SLIPPAGE * side)
    try:
        exit_fill = entry_fill * (1 + (roi_value / 100.0) / side)
    except ZeroDivisionError:
        return None
    exit_factor = 1 - FORWARD_SLIPPAGE * side
    if exit_factor == 0:
        return None
    return float(exit_fill / exit_factor)


def _load_forward_tests(cur: sqlite3.Cursor) -> Sequence[sqlite3.Row]:
    cur.execute(
        """
        SELECT
            fav_id,
            created_at,
            interval,
            entry_price,
            exit_reason,
            roi_forward,
            bars_to_exit,
            max_drawdown_pct,
            dd_forward,
            direction,
            rule,
            time_to_hit,
            time_to_stop,
            updated_at,
            last_run_at
        FROM forward_tests
        ORDER BY created_at
        """
    )
    return cur.fetchall()


def _determine_outcome(raw: Any) -> str | None:
    label = (raw or "").strip().lower()
    if label in {"target", "hit"}:
        return "hit"
    if label in {"stop", "timeout"}:
        return label
    return None


def backfill_forward_runs(db_path: str, dry_run: bool = False) -> BackfillStats:
    start_ts = time.perf_counter()
    stats = BackfillStats()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = _load_forward_tests(cur)
    log_telemetry({"event": "forward_runs_backfill_start", "count": len(rows)})

    for row in rows:
        fav_id = row["fav_id"]
        entry_ts = _estimate_entry_ts(row["created_at"], row["interval"])
        if fav_id is None or entry_ts is None:
            stats.skipped += 1
            continue
        entry_price = _coerce_float(row["entry_price"])
        rule_hash = forward_rule_hash(row["rule"])
        fav_key = str(fav_id)

        cur.execute(
            "SELECT 1 FROM forward_runs WHERE favorite_id=? AND entry_ts=?",
            (fav_key, entry_ts),
        )
        existing = cur.fetchone() is not None

        if not dry_run:
            log_forward_entry(cur, fav_key, entry_ts, entry_price, rule_hash)

        outcome = _determine_outcome(row["exit_reason"])
        roi_decimal = _coerce_float(row["roi_forward"])
        roi_value = (roi_decimal / 100.0) if roi_decimal is not None else None
        dd_pct = _coerce_float(row["dd_forward"]) or _coerce_float(row["max_drawdown_pct"])
        dd_decimal = (dd_pct / 100.0) if dd_pct is not None else None
        tt_bars = _coerce_int(row["bars_to_exit"])

        if outcome and not dry_run:
            exit_ts = _estimate_exit_ts(entry_ts, row)
            exit_px = _estimate_exit_price(entry_price, roi_decimal, row["direction"])
            log_forward_exit(
                cur,
                fav_key,
                entry_ts,
                exit_ts,
                exit_px,
                outcome,
                roi_value,
                tt_bars,
                dd_decimal,
            )

        if existing:
            stats.updated += 1
        else:
            stats.inserted += 1

    if not dry_run:
        conn.commit()
    else:
        conn.rollback()
    conn.close()

    duration_ms = (time.perf_counter() - start_ts) * 1000.0
    payload = stats.to_payload() | {"duration_ms": round(duration_ms, 2)}
    log_telemetry({"event": "forward_runs_backfill_done", **payload})
    LOGGER.info(
        "backfill complete inserted=%s updated=%s skipped=%s duration_ms=%.2f",
        stats.inserted,
        stats.updated,
        stats.skipped,
        duration_ms,
    )
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill forward_runs table from forward_tests history")
    parser.add_argument("--db", dest="db_path", default=db.DB_PATH, help="Path to SQLite database")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing changes")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        stats = backfill_forward_runs(args.db_path, dry_run=args.dry_run)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("forward runs backfill failed")
        log_telemetry({"event": "forward_runs_backfill_done", "error": str(exc)})
        return 1

    if args.dry_run:
        print("DRY RUN -- no changes committed")
    print(
        "Inserted: {inserted}, Updated: {updated}, Skipped: {skipped}".format(
            **stats.to_payload()
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
