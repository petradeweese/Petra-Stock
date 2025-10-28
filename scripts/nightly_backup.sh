#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DB="/home/ubuntu/Petra-Stock/patternfinder.db"
OUT_DIR="db_backups"
STAMP="$(date -u +%F_%H%M%S)"
OUT="$OUT_DIR/patternfinder.$STAMP.db"

mkdir -p "$OUT_DIR" logs
echo "$(date -u '+%F %T') start backup -> $OUT" | tee -a logs/backup.log

# Make a consistent copy even if the DB is busy (safe for WAL/live writes)
sqlite3 "$DB" ".backup '$OUT'"

# Sanity check the copy
ICHECK="$(sqlite3 "$OUT" 'PRAGMA integrity_check;')"
echo "$(date -u '+%F %T') integrity_check: $ICHECK" | tee -a logs/backup.log
if [ "$ICHECK" != "ok" ]; then
  echo "$(date -u '+%F %T') WARNING: backup integrity_check != ok" | tee -a logs/backup.log
fi

# Compress (saves space). Result will be patternfinder.<ts>.db.xz
xz -T0 -f "$OUT" || true

# Retain 14 days
find "$OUT_DIR" -name 'patternfinder.*.db*' -mtime +14 -delete || true

echo "$(date -u '+%F %T') done" | tee -a logs/backup.log
