ALTER TABLE favorites ADD COLUMN roi_snapshot REAL;
ALTER TABLE favorites ADD COLUMN hit_pct_snapshot REAL;
ALTER TABLE favorites ADD COLUMN dd_pct_snapshot REAL;
ALTER TABLE favorites ADD COLUMN rule_snapshot TEXT;
ALTER TABLE favorites ADD COLUMN settings_json_snapshot TEXT;
ALTER TABLE favorites ADD COLUMN snapshot_at TEXT;
