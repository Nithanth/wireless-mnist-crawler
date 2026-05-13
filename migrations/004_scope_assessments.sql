CREATE TABLE IF NOT EXISTS scope_assessments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER REFERENCES pipeline_runs(id),
  conference_instance_id INTEGER REFERENCES conference_instances(id),
  paper_count INTEGER NOT NULL,
  networking_like_count INTEGER NOT NULL,
  wireless_like_count INTEGER NOT NULL,
  malformed_count INTEGER NOT NULL,
  networking_like_ratio REAL NOT NULL,
  wireless_like_ratio REAL NOT NULL,
  should_proceed INTEGER NOT NULL,
  decision TEXT NOT NULL,
  confidence REAL NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
