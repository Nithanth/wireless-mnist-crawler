CREATE TABLE IF NOT EXISTS resolver_cache (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  cache_key TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'ok',
  error_message TEXT,
  fetched_at TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(provider, cache_key)
);

CREATE TABLE IF NOT EXISTS paper_analysis_reflections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  analysis_run_id INTEGER REFERENCES pipeline_runs(id),
  decision TEXT NOT NULL,
  confidence REAL NOT NULL,
  issues_json TEXT NOT NULL,
  reflection_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, run_id)
);
