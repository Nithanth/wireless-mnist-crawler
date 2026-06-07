-- Simpler experiment: title+abstract wireless-candidate classification and
-- Jaccard/IoU overlap against a manually curated gold set.

CREATE TABLE IF NOT EXISTS wireless_candidate_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  classifier TEXT NOT NULL,
  model_version TEXT NOT NULL,
  label TEXT NOT NULL,
  confidence REAL NOT NULL,
  evidence TEXT,
  high_pass INTEGER NOT NULL DEFAULT 0,
  low_pass INTEGER NOT NULL DEFAULT 0,
  used_abstract INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, run_id, classifier)
);

CREATE INDEX IF NOT EXISTS idx_wireless_candidate_predictions_run
  ON wireless_candidate_predictions(run_id);

CREATE TABLE IF NOT EXISTS gold_papers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conference_instance_id INTEGER NOT NULL REFERENCES conference_instances(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  title TEXT NOT NULL,
  normalized_title TEXT NOT NULL,
  doi TEXT,
  normalized_doi TEXT,
  raw_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(conference_instance_id, normalized_title)
);

CREATE INDEX IF NOT EXISTS idx_gold_papers_instance
  ON gold_papers(conference_instance_id);
