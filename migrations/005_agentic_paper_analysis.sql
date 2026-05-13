CREATE TABLE IF NOT EXISTS paper_agentic_analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  provider_name TEXT NOT NULL,
  wireless_label TEXT NOT NULL,
  is_wireless INTEGER,
  wireless_confidence REAL NOT NULL,
  wireless_evidence TEXT,
  modalities_json TEXT NOT NULL,
  osi_layers_json TEXT NOT NULL,
  summary TEXT,
  analysis_json TEXT NOT NULL,
  review_needed INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, run_id)
);

CREATE TABLE IF NOT EXISTS paper_analysis_dataset_claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  dataset_id INTEGER REFERENCES datasets(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  dataset_name TEXT NOT NULL,
  relationship_type TEXT NOT NULL,
  confidence REAL NOT NULL,
  modalities_json TEXT NOT NULL,
  osi_layers_json TEXT NOT NULL,
  evidence_text TEXT,
  source_url TEXT,
  availability_status TEXT,
  review_needed INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, run_id, dataset_name, relationship_type)
);
