CREATE TABLE IF NOT EXISTS venues (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conference_instances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  venue_id INTEGER NOT NULL REFERENCES venues(id),
  year INTEGER NOT NULL,
  official_url TEXT,
  proceedings_url TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(venue_id, year)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conference_instance_id INTEGER REFERENCES conference_instances(id),
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  source_type TEXT,
  source_value TEXT,
  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT,
  message TEXT
);

CREATE TABLE IF NOT EXISTS papers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conference_instance_id INTEGER NOT NULL REFERENCES conference_instances(id),
  title TEXT NOT NULL,
  authors TEXT NOT NULL,
  doi TEXT,
  abstract TEXT,
  paper_url TEXT,
  pdf_url TEXT,
  session TEXT,
  bibtex_key TEXT,
  source_confidence REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(conference_instance_id, title)
);

CREATE TABLE IF NOT EXISTS paper_sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER REFERENCES papers(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  source_url TEXT NOT NULL,
  source_method TEXT NOT NULL,
  evidence_text TEXT,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_classifications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  is_wireless INTEGER,
  label TEXT NOT NULL,
  confidence REAL NOT NULL,
  evidence TEXT,
  model_version TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_name TEXT NOT NULL UNIQUE,
  normalized_name TEXT NOT NULL,
  source_paper_id INTEGER REFERENCES papers(id),
  availability_status TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_dataset_links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  dataset_id INTEGER NOT NULL REFERENCES datasets(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  relationship_type TEXT NOT NULL,
  confidence REAL NOT NULL,
  evidence_text TEXT,
  evidence_url TEXT,
  review_needed INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, dataset_id, relationship_type)
);

CREATE TABLE IF NOT EXISTS dataset_links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id INTEGER REFERENCES datasets(id),
  url TEXT NOT NULL,
  link_type TEXT,
  status_code INTEGER,
  availability_status TEXT,
  checked_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS availability_checks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id INTEGER REFERENCES datasets(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  url TEXT NOT NULL,
  availability_status TEXT NOT NULL,
  confidence REAL NOT NULL,
  evidence_text TEXT,
  checked_at TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bibtex_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER REFERENCES papers(id),
  citation_key TEXT NOT NULL,
  doi TEXT,
  bibtex TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(citation_key)
);

CREATE TABLE IF NOT EXISTS evidence_claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  claim_id TEXT NOT NULL UNIQUE,
  run_id INTEGER REFERENCES pipeline_runs(id),
  paper_id INTEGER REFERENCES papers(id),
  dataset_id INTEGER REFERENCES datasets(id),
  claim_type TEXT NOT NULL,
  claim_value TEXT NOT NULL,
  evidence_text TEXT,
  source_url TEXT,
  confidence REAL NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER REFERENCES pipeline_runs(id),
  item_type TEXT NOT NULL,
  paper_title TEXT,
  dataset_name TEXT,
  field TEXT NOT NULL,
  suggested_value TEXT,
  confidence REAL NOT NULL,
  review_reason TEXT NOT NULL,
  evidence TEXT,
  source_url TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  resolved_at TEXT
);
