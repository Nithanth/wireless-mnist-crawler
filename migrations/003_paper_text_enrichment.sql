CREATE TABLE IF NOT EXISTS paper_text_artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  source_type TEXT NOT NULL,
  source_url TEXT,
  fetch_status TEXT NOT NULL,
  content_text TEXT NOT NULL DEFAULT '',
  content_sha256 TEXT NOT NULL,
  error_message TEXT,
  fetched_at TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, run_id, source_type, source_url)
);

CREATE TABLE IF NOT EXISTS paper_text_links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  artifact_id INTEGER REFERENCES paper_text_artifacts(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  url TEXT NOT NULL,
  link_text TEXT,
  link_type TEXT NOT NULL,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, run_id, url)
);

CREATE TABLE IF NOT EXISTS paper_text_snippets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  artifact_id INTEGER REFERENCES paper_text_artifacts(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  snippet_type TEXT NOT NULL,
  snippet_text TEXT NOT NULL,
  source_url TEXT,
  start_char INTEGER,
  end_char INTEGER,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
