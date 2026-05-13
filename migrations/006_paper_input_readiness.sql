CREATE TABLE IF NOT EXISTS paper_input_readiness (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL REFERENCES papers(id),
  run_id INTEGER REFERENCES pipeline_runs(id),
  has_abstract INTEGER NOT NULL,
  has_fetched_text INTEGER NOT NULL,
  has_pdf_link INTEGER NOT NULL,
  has_artifact_link INTEGER NOT NULL,
  snippet_count INTEGER NOT NULL,
  usable_text_chars INTEGER NOT NULL,
  readiness_level TEXT NOT NULL,
  should_analyze INTEGER NOT NULL,
  limitations_json TEXT NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(paper_id, run_id)
);
