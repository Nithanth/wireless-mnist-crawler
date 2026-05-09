CREATE TABLE IF NOT EXISTS paper_list_verification_reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER REFERENCES pipeline_runs(id),
  conference_instance_id INTEGER REFERENCES conference_instances(id),
  paper_count INTEGER NOT NULL,
  missing_authors_count INTEGER NOT NULL,
  missing_abstract_count INTEGER NOT NULL,
  missing_doi_count INTEGER NOT NULL,
  duplicate_title_count INTEGER NOT NULL,
  low_confidence_count INTEGER NOT NULL,
  external_checked_count INTEGER NOT NULL DEFAULT 0,
  external_mismatch_count INTEGER NOT NULL DEFAULT 0,
  llm_correction_count INTEGER NOT NULL DEFAULT 0,
  final_confidence REAL NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
