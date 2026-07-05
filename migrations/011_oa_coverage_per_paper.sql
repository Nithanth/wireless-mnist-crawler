-- Per-paper OA coverage state, written atomically as each paper resolves.
-- This makes fetch-coverage crash-safe: a restart resumes from where it left
-- off (resolved papers skip instantly from the OA resolver cache), and
-- extract-datasets reads PDF URLs directly from here instead of requiring the
-- cov_*.json handoff file.
--
-- oa_status: 'gold'|'green'|'hybrid'|'bronze'|'diamond'|'closed'|NULL
--   NULL     = never attempted
--   'closed' = attempted, no OA copy found (retried after oa_negative_ttl_days)
--   other    = open-access copy found, pdf_url is the direct download link
--
-- oa_attempted_at: unix timestamp of the most recent resolve() call.
--   NULL     = never attempted
--   set      = we tried; result is in oa_status / pdf_url
--
-- oa_provider: which waterfall step found the PDF (unpaywall, openalex,
--   semantic_scholar, arxiv, brave_search, core, cache, none, …)

ALTER TABLE papers ADD COLUMN oa_status TEXT;
ALTER TABLE papers ADD COLUMN oa_attempted_at REAL;
ALTER TABLE papers ADD COLUMN oa_provider TEXT;
