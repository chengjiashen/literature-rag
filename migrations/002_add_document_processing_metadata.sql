-- Literature RAG: migration 002 - PDF parsing and chunking metadata
-- Prerequisite: 001_initial_schema.sql has already been applied.
-- First apply ONCE to literature_rag_dev using pgAdmin Query Tool.
-- Do not rerun 001 and do not apply this to the original database yet.
-- No documents, sections, chunks or embeddings are deleted or regenerated.
-- This migration adds metadata only; the Python parser must be updated next.
-- Existing documents default to pending: historical processing is unverified,
-- not necessarily absent. Never infer a successful fingerprint from old chunks.
-- completed refers ONLY to PDF parsing/chunking, not embedding readiness.
-- Runtime database verification is still required.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

ALTER TABLE public.documents
    ADD COLUMN parse_status text NOT NULL DEFAULT 'pending',
    ADD COLUMN parsed_file_sha256 text,
    ADD COLUMN parse_pipeline_version text,
    ADD COLUMN parsed_at timestamp with time zone,
    ADD COLUMN last_parse_attempt_at timestamp with time zone,
    ADD COLUMN parse_error text,

    ADD CONSTRAINT documents_parse_status_check
        CHECK (parse_status IN ('pending', 'processing', 'completed', 'failed')),

    ADD CONSTRAINT documents_parsed_file_sha256_check
        CHECK (parsed_file_sha256 IS NULL
               OR parsed_file_sha256 ~ '^[0-9a-f]{64}$'),

    ADD CONSTRAINT documents_parse_pipeline_version_check
        CHECK (parse_pipeline_version IS NULL
               OR length(btrim(parse_pipeline_version)) > 0),

    ADD CONSTRAINT documents_parse_snapshot_check
        CHECK (
            (parsed_file_sha256 IS NULL
             AND parse_pipeline_version IS NULL
             AND parsed_at IS NULL)
            OR
            (parsed_file_sha256 IS NOT NULL
             AND parse_pipeline_version IS NOT NULL
             AND parsed_at IS NOT NULL)
        ),

    ADD CONSTRAINT documents_parse_completed_check
        CHECK (
            parse_status <> 'completed'
            OR (parsed_file_sha256 IS NOT NULL
                AND parse_pipeline_version IS NOT NULL
                AND parsed_at IS NOT NULL
                AND last_parse_attempt_at IS NOT NULL
                AND parse_error IS NULL)
        );

COMMENT ON COLUMN public.documents.parse_status IS
    'Latest parsing/chunking attempt: pending, processing, completed or failed. Independent of embedding status. A skipped unchanged document remains completed.';

COMMENT ON COLUMN public.documents.parsed_file_sha256 IS
    'Lowercase SHA-256 of the exact PDF bytes used in the last successful parse. Commit with replacement sections/chunks; retain on a failed attempt.';

COMMENT ON COLUMN public.documents.parse_pipeline_version IS
    'Identifier for the parser rules and chunk settings used in the last successful parse. Must change when parser behavior or chunk size/overlap changes.';

COMMENT ON COLUMN public.documents.parsed_at IS
    'Time of the last successful parsing/chunking operation. Commit atomically with its output and successful fingerprint.';

COMMENT ON COLUMN public.documents.last_parse_attempt_at IS
    'Start time of the latest actual parsing attempt; unchanged when skipping an unchanged document.';

COMMENT ON COLUMN public.documents.parse_error IS
    'Error summary from the latest failed parsing attempt. Clear on a new attempt or success; do not store credentials.';

COMMIT;

-- Verification: a successful application should return exactly these 6 rows.
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'documents'
  AND column_name IN (
      'parse_status',
      'parsed_file_sha256',
      'parse_pipeline_version',
      'parsed_at',
      'last_parse_attempt_at',
      'parse_error'
  )
ORDER BY ordinal_position;

-- If execution fails before COMMIT, execute ROLLBACK in that same session.
-- A duplicate-column error can mean 002 was already applied: inspect first.
