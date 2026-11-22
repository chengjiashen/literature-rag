# Testing incremental PDF parsing

Use one synthetic paper to check when the parser rebuilds its sections and chunks. An unchanged PDF should be skipped. A revised PDF should be processed again. If a replacement PDF is unreadable, the database should retain the chunks from the last successful parse.

## 1. Set up the test database

Run the commands below from the repository root using the project's Python environment. The parser uses PyMuPDF, psycopg, pgvector and python-dotenv. The manual checks require PostgreSQL with the pgvector extension available. All examples run without an OpenAI API call.

Create a separate database named `literature_rag_dev`. For a fresh database, run the two SQL files in pgAdmin's Query Tool in this order, once each:

1. [001_initial_schema.sql](migrations/001_initial_schema.sql) creates the schema.
2. [002_add_document_processing_metadata.sql](migrations/002_add_document_processing_metadata.sql) adds parsing status, fingerprints, timestamps and error details.

If both migrations have already been applied, continue with the existing test database. Set the database name in the local `.env` file and fill in the connection settings for your PostgreSQL instance:

```dotenv
DB_NAME=literature_rag_dev
```

The parser checks the connected database name before processing. If the check fails, inspect the IDE's environment settings as well as `.env`. The `--expected-database` option controls this check. Connection settings determine which database is opened.

## 2. Process one document twice

Generate a two-page PDF and register it in the test database:

```bash
python tools/prepare_incremental_demo.py
```

The helper stores its PDF under `.incremental_demo/` and prints a document ID. Use that ID wherever the examples show `1`. Running the helper again from the same checkout reuses the document registration.

Run the same parser command twice, leaving the file untouched between runs:

```bash
python 02_pdf_to_sections_chunks.py --document-id 1
python 02_pdf_to_sections_chunks.py --document-id 1
```

| Run | Expected summary for a newly registered document |
| --- | --- |
| First run | `processed=1 skipped=0 failed=0 blocked=0` |
| Second run | `processed=0 skipped=1 failed=0 blocked=0` |

On the second run, the parser recognizes the successful fingerprint and pipeline version. The existing chunk IDs, successful parse time and latest attempt time should all stay the same.

## 3. Inspect the result in pgAdmin

Use this read-only query in `literature_rag_dev` to compare the document state before and after each test. Substitute the ID printed by the helper.

```sql
SELECT d.document_id,
       d.parse_status,
       d.parsed_file_sha256,
       d.parse_pipeline_version,
       d.parsed_at,
       d.last_parse_attempt_at,
       d.parse_error,
       (SELECT count(*) FROM public.sections AS s
        WHERE s.document_id = d.document_id) AS section_count,
       (SELECT count(*) FROM public.chunks AS c
        WHERE c.document_id = d.document_id) AS chunk_count,
       (SELECT array_agg(c.chunk_id ORDER BY c.chunk_id)
        FROM public.chunks AS c
        WHERE c.document_id = d.document_id) AS chunk_ids
FROM public.documents AS d
WHERE d.document_id = 1;
```

## 4. Change the PDF

```bash
python tools/prepare_incremental_demo.py --variant changed
python 02_pdf_to_sections_chunks.py --document-id 1
python 02_pdf_to_sections_chunks.py --document-id 1
```

Expect `processed=1` on the first run and `skipped=1` on the second. In pgAdmin, record the new fingerprint, successful parse time and chunk IDs. These values serve as the reference for the failure test.

A successful rebuild replaces the document's sections and chunks. Any embeddings on the old chunks are removed with those rows. In normal ingestion, `03_generate_embeddings.py` generates embeddings for the new chunks. The parsing tests finish without running that stage.

## 5. Test failure and recovery

First replace the synthetic PDF with invalid bytes and run the parser:

```bash
python tools/prepare_incremental_demo.py --variant broken
python 02_pdf_to_sections_chunks.py --document-id 1
```

Expect `failed=1` and exit code 1. Re-run the query from step 3. The status should be `failed`, the error field should explain the failure, and the latest attempt time should advance. Compare the successful fingerprint, pipeline version, successful parse time and chunk IDs against step 4. They should still describe the last successful parse.

Restore the original readable PDF, retry, then run once more:

```bash
python tools/prepare_incremental_demo.py --variant initial
python 02_pdf_to_sections_chunks.py --document-id 1
python 02_pdf_to_sections_chunks.py --document-id 1
```

Expect `processed=1`, followed by `skipped=1`. The database should show `completed`, a cleared error (`NULL`), a new successful parse time and newly generated chunk IDs.

For a separate text-extraction check, use `--variant blank` to generate a valid PDF with an empty page. Parsing should report that no text was extracted. Use `--variant initial` again to restore the demo PDF afterward.

## 6. Run the offline tests

```bash
python -m unittest discover -s tests -v
```

The suite contains 20 regression tests. It uses real PDFs for text extraction and mock connections for database calls. It checks skip decisions, changes to parser settings, invalid inputs, protection of legacy chunks and the order of transaction and failure-recording operations.

**Verification record.** All 20 offline tests passed. A manual run on PostgreSQL 17 on 2026-11-15 also verified first-time processing, skipping a repeat run, rebuilding after a PDF change, retaining previous chunks after a PDF read failure, and recovery after restoring a readable file. For example, chunk IDs `{5,6,7,8}` survived the read failure. Recovery created `{9,10,11,12}` and cleared the error. Actual IDs will vary. A failure during a SQL insert has been simulated in the offline tests. Rollback during an actual PostgreSQL insert failure remains unverified.

## Parser behavior and exit codes

- The pipeline version includes `PARSER_RULES_VERSION`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, the recognized headings and the PyMuPDF version. Bump `PARSER_RULES_VERSION` when changing the extraction, heading detection or splitting algorithms.
- Each PDF is read into memory once so that hashing and parsing use the same bytes. The parser uses rule-based section detection and character-based chunks. Page range fields are currently NULL.
- Legacy documents with sections or chunks but no successful fingerprint return `blocked` by default. Use `--rebuild-untracked` for a deliberate rebuild of legacy output and `--force` to rebuild tracked output. Both can require regenerating embeddings.
- A session-level advisory lock allows one incremental parser run per database at a time. Keep other writers such as older parsers, embedding jobs and manual changes idle during the test. A later run will retry a document left in `processing` after an interruption.
- Keep local database settings in `.env` and generated PDFs in `.incremental_demo/`. Exclude both from version control.

| Exit code | Meaning |
| --- | --- |
| 0 | All selected documents were processed or skipped, or none were registered |
| 1 | At least one document failed or was blocked |
| 2 | The run stopped because of a configuration, database or other pipeline error |
| 130 | The user interrupted the run |
