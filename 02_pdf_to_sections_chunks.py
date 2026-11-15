"""
PDF-to-Chunk ETL Pipeline for Literature RAG.

This script extracts text from academic PDF files, detects section boundaries
using rule-based heading matching, splits section text into overlapping chunks,
and loads the resulting records into PostgreSQL.

Pipeline:
    documents table
        -> PDF text extraction with PyMuPDF
        -> rule-based section segmentation
        -> overlapping chunk generation
        -> sections and chunks tables

Notes:
    - Requires migrations 001 and 002; defaults to validating literature_rag_dev.
    - Unchanged successfully processed documents are skipped.
    - Changed output and successful metadata commit atomically per document.
    - Section detection is rule-based and intentionally conservative.
    - Chunks are character-based rather than token-based.
    - Page ranges are currently stored as NULL and can be added later.
"""

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


# Bump this when extraction, normalization, sectioning or chunking logic changes.
# Chunk parameters, heading vocabulary and PyMuPDF version are included separately.
PARSER_RULES_VERSION = "section-parser-v2"
LOGGER = logging.getLogger(__name__)
REQUIRED_METADATA_COLUMNS = {
    "parse_status", "parsed_file_sha256", "parse_pipeline_version",
    "parsed_at", "last_parse_attempt_at", "parse_error",
}


# ============================================================
# Section heading configuration
# ============================================================

MAJOR_SECTION_TITLES = {
    # Front matter
    "abstract",
    "keyword",
    "keywords",

    # Introduction and background
    "introduction",
    "background",
    "literature review",
    "review of literature",
    "previous research",
    "related work",
    "theoretical framework",
    "conceptual framework",
    "background literature",
    "formulaic language",

    # Study aims and research questions
    "research question",
    "research questions",
    "aim",
    "aims",
    "objective",
    "objectives",
    "the present study",
    "present study",
    "current study",
    "the current study",

    # Methods
    "method",
    "methods",
    "methodology",
    "materials and methods",
    "material and method",
    "research design",

    # Data, corpora, and participants
    "data",
    "the data",
    "dataset",
    "datasets",
    "corpus",
    "corpora",
    "participants",
    "subjects",
    "informants",
    "data and participants",
    "participants and data",
    "data and methodology",
    "data collection",
    "data analysis",

    # Procedures and instruments
    "procedure",
    "procedures",
    "materials",
    "measures",
    "instruments",
    "coding",
    "annotation",
    "coding scheme",

    # Analysis
    "analysis",
    "analyses",
    "statistical analysis",
    "qualitative analysis",
    "quantitative analysis",

    # Results
    "result",
    "results",
    "findings",
    "results and analysis",
    "results and discussion",
    "results and discussions",
    "findings and discussion",

    # Discussion
    "discussion",
    "general discussion",

    # Conclusion
    "conclusion",
    "conclusions",
    "concluding remarks",
    "final remarks",

    # Limitations and implications
    "limitations",
    "limitations of the study",
    "implications",
    "pedagogical implications",
    "future research",
    "directions for future research",
    "limitations and future directions",

    # Acknowledgements
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",

    # References
    "reference",
    "references",
    "bibliography",
    "works cited",

    # Appendices
    "appendix",
    "appendices",
}


# ============================================================
# PDF text extraction
# ============================================================

def extract_pdf_text(pdf_bytes):
    """
    Extract plain text from a PDF file.

    Args:
        pdf_bytes: The same immutable bytes used to compute the file hash.

    Returns:
        A single string containing extracted text from all pages.
        Synthetic page markers are included for debugging and future
        page-level metadata support.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDF requires a decrypted copy.")
        all_pages = []
        has_body_text = False

        for page_index in range(document.page_count):
            page = document[page_index]
            text = page.get_text("text", sort=True)
            has_body_text = has_body_text or bool(text.strip())

            page_text = (
                f"\n\n===== Page {page_index + 1} =====\n\n{text}"
            )
            all_pages.append(page_text)

    if not has_body_text:
        raise ValueError("No extractable text. The PDF may require OCR.")

    return "\n".join(all_pages)


# ============================================================
# Section heading normalization
# ============================================================

def normalize_heading_text(line):
    """
    Normalize a candidate heading for strict matching.

    Examples:
        Abstract             -> abstract
        III Data             -> data
        IV. Results          -> results
        3.1 Data collection  -> data collection
        2. Methods:          -> methods
        A. Participants      -> participants
    """
    line = line.strip()
    line = re.sub(r"\s+", " ", line)

    # Remove Arabic numbering such as "1 Introduction"
    # or "3.1 Data collection".
    line = re.sub(
        r"^\d+(\.\d+)*\.?\s+",
        "",
        line,
    )

    # Remove Roman numbering such as "III Data" or "IV. Method".
    line = re.sub(
        (
            r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|"
            r"XIV|XV|XVI|XVII|XVIII|XIX|XX)\.?\s+"
        ),
        "",
        line,
        flags=re.IGNORECASE,
    )

    # Remove letter-based numbering such as "A. Participants".
    line = re.sub(
        r"^[A-Z]\.?\s+",
        "",
        line,
    )

    # Keep only letters, whitespace, and hyphens.
    line = re.sub(
        r"[^A-Za-z\s\-]",
        " ",
        line,
    )

    # Collapse whitespace and convert to lower case.
    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line.strip().lower()


def clean_heading_for_display(line):
    """
    Clean a detected heading while preserving readable capitalization.

    The returned value is stored in sections.section_title.
    """
    line = line.strip()
    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    line = re.sub(
        r"^\d+(\.\d+)*\.?\s+",
        "",
        line,
    )

    line = re.sub(
        (
            r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|"
            r"XIV|XV|XVI|XVII|XVIII|XIX|XX)\.?\s+"
        ),
        "",
        line,
        flags=re.IGNORECASE,
    )

    line = re.sub(
        r"^[A-Z]\.?\s+",
        "",
        line,
    )

    return line.strip(" :.-–—")


# ============================================================
# Section heading detection
# ============================================================

def is_section_heading(line):
    """
    Determine whether a line is likely to be a section heading.

    The detector is intentionally conservative:

        1. Normalize the candidate line.
        2. Reject obvious non-headings such as page markers, table rows,
           standalone page numbers, and long sentences.
        3. Require an exact match against a controlled set of common
           academic section titles.

    Returns:
        True when the line matches a recognized section heading,
        otherwise False.
    """
    original_line = line.strip()

    if not original_line:
        return False

    original_line = re.sub(
        r"\s+",
        " ",
        original_line,
    )

    # Ignore synthetic page markers added during PDF extraction.
    if original_line.startswith("===== Page"):
        return False

    # Ignore standalone page numbers.
    if re.fullmatch(r"\d+", original_line):
        return False

    # Long lines are more likely to be prose or table content.
    if len(original_line) > 120:
        return False

    # Most academic section headings do not end with a period.
    if original_line.endswith("."):
        return False

    # Exclude table-like rows containing notation and numeric values.
    if "+" in original_line and re.search(
        r"\d+(\.\d+)?",
        original_line,
    ):
        return False

    # Rows containing multiple numbers are often table rows.
    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        original_line,
    )

    if len(numbers) >= 2:
        return False

    normalized = normalize_heading_text(
        original_line
    )

    if not normalized:
        return False

    # Very long normalized headings are less reliable.
    if len(normalized.split()) > 10:
        return False

    return normalized in MAJOR_SECTION_TITLES


# ============================================================
# Section segmentation
# ============================================================

def split_into_sections(full_text):
    """
    Split extracted PDF text into section-level units.

    Args:
        full_text: Full extracted text from a PDF.

    Returns:
        A list of dictionaries in the following form:

            [
                {"heading": "Abstract", "content": "..."},
                {"heading": "Introduction", "content": "..."},
                ...
            ]
    """
    lines = full_text.splitlines()

    sections = []
    current_heading = "FRONT_MATTER"
    current_content = []

    for line in lines:
        stripped_line = line.strip()

        if is_section_heading(stripped_line):
            if current_content:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": "\n".join(
                            current_content
                        ).strip(),
                    }
                )

            display_heading = clean_heading_for_display(
                stripped_line
            )

            if not display_heading:
                display_heading = normalize_heading_text(
                    stripped_line
                ).title()

            current_heading = display_heading
            current_content = []

        else:
            current_content.append(line)

    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "content": "\n".join(
                    current_content
                ).strip(),
            }
        )

    return [
        section
        for section in sections
        if section["content"]
    ]


# ============================================================
# Chunk generation
# ============================================================

def split_text_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Split section text into overlapping character-based chunks.

    Args:
        text: Section text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by neighboring chunks.

    Returns:
        A list of chunk strings.

    Raises:
        ValueError: If the chunking parameters are invalid.

    Notes:
        Character-based chunking is used intentionally in this version.
        Token-based chunking can be introduced in a later iteration.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk_text = text[
            start:end
        ].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# Database access
# ============================================================

def get_documents_to_process(conn, document_ids=None):
    """
    Retrieve PDF records from the documents table.

    Args:
        conn: Active PostgreSQL database connection.

    Returns:
        A list of dictionaries containing document_id, paper_id,
        and file_path.
    """
    sql = """
        SELECT
            d.document_id, d.paper_id, d.file_path,
            d.parse_status, d.parsed_file_sha256, d.parse_pipeline_version,
            EXISTS (SELECT 1 FROM public.sections s
                    WHERE s.document_id = d.document_id) AS has_sections,
            EXISTS (SELECT 1 FROM public.chunks c
                    WHERE c.document_id = d.document_id) AS has_chunks
        FROM public.documents d
    """
    parameters = ()
    if document_ids:
        sql += " WHERE d.document_id = ANY(%s)"
        parameters = (document_ids,)
    sql += " ORDER BY d.document_id;"

    with conn.cursor() as cursor:
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()

    return [
        {
            "document_id": row[0],
            "paper_id": row[1],
            "file_path": row[2],
            "parse_status": row[3],
            "parsed_file_sha256": row[4],
            "parse_pipeline_version": row[5],
            "has_sections": row[6],
            "has_chunks": row[7],
        }
        for row in rows
    ]


def delete_old_sections_and_chunks(
    conn,
    document_id,
):
    """
    Remove previously generated sections and chunks for a document.

    This makes the ETL pipeline idempotent at the document level:
    rerunning the pipeline replaces previously generated records
    rather than appending duplicates.

    Args:
        conn: Active PostgreSQL database connection.
        document_id: Document whose generated data should be replaced.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM public.chunks
            WHERE document_id = %s;
            """,
            (document_id,),
        )

        cursor.execute(
            """
            DELETE FROM public.sections
            WHERE document_id = %s;
            """,
            (document_id,),
        )


def insert_section(
    conn,
    paper_id,
    document_id,
    section_order,
    section_title,
):
    """
    Insert one section record and return its generated section_id.

    Args:
        conn: Active PostgreSQL database connection.
        paper_id: Parent paper identifier.
        document_id: Parent document identifier.
        section_order: Position of the section within the document.
        section_title: Human-readable section heading.

    Returns:
        The generated section_id.
    """
    sql = """
        INSERT INTO public.sections (
            paper_id,
            document_id,
            section_title,
            section_type,
            section_order,
            page_start,
            page_end,
            notes
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING section_id;
    """

    with conn.cursor() as cursor:
        cursor.execute(
            sql,
            (
                paper_id,
                document_id,
                section_title,
                "main",
                section_order,
                None,
                None,
                None,
            ),
        )

        return cursor.fetchone()[0]


def insert_chunk(
    conn,
    paper_id,
    document_id,
    section_id,
    chunk_order,
    chunk_text,
):
    """
    Insert one chunk record into the chunks table.

    Args:
        conn: Active PostgreSQL database connection.
        paper_id: Parent paper identifier.
        document_id: Parent document identifier.
        section_id: Parent section identifier.
        chunk_order: Position of the chunk within the section.
        chunk_text: Text stored for the chunk.
    """
    sql = """
        INSERT INTO public.chunks (
            paper_id,
            document_id,
            section_id,
            chunk_order,
            chunk_text,
            page_start,
            page_end,
            notes
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        );
    """

    with conn.cursor() as cursor:
        cursor.execute(
            sql,
            (
                paper_id,
                document_id,
                section_id,
                chunk_order,
                chunk_text,
                None,
                None,
                None,
            ),
        )


# ============================================================
# Document-level processing
# ============================================================

def build_pipeline_version():
    """Identify both the algorithm revision and its effective configuration."""
    settings = {
        "rules_version": PARSER_RULES_VERSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "headings": sorted(MAJOR_SECTION_TITLES),
        "pymupdf_version": pymupdf.VersionBind,
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return f"{PARSER_RULES_VERSION}:{hashlib.sha256(encoded).hexdigest()}"


def should_skip_document(document, file_hash, pipeline_version, force=False):
    """Skip only a verified success with matching inputs and existing output."""
    return (
        not force
        and document["parse_status"] == "completed"
        and document["parsed_file_sha256"] == file_hash
        and document["parse_pipeline_version"] == pipeline_version
        and document["has_sections"]
        and document["has_chunks"]
    )


def prepare_sections(pdf_bytes):
    """Extract and validate all text/chunks before deleting any existing output."""
    full_text = extract_pdf_text(pdf_bytes)
    # Page markers are useful while extracting but are not evidence text.
    full_text = re.sub(r"(?m)^===== Page \d+ =====\s*$", "", full_text)
    prepared = []
    for section in split_into_sections(full_text):
        chunks = split_text_into_chunks(
            section["content"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
        )
        if chunks:
            prepared.append({"heading": section["heading"], "chunks": chunks})
    if not prepared:
        raise ValueError("No usable section text or chunks were extracted.")
    return prepared


def mark_processing(conn, document_id, attempted_at):
    """Persist the start independently, so a crash leaves a retryable state."""
    with conn.cursor() as cursor:
        cursor.execute(
            """UPDATE public.documents
               SET parse_status = 'processing', last_parse_attempt_at = %s,
                   parse_error = NULL
               WHERE document_id = %s""",
            (attempted_at, document_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Document {document_id} no longer exists.")


def record_failure(conn, document_id, attempted_at, error):
    """Leave the previous successful fingerprint/version/time untouched."""
    message = f"{type(error).__name__}: {error}"[:2000]
    with conn.cursor() as cursor:
        cursor.execute(
            """UPDATE public.documents
               SET parse_status = 'failed', last_parse_attempt_at = %s,
                   parse_error = %s
               WHERE document_id = %s""",
            (attempted_at, message, document_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Cannot record failure: document {document_id} is missing.")


def replace_document_output(conn, document, sections, file_hash, pipeline_version):
    """Commit output and success metadata together, or roll them all back."""
    document_id = document["document_id"]
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT paper_id, file_path FROM public.documents
                   WHERE document_id = %s FOR UPDATE""",
                (document_id,),
            )
            current_source = cursor.fetchone()
            if current_source != (document["paper_id"], document["file_path"]):
                raise RuntimeError("Document registration changed during processing; retry.")

        delete_old_sections_and_chunks(conn, document_id)
        total_chunks = 0
        for section_order, section in enumerate(sections, start=1):
            section_id = insert_section(
                conn, document["paper_id"], document_id,
                section_order, section["heading"],
            )
            for chunk_order, chunk_text in enumerate(section["chunks"], start=1):
                insert_chunk(
                    conn, document["paper_id"], document_id,
                    section_id, chunk_order, chunk_text,
                )
                total_chunks += 1

        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE public.documents
                   SET parse_status = 'completed', parsed_file_sha256 = %s,
                       parse_pipeline_version = %s, parsed_at = clock_timestamp(),
                       parse_error = NULL
                   WHERE document_id = %s""",
                (file_hash, pipeline_version, document_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Cannot save document completion metadata.")
    return total_chunks


def process_one_document(
    conn, document, pipeline_version=None, force=False, rebuild_untracked=False
):
    """Process one document on an autocommit connection with explicit transactions.

    The caller holds the pipeline advisory lock. All hashes describe the exact
    in-memory bytes parsed. Successful old output survives extraction/SQL errors.
    """
    if not conn.autocommit:
        raise ValueError("Use autocommit with explicit per-document transactions.")
    pipeline_version = pipeline_version or build_pipeline_version()
    document_id = document["document_id"]

    # Historical chunks have no trustworthy provenance. Do not silently rebuild
    # and discard their embeddings. The caller must opt into a deliberate rebuild.
    if (
        document["parsed_file_sha256"] is None
        and (document["has_sections"] or document["has_chunks"])
        and not rebuild_untracked
    ):
        LOGGER.warning(
            "document_id=%s BLOCKED: existing output has no successful fingerprint. "
            "Use --rebuild-untracked only for a deliberate rebuild; embeddings "
            "for that document will need regeneration.", document_id,
        )
        return "blocked"

    attempted_at = datetime.now(timezone.utc)
    try:
        # Read once: hashing and extraction cannot accidentally use different bytes.
        pdf_bytes = Path(document["file_path"]).read_bytes()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        if should_skip_document(document, file_hash, pipeline_version, force):
            return "skipped"

        mark_processing(conn, document_id, attempted_at)
        sections = prepare_sections(pdf_bytes)
        replace_document_output(conn, document, sections, file_hash, pipeline_version)
    except KeyboardInterrupt as error:
        record_failure(conn, document_id, attempted_at, error)
        raise
    except Exception as error:
        # conn.transaction() has already rolled back replacement writes if needed.
        # Autocommit saves this failure separately from that failed transaction.
        record_failure(conn, document_id, attempted_at, error)
        LOGGER.error("document_id=%s FAILED: %s: %s", document_id, type(error).__name__, error)
        return "failed"
    return "processed"


def validate_target_database(conn, expected_database):
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        actual_database = cursor.fetchone()[0]
        if actual_database != expected_database:
            raise RuntimeError(
                f"Connected to {actual_database!r}, expected {expected_database!r}. "
                "For this test set DB_NAME=literature_rag_dev in your local .env."
            )
        cursor.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'documents'"""
        )
        columns = {row[0] for row in cursor.fetchall()}
    missing = REQUIRED_METADATA_COLUMNS - columns
    if missing:
        raise RuntimeError(
            "Apply migration 002 first. Missing columns: " + ", ".join(sorted(missing))
        )


def acquire_pipeline_lock(conn):
    """Serialize this parser's runs; the connection releases the lock on close."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT pg_try_advisory_lock(%s, %s)", (71402, 1))
        if not cursor.fetchone()[0]:
            raise RuntimeError("Another incremental PDF parser is running; retry later.")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Incremental PDF-to-chunk ingestion.")
    parser.add_argument("--document-id", type=int, action="append",
                        help="Process only this ID; repeat to select several documents.")
    parser.add_argument("--expected-database", default="literature_rag_dev",
                        help="Require this database name; does not change the connection.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild tracked documents even if unchanged.")
    parser.add_argument("--rebuild-untracked", action="store_true",
                        help="Explicitly rebuild legacy output with no successful hash.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts = {key: 0 for key in ("processed", "skipped", "failed", "blocked")}
    try:
        # Lazy import lets --help and pure parsing tests run without a DB driver.
        from database import create_database_connection

        # Check config before touching a database or marking any document started.
        split_text_into_chunks("configuration check", CHUNK_SIZE, CHUNK_OVERLAP)
        pipeline_version = build_pipeline_version()
        with create_database_connection() as conn:
            conn.autocommit = True
            validate_target_database(conn, args.expected_database)
            acquire_pipeline_lock(conn)
            documents = get_documents_to_process(conn, args.document_id)
            if args.document_id:
                missing = set(args.document_id) - {d["document_id"] for d in documents}
                if missing:
                    raise RuntimeError(f"Document IDs not found: {sorted(missing)}")
            if not documents:
                LOGGER.info("No registered documents. Prepare a demo PDF before testing.")
            LOGGER.info("Documents=%s pipeline_version=%s", len(documents), pipeline_version)
            for document in documents:
                result = process_one_document(
                    conn, document, pipeline_version, args.force, args.rebuild_untracked
                )
                counts[result] += 1
                LOGGER.info("document_id=%s result=%s", document["document_id"], result)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Completed documents remain saved; retry this command.")
        return 130
    except Exception as error:
        LOGGER.error("Pipeline stopped: %s: %s", type(error).__name__, error)
        return 2
    LOGGER.info("SUMMARY %s", " ".join(f"{k}={v}" for k, v in counts.items()))
    return 1 if counts["failed"] or counts["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
