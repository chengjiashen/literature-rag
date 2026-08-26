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
    - Section detection is rule-based and intentionally conservative.
    - Chunks are character-based rather than token-based.
    - Page ranges are currently stored as NULL and can be added later.
"""

import re
from pathlib import Path

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from database import create_database_connection

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


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

def extract_pdf_text(pdf_path):
    """
    Extract plain text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A single string containing extracted text from all pages.
        Synthetic page markers are included for debugging and future
        page-level metadata support.
    """
    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as document:
        all_pages = []

        for page_index in range(document.page_count):
            page = document[page_index]
            text = page.get_text("text", sort=True)

            page_text = (
                f"\n\n===== Page {page_index + 1} =====\n\n{text}"
            )
            all_pages.append(page_text)

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

def get_documents_to_process(conn):
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
            document_id,
            paper_id,
            file_path
        FROM documents
        ORDER BY document_id;
    """

    with conn.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()

    return [
        {
            "document_id": row[0],
            "paper_id": row[1],
            "file_path": row[2],
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
            DELETE FROM chunks
            WHERE document_id = %s;
            """,
            (document_id,),
        )

        cursor.execute(
            """
            DELETE FROM sections
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
        INSERT INTO sections (
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
        INSERT INTO chunks (
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

def process_one_document(
    conn,
    document,
):
    """
    Process one PDF from text extraction through database insertion.

    Args:
        conn: Active PostgreSQL database connection.
        document: Dictionary containing document_id, paper_id,
            and file_path.
    """
    document_id = document["document_id"]
    paper_id = document["paper_id"]
    file_path = document["file_path"]

    print("\n" + "=" * 80)
    print(
        f"Processing document_id={document_id}, "
        f"paper_id={paper_id}"
    )
    print(f"PDF path: {file_path}")
    print("=" * 80)

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        print(
            f"File not found. Skipping: {pdf_path}"
        )
        return

    full_text = extract_pdf_text(
        pdf_path
    )

    if not full_text.strip():
        print(
            f"No text extracted. Skipping: {pdf_path}"
        )
        return

    sections = split_into_sections(
        full_text
    )

    print(
        f"Detected sections: {len(sections)}"
    )

    delete_old_sections_and_chunks(
        conn=conn,
        document_id=document_id,
    )

    total_chunks = 0

    for section_index, section in enumerate(
        sections,
        start=1,
    ):
        section_title = section["heading"]
        section_content = section["content"]

        section_id = insert_section(
            conn=conn,
            paper_id=paper_id,
            document_id=document_id,
            section_order=section_index,
            section_title=section_title,
        )

        chunks = split_text_into_chunks(
            section_content
        )

        for chunk_index, chunk_text in enumerate(
            chunks,
            start=1,
        ):
            insert_chunk(
                conn=conn,
                paper_id=paper_id,
                document_id=document_id,
                section_id=section_id,
                chunk_order=chunk_index,
                chunk_text=chunk_text,
            )

        total_chunks += len(chunks)

        print(
            f"  Section {section_index}: "
            f"{section_title} | "
            f"characters={len(section_content)} | "
            f"chunks={len(chunks)}"
        )

    print(
        f"Completed document_id={document_id}: "
        f"sections={len(sections)}, "
        f"chunks={total_chunks}"
    )


# ============================================================
# Entry point
# ============================================================

def main():
    """
    Run the PDF-to-chunk ETL pipeline for all registered documents.

    Each document is committed independently. If processing one document
    fails, its transaction is rolled back without discarding previously
    completed documents.
    """
    with create_database_connection() as conn:
        documents = get_documents_to_process(
            conn
        )

        print(
            f"Found {len(documents)} documents "
            "in the database."
        )

        for document in documents:
            try:
                process_one_document(
                    conn=conn,
                    document=document,
                )

                # Commit after each document so one failed PDF does not
                # discard successfully processed previous documents.
                conn.commit()

            except Exception as error:
                conn.rollback()

                print(
                    "\nProcessing failed. "
                    "Transaction rolled back."
                )

                print(
                    f"document_id: "
                    f"{document['document_id']}"
                )

                print(
                    f"error: {error}"
                )


if __name__ == "__main__":
    main()
