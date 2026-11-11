"""Paper-centric literature search and evidence-grounded RAG answering.

Uses shared project configuration and database connectivity while keeping
retrieval, reranking, evidence-window construction, and generation logic local.
"""

import re
from typing import Any

import numpy as np
import psycopg
from openai import OpenAI

from config import (
    CANDIDATE_K,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GENERATION_REASONING_EFFORT,
    MAX_ANSWER_TOKENS,
    MAX_CONTEXT_CHARACTERS,
    MAX_EVIDENCE_WINDOWS,
    NEIGHBOR_RADIUS,
    OPENAI_API_KEY,
    SHOW_FULL_EVIDENCE_WINDOWS,
)
from database import create_database_connection


# =========================================================
# Question analysis rules
# =========================================================

QUESTION_TYPE_RULES = [
    (
        "operationalization",
        [
            "operationalization",
            "operationalisation",
            "operationalize",
            "operationalise",
            "operationalized",
            "operationalised",
            "how was",
            "how is",
            "measured",
            "measure",
            "calculated",
            "calculation",
            "index",
            "指标",
            "操作化",
            "如何测量",
            "怎么测量",
            "如何计算",
        ],
    ),
    (
        "limitations",
        [
            "limitation",
            "limitations",
            "weakness",
            "weaknesses",
            "future research",
            "caution",
            "局限",
            "局限性",
            "限制",
            "未来研究",
        ],
    ),
    (
        "research_questions",
        [
            "research question",
            "research questions",
            "research aim",
            "research aims",
            "objective",
            "objectives",
            "purpose of the study",
            "研究问题",
            "研究目的",
        ],
    ),
    (
        "data",
        [
            "dataset",
            "data set",
            "data",
            "corpus",
            "participants",
            "participant",
            "sample",
            "learners",
            "数据",
            "语料库",
            "参与者",
            "样本",
        ],
    ),
    (
        "method",
        [
            "methodology",
            "method",
            "methods",
            "procedure",
            "research design",
            "statistical analysis",
            "方法",
            "研究设计",
            "分析方法",
            "研究程序",
        ],
    ),
    (
        "discussion",
        [
            "interpret",
            "interpretation",
            "explain",
            "explanation",
            "discussion",
            "why did",
            "why does",
            "如何解释",
            "怎么解释",
            "为什么",
            "讨论",
        ],
    ),
    (
        "results",
        [
            "result",
            "results",
            "finding",
            "findings",
            "show",
            "shows",
            "showed",
            "found",
            "demonstrate",
            "结果",
            "发现",
            "主要发现",
        ],
    ),
    (
        "summary",
        [
            "summarize",
            "summary",
            "overview",
            "main point",
            "main points",
            "总结",
            "概述",
            "主要内容",
        ],
    ),
]


QUERY_EXPANSIONS = {
    "results": (
        " Main findings, statistically significant and non-significant "
        "results, observed patterns, differences between proficiency "
        "groups, effect sizes, discussion, and conclusions."
    ),
    "operationalization": (
        " Definition and operationalization of the target construct, "
        "exact measure or index, formula, calculation procedure, "
        "normalization, unit of analysis, reference corpus, thresholds, "
        "and variable construction."
    ),
    "method": (
        " Research design, methodology, procedure, variables, measures, "
        "statistical analysis, annotation, and analytical steps."
    ),
    "data": (
        " Dataset, learner corpus, participants, sample size, proficiency "
        "levels, language backgrounds, tasks, genres, and data collection."
    ),
    "limitations": (
        " Limitations, cautions, weaknesses, restrictions on interpretation, "
        "generalizability, and recommendations for future research."
    ),
    "discussion": (
        " Authors' interpretation, explanations for the findings, "
        "theoretical implications, comparison with previous research, "
        "and alternative explanations."
    ),
    "research_questions": (
        " Research aims, research questions, objectives, hypotheses, "
        "and purpose of the study."
    ),
    "summary": (
        " Research purpose, data, methods, measures, main findings, "
        "discussion, conclusions, and limitations."
    ),
}


SECTION_BOOST_RULES = {
    "results": [
        (("results", "findings"), 0.18),
        (("discussion",), 0.12),
        (("conclusion", "conclusions"), 0.11),
        (("abstract",), 0.04),
    ],
    "operationalization": [
        (("method", "methods", "methodology"), 0.18),
        (("measure", "measures"), 0.17),
        (("analysis", "analyses"), 0.14),
        (("data",), 0.07),
        (("appendix",), 0.04),
    ],
    "method": [
        (("method", "methods", "methodology"), 0.18),
        (("analysis", "analyses"), 0.14),
        (("data",), 0.09),
        (("appendix",), 0.04),
    ],
    "data": [
        (("data", "learner data"), 0.18),
        (("participants", "sample", "corpus"), 0.16),
        (("method", "methods"), 0.08),
    ],
    "limitations": [
        (("limitation", "limitations"), 0.18),
        (("discussion",), 0.14),
        (("conclusion", "conclusions"), 0.12),
    ],
    "discussion": [
        (("discussion",), 0.18),
        (("conclusion", "conclusions"), 0.10),
        (("results",), 0.06),
    ],
    "research_questions": [
        (("introduction",), 0.14),
        (("research question", "research questions"), 0.18),
        (("aim", "aims", "objectives"), 0.16),
        (("abstract",), 0.08),
    ],
    "summary": [
        (("abstract",), 0.14),
        (("conclusion", "conclusions"), 0.12),
        (("discussion",), 0.09),
        (("results",), 0.08),
        (("introduction",), 0.06),
        (("method", "methods"), 0.04),
    ],
}


ANCHOR_COUNTS = {
    "results": 6,
    "operationalization": 4,
    "method": 4,
    "data": 4,
    "limitations": 4,
    "discussion": 5,
    "research_questions": 4,
    "summary": 7,
    "general": 5,
}


ANSWER_FORMATS = {
    "results": (
        "Organize the answer under: Direct answer; Main findings; "
        "Statistical evidence (only when present); Authors' interpretation; "
        "Qualifications or limitations in the evidence."
    ),
    "operationalization": (
        "Organize the answer under: Construct; Operationalization; "
        "Exact measure or index; Calculation or normalization; "
        "Unit of analysis; Reference corpus, threshold, or resource; "
        "Variants or caveats. Omit unsupported headings."
    ),
    "method": (
        "Organize the answer under: Research design; Data or participants; "
        "Measures; Procedure; Analysis. Omit unsupported headings."
    ),
    "data": (
        "Organize the answer under: Dataset or corpus; Participants; "
        "Tasks or texts; Proficiency or background; Sample size; "
        "Data collection. Omit unsupported headings."
    ),
    "limitations": (
        "Organize the answer under: Main limitations; Consequences for "
        "interpretation or generalizability; Future research proposed."
    ),
    "discussion": (
        "Organize the answer under: Authors' interpretation; Explanations; "
        "Theoretical or practical implications; Comparison with previous "
        "research; Caveats."
    ),
    "research_questions": (
        "List the research aims, research questions, or hypotheses exactly "
        "as supported by the evidence, then briefly explain their scope."
    ),
    "summary": (
        "Organize the answer under: Purpose; Data and method; Main findings; "
        "Interpretation; Limitations. Keep it compact."
    ),
    "general": (
        "Give a direct answer first, followed by the most relevant "
        "supporting details and any evidence limitations."
    ),
}


# =========================================================
# API configuration validation
# =========================================================


def validate_openai_configuration() -> None:
    """Check the API key; database.py validates PostgreSQL settings."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Missing required environment variable: OPENAI_API_KEY"
        )


# =========================================================
# User input
# =========================================================


def get_non_empty_input(prompt: str) -> str:
    """Ask the user for a non-empty input value."""
    while True:
        value = input(prompt).strip()

        if value:
            return value

        print("The value cannot be empty. Please try again.")



def get_publication_year() -> int:
    """Ask the user for a valid four-digit publication year."""
    while True:
        year_text = input("Enter publication year:\n> ").strip()

        if year_text.isdigit() and len(year_text) == 4:
            return int(year_text)

        print(
            "Please enter a valid four-digit year, "
            "for example 2019."
        )


# =========================================================
# Paper resolution
# =========================================================


def find_matching_papers(
    connection: psycopg.Connection,
    author_surname: str,
    publication_year: int,
) -> list[tuple[Any, ...]]:
    """Find papers matching an author surname and publication year."""
    query = """
        WITH matching_papers AS (
            SELECT DISTINCT
                p.paper_id

            FROM papers AS p

            JOIN paper_authors AS pa_match
                ON pa_match.paper_id = p.paper_id

            JOIN authors AS a_match
                ON a_match.author_id = pa_match.author_id

            WHERE p.publication_year = %s
              AND (
                    COALESCE(a_match.family_name, '') ILIKE %s
                    OR COALESCE(a_match.full_name, '') ILIKE %s
              )
        ),

        author_list AS (
            SELECT
                pa.paper_id,

                STRING_AGG(
                    COALESCE(
                        NULLIF(BTRIM(a.full_name), ''),
                        NULLIF(
                            BTRIM(
                                CONCAT_WS(
                                    ' ',
                                    a.given_name,
                                    a.family_name
                                )
                            ),
                            ''
                        ),
                        'Unknown author'
                    ),
                    ', '
                    ORDER BY pa.author_order
                ) AS authors

            FROM paper_authors AS pa

            JOIN authors AS a
                ON a.author_id = pa.author_id

            GROUP BY pa.paper_id
        )

        SELECT
            p.paper_id,
            p.title,
            p.publication_year,
            COALESCE(al.authors, 'Unknown author') AS authors,
            p.doi,
            p.file_path

        FROM matching_papers AS mp

        JOIN papers AS p
            ON p.paper_id = mp.paper_id

        LEFT JOIN author_list AS al
            ON al.paper_id = p.paper_id

        ORDER BY
            p.title,
            p.paper_id;
    """

    search_pattern = f"%{author_surname}%"

    with connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                publication_year,
                search_pattern,
                search_pattern,
            ),
        )
        return cursor.fetchall()



def choose_paper(
    matching_papers: list[tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Automatically select one match or ask the user to choose."""
    if not matching_papers:
        raise LookupError(
            "No paper matched the supplied author and year."
        )

    if len(matching_papers) == 1:
        print()
        print("One matching paper was found.")
        return matching_papers[0]

    print()
    print(
        f"Multiple matching papers were found: "
        f"{len(matching_papers)}"
    )
    print()

    for index, paper in enumerate(matching_papers, start=1):
        (
            paper_id,
            title,
            publication_year,
            authors,
            doi,
            _file_path,
        ) = paper

        print(f"{index}. {title}")
        print(f"   Authors: {authors}")
        print(f"   Year: {publication_year}")
        print(f"   Paper ID: {paper_id}")

        if doi:
            print(f"   DOI: {doi}")

        print()

    while True:
        selection_text = input(
            "Select a paper by number:\n> "
        ).strip()

        if not selection_text.isdigit():
            print("Please enter one of the displayed numbers.")
            continue

        selection_index = int(selection_text)

        if not 1 <= selection_index <= len(matching_papers):
            print("The selected number is out of range.")
            continue

        return matching_papers[selection_index - 1]



def print_selected_paper(selected_paper: tuple[Any, ...]) -> None:
    """Display the selected paper metadata."""
    (
        paper_id,
        title,
        publication_year,
        authors,
        doi,
        file_path,
    ) = selected_paper

    print()
    print("=" * 100)
    print("SELECTED PAPER")
    print("=" * 100)
    print(f"Title: {title}")
    print(f"Authors: {authors}")
    print(f"Publication year: {publication_year}")


# =========================================================
# Question analysis
# =========================================================


def detect_question_type(query_text: str) -> str:
    """Classify the user's information need with keyword rules."""
    normalized_query = query_text.lower()

    for question_type, keywords in QUESTION_TYPE_RULES:
        if any(
            keyword in normalized_query
            for keyword in keywords
        ):
            return question_type

    return "general"



def build_search_text(
    query_text: str,
    question_type: str,
) -> str:
    """Expand the question for retrieval while preserving its meaning."""
    return query_text + QUERY_EXPANSIONS.get(question_type, "")



def normalize_section_title(section_title: str | None) -> str:
    """Normalize a section title for rule-based matching."""
    if not section_title:
        return ""

    normalized = section_title.lower()
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()



def get_section_adjustment(
    question_type: str,
    section_title: str | None,
) -> float:
    """Calculate a soft section-aware ranking adjustment."""
    normalized_section = normalize_section_title(section_title)
    positive_adjustment = 0.0

    for patterns, adjustment in SECTION_BOOST_RULES.get(
        question_type,
        [],
    ):
        if any(
            pattern in normalized_section
            for pattern in patterns
        ):
            positive_adjustment = max(
                positive_adjustment,
                adjustment,
            )

    negative_adjustment = 0.0

    if (
        "front matter" in normalized_section
        or normalized_section == "frontmatter"
    ):
        negative_adjustment -= 0.20

    if (
        "reference" in normalized_section
        or "bibliography" in normalized_section
    ):
        negative_adjustment -= 0.08

    if "acknowledg" in normalized_section:
        negative_adjustment -= 0.10

    if "title page" in normalized_section:
        negative_adjustment -= 0.15

    return positive_adjustment + negative_adjustment



def get_anchor_count(question_type: str) -> int:
    """Return the number of anchor chunks for one question type."""
    return ANCHOR_COUNTS.get(
        question_type,
        ANCHOR_COUNTS["general"],
    )


# =========================================================
# Embedding generation
# =========================================================


def generate_query_embedding(
    client: OpenAI,
    search_text: str,
) -> tuple[np.ndarray, int]:
    """Generate and validate an embedding for the search text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=search_text,
        dimensions=EMBEDDING_DIMENSION,
        encoding_format="float",
    )

    query_embedding = np.asarray(
        response.data[0].embedding,
        dtype=np.float32,
    )

    if query_embedding.ndim != 1:
        raise ValueError(
            "The query embedding is not one-dimensional."
        )

    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"The query embedding has {len(query_embedding)} "
            f"dimensions; expected "
            f"{EMBEDDING_DIMENSION}."
        )

    if not np.isfinite(query_embedding).all():
        raise ValueError(
            "The query embedding contains NaN or infinite values."
        )

    return query_embedding, response.usage.total_tokens


# =========================================================
# Paper-level database status
# =========================================================


def count_searchable_chunks(
    connection: psycopg.Connection,
    paper_id: int,
) -> int:
    """Count valid embedded chunks in the selected paper."""
    query = """
        SELECT COUNT(*)
        FROM chunks
        WHERE paper_id = %s
          AND embedding IS NOT NULL
          AND chunk_text IS NOT NULL
          AND BTRIM(chunk_text) <> '';
    """

    with connection.cursor() as cursor:
        cursor.execute(query, (paper_id,))
        return cursor.fetchone()[0]


# =========================================================
# Candidate retrieval and reranking
# =========================================================


def retrieve_candidate_chunks(
    connection: psycopg.Connection,
    paper_id: int,
    query_embedding: np.ndarray,
    candidate_k: int,
) -> list[tuple[Any, ...]]:
    """Retrieve a broad semantic candidate set from one paper."""
    query = """
        SELECT
            c.chunk_id,
            c.section_id,
            c.chunk_order,
            COALESCE(s.section_title, 'Unknown section') AS section_title,
            COALESCE(c.page_start, s.page_start) AS page_start,
            COALESCE(c.page_end, s.page_end) AS page_end,
            c.chunk_text,
            c.embedding <=> %s AS cosine_distance

        FROM chunks AS c

        LEFT JOIN sections AS s
            ON s.section_id = c.section_id

        WHERE c.paper_id = %s
          AND c.embedding IS NOT NULL
          AND c.chunk_text IS NOT NULL
          AND BTRIM(c.chunk_text) <> ''

        ORDER BY c.embedding <=> %s
        LIMIT %s;
    """

    with connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                query_embedding,
                paper_id,
                query_embedding,
                candidate_k,
            ),
        )
        return cursor.fetchall()



def rerank_candidates(
    candidate_rows: list[tuple[Any, ...]],
    question_type: str,
) -> list[dict[str, Any]]:
    """Combine semantic similarity with section-aware adjustments."""
    reranked_candidates: list[dict[str, Any]] = []

    for row in candidate_rows:
        (
            chunk_id,
            section_id,
            chunk_order,
            section_title,
            page_start,
            page_end,
            chunk_text,
            cosine_distance,
        ) = row

        cosine_distance = float(cosine_distance)
        cosine_similarity = 1.0 - cosine_distance

        section_adjustment = get_section_adjustment(
            question_type=question_type,
            section_title=section_title,
        )

        ranking_score = cosine_similarity + section_adjustment

        reranked_candidates.append(
            {
                "chunk_id": chunk_id,
                "section_id": section_id,
                "chunk_order": chunk_order,
                "section_title": section_title,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_text": chunk_text,
                "cosine_distance": cosine_distance,
                "cosine_similarity": cosine_similarity,
                "section_adjustment": section_adjustment,
                "ranking_score": ranking_score,
            }
        )

    return sorted(
        reranked_candidates,
        key=lambda candidate: candidate["ranking_score"],
        reverse=True,
    )



def select_anchor_chunks(
    reranked_candidates: list[dict[str, Any]],
    anchor_count: int,
) -> list[dict[str, Any]]:
    """Select diverse anchor chunks while avoiding near-duplicates."""
    selected_anchors: list[dict[str, Any]] = []
    minimum_order_gap = NEIGHBOR_RADIUS * 2

    for candidate in reranked_candidates:
        overlaps_existing_anchor = any(
            candidate["section_id"] == selected["section_id"]
            and abs(
                candidate["chunk_order"]
                - selected["chunk_order"]
            ) <= minimum_order_gap
            for selected in selected_anchors
        )

        if overlaps_existing_anchor:
            continue

        selected_anchors.append(candidate)

        if len(selected_anchors) >= anchor_count:
            break

    return selected_anchors


# =========================================================
# Neighbor expansion and window merging
# =========================================================


def fetch_neighbor_chunks(
    connection: psycopg.Connection,
    paper_id: int,
    anchor: dict[str, Any],
    radius: int,
) -> list[dict[str, Any]]:
    """Fetch the anchor chunk and its neighbors in the same section."""
    minimum_order = max(0, anchor["chunk_order"] - radius)
    maximum_order = anchor["chunk_order"] + radius

    query = """
        SELECT
            c.chunk_id,
            c.section_id,
            c.chunk_order,
            COALESCE(s.section_title, 'Unknown section') AS section_title,
            COALESCE(c.page_start, s.page_start) AS page_start,
            COALESCE(c.page_end, s.page_end) AS page_end,
            c.chunk_text

        FROM chunks AS c

        LEFT JOIN sections AS s
            ON s.section_id = c.section_id

        WHERE c.paper_id = %s
          AND c.section_id IS NOT DISTINCT FROM %s
          AND c.chunk_order BETWEEN %s AND %s
          AND c.chunk_text IS NOT NULL
          AND BTRIM(c.chunk_text) <> ''

        ORDER BY c.chunk_order;
    """

    with connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                paper_id,
                anchor["section_id"],
                minimum_order,
                maximum_order,
            ),
        )
        rows = cursor.fetchall()

    chunks: list[dict[str, Any]] = []

    for row in rows:
        (
            chunk_id,
            section_id,
            chunk_order,
            section_title,
            page_start,
            page_end,
            chunk_text,
        ) = row

        chunks.append(
            {
                "chunk_id": chunk_id,
                "section_id": section_id,
                "chunk_order": chunk_order,
                "section_title": section_title,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_text": chunk_text,
            }
        )

    return chunks



def build_initial_windows(
    connection: psycopg.Connection,
    paper_id: int,
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one evidence window around each anchor chunk."""
    windows: list[dict[str, Any]] = []

    for anchor in anchors:
        chunks = fetch_neighbor_chunks(
            connection=connection,
            paper_id=paper_id,
            anchor=anchor,
            radius=NEIGHBOR_RADIUS,
        )

        if not chunks:
            continue

        windows.append(
            {
                "section_id": anchor["section_id"],
                "section_title": anchor["section_title"],
                "start_order": min(
                    chunk["chunk_order"]
                    for chunk in chunks
                ),
                "end_order": max(
                    chunk["chunk_order"]
                    for chunk in chunks
                ),
                "chunks": chunks,
                "anchor_chunk_ids": [anchor["chunk_id"]],
                "best_cosine_similarity": (
                    anchor["cosine_similarity"]
                ),
                "best_ranking_score": anchor["ranking_score"],
            }
        )

    return windows



def merge_overlapping_windows(
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge evidence windows that overlap or touch in one section."""
    if not windows:
        return []

    windows_by_document_order = sorted(
        windows,
        key=lambda window: (
            window["section_id"]
            if window["section_id"] is not None
            else -1,
            window["start_order"],
        ),
    )

    merged_windows: list[dict[str, Any]] = []

    for window in windows_by_document_order:
        if not merged_windows:
            merged_windows.append(
                {
                    **window,
                    "chunks": list(window["chunks"]),
                    "anchor_chunk_ids": list(
                        window["anchor_chunk_ids"]
                    ),
                }
            )
            continue

        previous_window = merged_windows[-1]
        same_section = (
            previous_window["section_id"]
            == window["section_id"]
        )
        ranges_overlap_or_touch = (
            window["start_order"]
            <= previous_window["end_order"] + 1
        )

        if same_section and ranges_overlap_or_touch:
            chunk_map = {
                chunk["chunk_id"]: chunk
                for chunk in previous_window["chunks"]
            }

            for chunk in window["chunks"]:
                chunk_map[chunk["chunk_id"]] = chunk

            previous_window["chunks"] = sorted(
                chunk_map.values(),
                key=lambda chunk: chunk["chunk_order"],
            )
            previous_window["start_order"] = min(
                previous_window["start_order"],
                window["start_order"],
            )
            previous_window["end_order"] = max(
                previous_window["end_order"],
                window["end_order"],
            )
            previous_window["anchor_chunk_ids"] = sorted(
                set(
                    previous_window["anchor_chunk_ids"]
                    + window["anchor_chunk_ids"]
                )
            )
            previous_window["best_cosine_similarity"] = max(
                previous_window["best_cosine_similarity"],
                window["best_cosine_similarity"],
            )
            previous_window["best_ranking_score"] = max(
                previous_window["best_ranking_score"],
                window["best_ranking_score"],
            )
        else:
            merged_windows.append(
                {
                    **window,
                    "chunks": list(window["chunks"]),
                    "anchor_chunk_ids": list(
                        window["anchor_chunk_ids"]
                    ),
                }
            )

    merged_windows.sort(
        key=lambda window: window["best_ranking_score"],
        reverse=True,
    )

    return merged_windows[:MAX_EVIDENCE_WINDOWS]


# =========================================================
# Formatting and retrieval diagnostics
# =========================================================


def format_page_range(
    page_start: int | None,
    page_end: int | None,
) -> str:
    """Format optional page information."""
    if page_start is None and page_end is None:
        return "Unknown"

    if page_start is None:
        return str(page_end)

    if page_end is None:
        return str(page_start)

    if page_start == page_end:
        return str(page_start)

    return f"{page_start}-{page_end}"



def get_window_page_range(
    chunks: list[dict[str, Any]],
) -> str:
    """Calculate a combined page range for one evidence window."""
    page_starts = [
        chunk["page_start"]
        for chunk in chunks
        if chunk["page_start"] is not None
    ]
    page_ends = [
        chunk["page_end"]
        for chunk in chunks
        if chunk["page_end"] is not None
    ]

    if not page_starts and not page_ends:
        return "Unknown"

    page_start = (
        min(page_starts)
        if page_starts
        else min(page_ends)
    )
    page_end = (
        max(page_ends)
        if page_ends
        else max(page_starts)
    )

    return format_page_range(page_start, page_end)



# def print_selected_anchors(
#     anchors: list[dict[str, Any]],
# ) -> None:
#     """Display selected anchors after reranking."""
#     print()
#     print("=" * 100)
#     print("SELECTED ANCHOR CHUNKS")
#     print("=" * 100)
#
#     for rank, anchor in enumerate(anchors, start=1):
#         print()
#         print(f"Anchor rank: {rank}")
#         print(f"Chunk ID: {anchor['chunk_id']}")
#         print(f"Section: {anchor['section_title']}")
#         print(f"Chunk order: {anchor['chunk_order']}")
#         print(
#             "Cosine similarity: "
#             f"{anchor['cosine_similarity']:.6f}"
#         )
#         print(
#             "Section adjustment: "
#             f"{anchor['section_adjustment']:+.3f}"
#         )
#         print(
#             "Final ranking score: "
#             f"{anchor['ranking_score']:.6f}"
#         )



# def print_evidence_window_summary(
#     windows: list[dict[str, Any]],
# ) -> None:
#     """Display compact evidence-window metadata."""
#     print()
#     print("=" * 100)
#     print("EVIDENCE WINDOW SUMMARY")
#     print("=" * 100)
#
#     for index, window in enumerate(windows, start=1):
#         chunk_ids = [
#             chunk["chunk_id"]
#             for chunk in window["chunks"]
#         ]
#
#         print()
#         print(f"Window {index}")
#         print(f"Section: {window['section_title']}")
#         print(
#             "Chunk IDs: "
#             + ", ".join(str(chunk_id) for chunk_id in chunk_ids)
#         )
#         print(
#             "Chunk-order range: "
#             f"{window['start_order']}-{window['end_order']}"
#         )
#         print(
#             "Pages: "
#             f"{get_window_page_range(window['chunks'])}"
#         )
#         print(
#             "Best ranking score: "
#             f"{window['best_ranking_score']:.6f}"
#         )



def print_full_evidence_windows(
    windows: list[dict[str, Any]],
) -> None:
    """Display full evidence text for retrieval debugging."""
    print()
    print("=" * 100)
    print("FULL EVIDENCE WINDOWS")
    print("=" * 100)

    for window_number, window in enumerate(windows, start=1):
        print()
        print("#" * 100)
        print(f"EVIDENCE WINDOW {window_number}")
        print("#" * 100)
        print(f"Section: {window['section_title']}")
        print(
            "Pages: "
            f"{get_window_page_range(window['chunks'])}"
        )

        for chunk in window["chunks"]:
            print()
            print("-" * 100)
            print(
                f"CHUNK {chunk['chunk_id']} "
                f"(order {chunk['chunk_order']})"
            )
            print("-" * 100)
            print(chunk["chunk_text"])


# =========================================================
# Evidence-context construction
# =========================================================


def format_window_for_llm(
    window: dict[str, Any],
    window_number: int,
) -> str:
    """Convert one evidence window into a traceable LLM context block."""
    chunk_ids = [
        chunk["chunk_id"]
        for chunk in window["chunks"]
    ]

    lines = [
        f"[EVIDENCE WINDOW {window_number}]",
        f"Section: {window['section_title']}",
        (
            "Chunk IDs: "
            + ", ".join(str(chunk_id) for chunk_id in chunk_ids)
        ),
        (
            "Chunk-order range: "
            f"{window['start_order']}-{window['end_order']}"
        ),
        f"Pages: {get_window_page_range(window['chunks'])}",
        "",
    ]

    for chunk in window["chunks"]:
        lines.extend(
            [
                (
                    f"[CHUNK {chunk['chunk_id']} | "
                    f"ORDER {chunk['chunk_order']}]"
                ),
                chunk["chunk_text"].strip(),
                "",
            ]
        )

    return "\n".join(lines).strip()



def build_evidence_context(
    windows: list[dict[str, Any]],
    max_characters: int,
) -> tuple[str, list[dict[str, Any]], bool]:
    """
    Build an LLM context without cutting evidence windows in half.

    Returns:
        context_text,
        included_windows,
        was_limited
    """
    context_parts: list[str] = []
    included_windows: list[dict[str, Any]] = []
    current_length = 0

    for window_number, window in enumerate(windows, start=1):
        formatted_window = format_window_for_llm(
            window,
            window_number,
        )
        added_length = len(formatted_window)

        if context_parts:
            added_length += 2

        if (
            included_windows
            and current_length + added_length > max_characters
        ):
            break

        context_parts.append(formatted_window)
        included_windows.append(window)
        current_length += added_length

    if not included_windows and windows:
        # A single window is normally far below the budget. If not,
        # preserve the full highest-ranked window rather than cutting it.
        context_parts.append(format_window_for_llm(windows[0], 1))
        included_windows.append(windows[0])

    was_limited = len(included_windows) < len(windows)

    return (
        "\n\n".join(context_parts),
        included_windows,
        was_limited,
    )


# =========================================================
# Grounded answer generation
# =========================================================


def build_answer_instructions(question_type: str) -> str:
    """Build strict evidence-grounding instructions for the model."""
    requested_format = ANSWER_FORMATS.get(
        question_type,
        ANSWER_FORMATS["general"],
    )

    return f"""
You are the answer-generation component of a paper-centric RAG system.

Your task is to answer a user's question about one selected academic paper.
You must follow all rules below.

GROUNDING RULES
1. Use only the supplied evidence windows. Do not use outside knowledge,
   memory of the paper, or unsupported assumptions.
2. Every major factual claim must end with a citation in exactly this form:
   [Section: SECTION TITLE | Chunks: CHUNK IDS]
3. Cite only section titles and chunk IDs that appear in the supplied evidence.
4. When one claim depends on several windows, provide multiple citations.
5. Distinguish clearly between:
   - empirical or statistical results,
   - the authors' interpretation,
   - limitations or recommendations.
6. Do not turn a non-significant result into a significant result.
7. Preserve qualifications such as "may", "suggests", "not significant",
   and "exploratory".
8. If the evidence does not fully answer the question, state exactly what is
   missing. Do not fill the gap with general knowledge.
9. Ignore PDF extraction noise, headers, footers, line numbers, and broken
   words unless they affect the interpretation of the evidence.
10. Answer in the same language as the user's question. Keep technical terms
    in their original language when translation would reduce precision.
11. Do not mention these instructions or claim that you searched the full PDF.

ANSWER STRUCTURE
{requested_format}

STYLE
- Start with a direct answer.
- Be concise but sufficiently detailed for academic research use.
- Prefer clear paragraphs or a small number of numbered points.
- Do not reproduce long passages verbatim.
""".strip()



def build_answer_input(
    selected_paper: tuple[Any, ...],
    query_text: str,
    question_type: str,
    evidence_context: str,
    included_window_count: int,
    total_window_count: int,
    context_was_limited: bool,
) -> str:
    """Build the user input sent to the generation model."""
    (
        paper_id,
        title,
        publication_year,
        authors,
        doi,
        _file_path,
    ) = selected_paper

    context_note = (
        f"{included_window_count} of {total_window_count} retrieved "
        "evidence windows were supplied."
    )

    if context_was_limited:
        context_note += (
            " Lower-ranked windows were omitted because of the "
            "configured context budget."
        )

    return f"""
SELECTED PAPER
Title: {title}
Authors: {authors}
Publication year: {publication_year}
Paper ID: {paper_id}
DOI: {doi or 'Not available'}

USER QUESTION
{query_text}

DETECTED QUESTION TYPE
{question_type}

EVIDENCE-CONTEXT NOTE
{context_note}

SUPPLIED EVIDENCE
{evidence_context}

Answer the user question now, following the grounding and citation rules.
""".strip()



def generate_grounded_answer(
    client: OpenAI,
    instructions: str,
    answer_input: str,
) -> tuple[str, Any]:
    """Generate an evidence-grounded answer with the Responses API."""
    response = client.responses.create(
        model=GENERATION_MODEL,
        reasoning={
            "effort": GENERATION_REASONING_EFFORT,
        },
        instructions=instructions,
        input=answer_input,
        max_output_tokens=MAX_ANSWER_TOKENS,
        store=False,
    )

    answer = response.output_text.strip()

    if not answer:
        raise RuntimeError(
            "The generation model returned an empty answer."
        )

    return answer, response.usage



def get_usage_value(
    usage: Any,
    attribute_name: str,
) -> int | None:
    """Safely read a token-usage attribute from an SDK response."""
    if usage is None:
        return None

    value = getattr(usage, attribute_name, None)

    if value is None and isinstance(usage, dict):
        value = usage.get(attribute_name)

    return value



def print_generated_answer(
    answer: str,
    usage: Any,
) -> None:
    """Display the grounded answer and token usage."""
    print()
    print("=" * 100)
    print("GROUNDED ANSWER")
    print("=" * 100)
    print(answer)

    input_tokens = get_usage_value(usage, "input_tokens")
    output_tokens = get_usage_value(usage, "output_tokens")
    total_tokens = get_usage_value(usage, "total_tokens")

    # print()
    # print("=" * 100)
    # print("GENERATION USAGE")
    # print("=" * 100)
    # print(f"Generation model: {GENERATION_MODEL}")
    # print(f"Reasoning effort: {GENERATION_REASONING_EFFORT}")
    # print(
    #     "Input tokens: "
    #     f"{input_tokens if input_tokens is not None else 'Not reported'}"
    # )
    # print(
    #     "Output tokens: "
    #     f"{output_tokens if output_tokens is not None else 'Not reported'}"
    # )
    # print(
    #     "Total tokens: "
    #     f"{total_tokens if total_tokens is not None else 'Not reported'}"
    # )


# =========================================================
# Main program
# =========================================================


def main() -> None:
    """
    Run paper-centric Retrieval-Augmented Generation.

    Workflow:
        1. Resolve a paper from author surname and publication year.
        2. Detect the question type.
        3. Generate a retrieval embedding.
        4. Retrieve and rerank chunks within the selected paper.
        5. Expand anchors with neighboring chunks.
        6. Merge overlapping evidence windows.
        7. Build a bounded evidence context.
        8. Ask the generation model to answer only from that evidence.
    """
    validate_openai_configuration()

    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=5,
        timeout=180.0,
    )

    connection: psycopg.Connection | None = None

    try:
        print("=" * 100)
        print("PAPER-CENTRIC RAG")
        print("=" * 100)

        author_surname = get_non_empty_input(
            "\nEnter author surname:\n> "
        )
        publication_year = get_publication_year()

        print()
        print("[1] Connecting and registering pgvector...", flush=True)

        connection = create_database_connection(
            register_pgvector=True
        )

        print("[2] Connection and registration completed.", flush=True)

        connection.commit()
        connection.autocommit = True

        print("[3] Searching for matching papers...", flush=True)

        matching_papers = find_matching_papers(
            connection=connection,
            author_surname=author_surname,
            publication_year=publication_year,
        )

        print(f"[4] Found {len(matching_papers)} matching papers.", flush=True)

        selected_paper = choose_paper(matching_papers)
        print_selected_paper(selected_paper)

        selected_paper_id = selected_paper[0]

        searchable_chunk_count = count_searchable_chunks(
            connection=connection,
            paper_id=selected_paper_id,
        )

        if searchable_chunk_count == 0:
            raise RuntimeError(
                "The selected paper has no searchable chunks."
            )

        query_text = get_non_empty_input(
            "\nEnter your question about this paper:\n> "
        )

        question_type = detect_question_type(query_text)
        search_text = build_search_text(
            query_text=query_text,
            question_type=question_type,
        )

        query_embedding, embedding_token_count = (
            generate_query_embedding(
                client=openai_client,
                search_text=search_text,
            )
        )

        actual_candidate_k = min(
            CANDIDATE_K,
            searchable_chunk_count,
        )

        candidate_rows = retrieve_candidate_chunks(
            connection=connection,
            paper_id=selected_paper_id,
            query_embedding=query_embedding,
            candidate_k=actual_candidate_k,
        )

        reranked_candidates = rerank_candidates(
            candidate_rows=candidate_rows,
            question_type=question_type,
        )

        requested_anchor_count = get_anchor_count(question_type)

        anchors = select_anchor_chunks(
            reranked_candidates=reranked_candidates,
            anchor_count=requested_anchor_count,
        )

        if not anchors:
            raise RuntimeError(
                "No anchor chunks were selected."
            )

        # if SHOW_ANCHOR_DETAILS:
        #     print_selected_anchors(anchors)

        initial_windows = build_initial_windows(
            connection=connection,
            paper_id=selected_paper_id,
            anchors=anchors,
        )

        evidence_windows = merge_overlapping_windows(
            initial_windows
        )

        if not evidence_windows:
            raise RuntimeError(
                "No evidence windows were created."
            )

        # print_evidence_window_summary(evidence_windows)

        if SHOW_FULL_EVIDENCE_WINDOWS:
            print_full_evidence_windows(evidence_windows)

        (
            evidence_context,
            included_windows,
            context_was_limited,
        ) = build_evidence_context(
            windows=evidence_windows,
            max_characters=MAX_CONTEXT_CHARACTERS,
        )

        if not evidence_context:
            raise RuntimeError(
                "The evidence context is empty."
            )

        included_chunk_ids = [
            chunk["chunk_id"]
            for window in included_windows
            for chunk in window["chunks"]
        ]

        print()
        print("=" * 100)
        print("ANSWER CONTEXT")
        print("=" * 100)
        print(
            "Evidence windows supplied: "
            f"{len(included_windows)} of {len(evidence_windows)}"
        )
        print(
            "Evidence characters supplied: "
            f"{len(evidence_context)}"
        )
        print(
            "Chunk IDs supplied: "
            + ", ".join(
                str(chunk_id)
                for chunk_id in included_chunk_ids
            )
        )
        print(
            "Context limited by budget: "
            f"{context_was_limited}"
        )

        answer_instructions = build_answer_instructions(
            question_type
        )

        answer_input = build_answer_input(
            selected_paper=selected_paper,
            query_text=query_text,
            question_type=question_type,
            evidence_context=evidence_context,
            included_window_count=len(included_windows),
            total_window_count=len(evidence_windows),
            context_was_limited=context_was_limited,
        )

        print()
        print(
            f"Generating a grounded answer with "
            f"{GENERATION_MODEL}..."
        )

        answer, generation_usage = generate_grounded_answer(
            client=openai_client,
            instructions=answer_instructions,
            answer_input=answer_input,
        )

        print_generated_answer(
            answer=answer,
            usage=generation_usage,
        )

        print()
        print("=" * 100)
        print("RAG RUN SUMMARY")
        print("=" * 100)
        print(f"Selected paper: {selected_paper[1]}")
        print(f"Paper ID: {selected_paper_id}")
        print(f"Question: {query_text}")
        print(f"Question type: {question_type}")
        print(
            "Semantic candidates retrieved: "
            f"{len(candidate_rows)}"
        )
        print(f"Anchor chunks selected: {len(anchors)}")
        print(
            "Merged evidence windows: "
            f"{len(evidence_windows)}"
        )
        print(
            "Evidence windows sent to model: "
            f"{len(included_windows)}"
        )
        print("Database updated: No")
        print("Paper-centric RAG completed successfully.")

    except Exception:
        print()
        print("=" * 100)
        print("PAPER-CENTRIC RAG FAILED")
        print("=" * 100)
        raise

    finally:
        if connection is not None:
            connection.close()
            print("PostgreSQL connection closed.")


if __name__ == "__main__":
    main()