"""
Check whether PDF files referenced in the documents table exist locally.

This script reads document paths from PostgreSQL and reports any files
that cannot be found on the local filesystem.
"""

from pathlib import Path

from database import create_database_connection


def get_document_paths(conn):
    """
    Retrieve document paths from the database.

    Args:
        conn: Active PostgreSQL database connection.

    Returns:
        A list of rows containing document_id, paper_id, and file_path.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                document_id,
                paper_id,
                file_path
            FROM documents
            ORDER BY document_id;
            """
        )

        return cur.fetchall()


def find_missing_files(documents):
    """
    Identify document records whose PDF files do not exist locally.

    Args:
        documents: Database rows containing document metadata.

    Returns:
        A list of tuples containing document_id, paper_id, and file_path
        for missing files.
    """
    missing_files = []

    for document_id, paper_id, file_path in documents:
        path = Path(file_path)

        if not path.exists():
            missing_files.append(
                (
                    document_id,
                    paper_id,
                    file_path,
                )
            )

    return missing_files


def main():
    """Run the PDF path validation process."""
    with create_database_connection() as conn:
        documents = get_document_paths(conn)

    missing_files = find_missing_files(documents)

    print(f"Documents total: {len(documents)}")
    print(f"Missing files: {len(missing_files)}")

    for item in missing_files:
        print(item)


if __name__ == "__main__":
    main()
