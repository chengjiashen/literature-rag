"""
Check whether PDF files referenced in the documents table exist locally.

This script reads document paths from PostgreSQL and reports any files
that cannot be found on the local filesystem.
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


load_dotenv()


def create_database_connection():
    """
    Create a PostgreSQL connection using environment variables.

    Returns:
        psycopg.Connection: An active PostgreSQL database connection.

    Raises:
        RuntimeError: If any required database environment variable is missing.
    """
    required_variables = [
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing_variables:
        missing = ", ".join(missing_variables)
        raise RuntimeError(
            f"Missing required environment variables: {missing}"
        )

    return psycopg.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )


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
            SELECT document_id, paper_id, file_path
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
                (document_id, paper_id, file_path)
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