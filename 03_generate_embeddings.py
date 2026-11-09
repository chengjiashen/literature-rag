import numpy as np
from openai import OpenAI

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    MAX_EMBEDDING_BATCHES,
    OPENAI_API_KEY,
)
from database import create_database_connection


# =========================================================
# Environment configuration
# =========================================================

def validate_openai_configuration():
    """Raise an error if the OpenAI API key is not configured."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing environment variable: OPENAI_API_KEY")


# =========================================================
# Database status
# =========================================================

def get_embedding_status(connection):
    """
    Return the current embedding progress in the chunks table.

    Returns:
        tuple[int, int, int, int]:
            total_chunks,
            embedded_chunks,
            pending_chunks,
            empty_text_chunks
    """
    query = """
        SELECT
            COUNT(*) AS total_chunks,

            COUNT(*) FILTER (
                WHERE embedding IS NOT NULL
            ) AS embedded_chunks,

            COUNT(*) FILTER (
                WHERE embedding IS NULL
                  AND chunk_text IS NOT NULL
                  AND BTRIM(chunk_text) <> ''
            ) AS pending_chunks,

            COUNT(*) FILTER (
                WHERE embedding IS NULL
                  AND (
                      chunk_text IS NULL
                      OR BTRIM(chunk_text) = ''
                  )
            ) AS empty_text_chunks

        FROM chunks;
    """

    with connection.cursor() as cursor:
        cursor.execute(query)
        status = cursor.fetchone()

    # End the read-only transaction.
    connection.commit()

    return status


def print_embedding_status(status, title):
    """
    Display the current embedding progress.

    Args:
        status:
            Status tuple returned by get_embedding_status().
        title:
            Heading displayed above the status.
    """
    (
        total_chunks,
        embedded_chunks,
        pending_chunks,
        empty_text_chunks,
    ) = status

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks with embeddings: {embedded_chunks}")
    print(f"Valid chunks still pending: {pending_chunks}")
    print(f"Empty-text chunks skipped: {empty_text_chunks}")


# =========================================================
# Load one batch
# =========================================================

def load_next_batch(connection, batch_size):
    """
    Load the next batch of chunks whose embeddings are still NULL.

    Existing embeddings are skipped automatically.

    Args:
        connection:
            An active PostgreSQL connection.
        batch_size:
            Maximum number of chunks to load.

    Returns:
        list[tuple[int, str]]:
            A list of (chunk_id, chunk_text) tuples.
    """
    query = """
        SELECT
            chunk_id,
            chunk_text
        FROM chunks
        WHERE embedding IS NULL
          AND chunk_text IS NOT NULL
          AND BTRIM(chunk_text) <> ''
        ORDER BY chunk_id
        LIMIT %s;
    """

    with connection.cursor() as cursor:
        cursor.execute(query, (batch_size,))
        rows = cursor.fetchall()

    # Close the read transaction before making the API request.
    connection.commit()

    return rows


# =========================================================
# Generate embeddings
# =========================================================

def generate_batch_embeddings(client, batch_rows):
    """
    Generate embeddings for one batch of chunk texts.

    Args:
        client:
            An initialized OpenAI client.
        batch_rows:
            A list of (chunk_id, chunk_text) tuples.

    Returns:
        tuple[list[numpy.ndarray], int]:
            Generated vectors and total input token count.

    Raises:
        RuntimeError:
            If the API returns an unexpected number of vectors.

        ValueError:
            If a vector has an incorrect dimension or invalid value.
    """
    chunk_texts = [
        chunk_text
        for _, chunk_text in batch_rows
    ]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunk_texts,
        dimensions=EMBEDDING_DIMENSION,
        encoding_format="float",
    )

    # Sort by API result index so every vector remains aligned
    # with its corresponding chunk text.
    sorted_results = sorted(
        response.data,
        key=lambda result: result.index,
    )

    if len(sorted_results) != len(batch_rows):
        raise RuntimeError(
            "The number of returned embeddings does not match "
            "the number of submitted chunks."
        )

    embedding_arrays = []

    for (chunk_id, _), result in zip(
        batch_rows,
        sorted_results,
    ):
        embedding_array = np.asarray(
            result.embedding,
            dtype=np.float32,
        )

        if embedding_array.ndim != 1:
            raise ValueError(
                f"Chunk {chunk_id} returned a non-1D embedding."
            )

        if len(embedding_array) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Chunk {chunk_id} returned "
                f"{len(embedding_array)} dimensions; "
                f"expected {EMBEDDING_DIMENSION}."
            )

        if not np.isfinite(embedding_array).all():
            raise ValueError(
                f"Chunk {chunk_id} contains NaN "
                "or infinite values."
            )

        embedding_arrays.append(embedding_array)

    total_tokens = response.usage.total_tokens

    return embedding_arrays, total_tokens


# =========================================================
# Save one batch
# =========================================================

def save_batch_embeddings(
    connection,
    batch_rows,
    embedding_arrays,
):
    """
    Save one batch of embeddings to PostgreSQL.

    The SQL condition `AND embedding IS NULL` prevents existing
    embeddings from being overwritten.

    Args:
        connection:
            An active PostgreSQL connection.
        batch_rows:
            A list of (chunk_id, chunk_text) tuples.
        embedding_arrays:
            Embeddings corresponding to batch_rows.
    """
    if len(batch_rows) != len(embedding_arrays):
        raise RuntimeError(
            "Chunk count and embedding count do not match."
        )

    update_query = """
        UPDATE chunks
        SET embedding = %s
        WHERE chunk_id = %s
          AND embedding IS NULL;
    """

    update_values = [
        (
            embedding_array,
            chunk_id,
        )
        for (chunk_id, _), embedding_array in zip(
            batch_rows,
            embedding_arrays,
        )
    ]

    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                update_query,
                update_values,
            )

        # Permanently save this completed batch.
        connection.commit()

    except Exception:
        # Undo the entire current batch if its database update fails.
        connection.rollback()
        raise


# =========================================================
# Verify one saved batch
# =========================================================

def verify_saved_batch(connection, chunk_ids):
    """
    Verify that every chunk in a completed batch has a valid vector.

    Args:
        connection:
            An active PostgreSQL connection.
        chunk_ids:
            IDs of the chunks in the completed batch.

    Raises:
        RuntimeError:
            If one or more embeddings are missing or have the
            wrong dimension.
    """
    query = """
        SELECT
            COUNT(*) AS batch_size,

            COUNT(*) FILTER (
                WHERE embedding IS NOT NULL
            ) AS saved_count,

            COUNT(*) FILTER (
                WHERE embedding IS NOT NULL
                  AND vector_dims(embedding) = %s
            ) AS correct_dimension_count

        FROM chunks
        WHERE chunk_id = ANY(%s);
    """

    with connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                EMBEDDING_DIMENSION,
                chunk_ids,
            ),
        )

        (
            batch_size,
            saved_count,
            correct_dimension_count,
        ) = cursor.fetchone()

    connection.commit()

    expected_count = len(chunk_ids)

    if batch_size != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} database rows, "
            f"but found {batch_size}."
        )

    if saved_count != expected_count:
        raise RuntimeError(
            f"Only {saved_count} of {expected_count} embeddings "
            "were saved."
        )

    if correct_dimension_count != expected_count:
        raise RuntimeError(
            "One or more saved embeddings have the wrong dimension."
        )


# =========================================================
# Main program
# =========================================================

def main():
    """
    Generate embeddings for all valid chunks that currently have
    a NULL embedding.

    Workflow:
        1. Check configuration.
        2. Connect to PostgreSQL.
        3. Count completed and pending chunks.
        4. Load pending chunks in batches.
        5. Generate embeddings for each batch.
        6. Validate and save each batch.
        7. Commit after every completed batch.
        8. Display final progress.

    Existing embeddings are not regenerated or overwritten.
    """
    validate_openai_configuration()

    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,

        # Retry temporary connection, rate-limit, and server errors.
        max_retries=5,

        # Maximum time allowed for one API request.
        timeout=120.0,
    )

    connection = None

    processed_this_run = 0
    total_tokens_this_run = 0
    batch_number = 0

    try:
        print("Connecting to PostgreSQL...")

        connection = create_database_connection(register_pgvector=True)

        initial_status = get_embedding_status(connection)

        print_embedding_status(
            initial_status,
            title="INITIAL EMBEDDING STATUS",
        )

        initial_pending = initial_status[2]

        if initial_pending == 0:
            print()
            print("No valid chunks require embeddings.")
            print("Existing embeddings were left unchanged.")
            return

        print()
        print(f"Embedding model: {EMBEDDING_MODEL}")
        print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
        print(f"Batch size: {EMBEDDING_BATCH_SIZE}")
        print("Existing embeddings will be skipped.")

        while True:
            if (
                MAX_EMBEDDING_BATCHES is not None
                and batch_number >= MAX_EMBEDDING_BATCHES
            ):
                print()
                print(
                    f"Stopped after MAX_EMBEDDING_BATCHES={MAX_EMBEDDING_BATCHES}."
                )
                break

            batch_rows = load_next_batch(
                connection=connection,
                batch_size=EMBEDDING_BATCH_SIZE,
            )

            if not batch_rows:
                print()
                print("No more valid chunks are waiting.")
                break

            batch_number += 1

            chunk_ids = [
                chunk_id
                for chunk_id, _ in batch_rows
            ]

            print()
            print("=" * 80)
            print(f"Processing batch {batch_number}")
            print("=" * 80)
            print(f"Chunks in batch: {len(batch_rows)}")
            print(
                f"Chunk ID range: "
                f"{chunk_ids[0]} to {chunk_ids[-1]}"
            )
            print("Generating embeddings...")

            embedding_arrays, batch_tokens = (
                generate_batch_embeddings(
                    client=openai_client,
                    batch_rows=batch_rows,
                )
            )

            print(
                f"Generated {len(embedding_arrays)} embeddings."
            )
            print(f"Batch input tokens: {batch_tokens}")
            print("Saving batch to PostgreSQL...")

            save_batch_embeddings(
                connection=connection,
                batch_rows=batch_rows,
                embedding_arrays=embedding_arrays,
            )

            verify_saved_batch(
                connection=connection,
                chunk_ids=chunk_ids,
            )

            processed_this_run += len(batch_rows)
            total_tokens_this_run += batch_tokens

            current_status = get_embedding_status(connection)
            remaining_chunks = current_status[2]

            print("Batch saved and verified successfully.")
            print(
                f"Processed this run: {processed_this_run}"
            )
            print(
                f"Valid chunks still pending: {remaining_chunks}"
            )
            print(
                f"Total input tokens this run: "
                f"{total_tokens_this_run}"
            )

        final_status = get_embedding_status(connection)

        print_embedding_status(
            final_status,
            title="FINAL EMBEDDING STATUS",
        )

        print()
        print("=" * 80)
        print("EMBEDDING RUN SUMMARY")
        print("=" * 80)
        print(f"Batches completed: {batch_number}")
        print(
            f"Chunks processed this run: "
            f"{processed_this_run}"
        )
        print(
            f"Input tokens used this run: "
            f"{total_tokens_this_run}"
        )

        if final_status[2] == 0:
            print(
                "All valid chunks now have embeddings."
            )
        else:
            print(
                f"Valid chunks still pending: "
                f"{final_status[2]}"
            )

        print(
            "Previously existing embeddings were not overwritten."
        )

    except KeyboardInterrupt:
        if connection is not None:
            connection.rollback()

        print()
        print("=" * 80)
        print("The process was stopped by the user.")
        print(
            "Completed batches remain saved. "
            "Run the script again to continue."
        )
        print("=" * 80)

    except Exception:
        if connection is not None:
            connection.rollback()

        print()
        print("=" * 80)
        print("Embedding generation failed.")
        print(
            "The current unfinished database transaction "
            "was rolled back."
        )
        print(
            "Previously completed batches remain saved. "
            "Run the script again after resolving the error."
        )
        print("=" * 80)

        # Show the complete error traceback in PyCharm.
        raise

    finally:
        if connection is not None:
            connection.close()
            print("PostgreSQL connection closed.")


if __name__ == "__main__":
    main()