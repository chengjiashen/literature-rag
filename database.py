import psycopg
from pgvector.psycopg import register_vector

from config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
)


# ============================================================
# Database configuration validation
# ============================================================

def validate_database_configuration():
    """
    Verify that all required database configuration values are available.

    Raises:
        RuntimeError:
            If one or more required database configuration values
            are missing.
    """
    required_variables = {
        "DB_HOST": DB_HOST,
        "DB_PORT": DB_PORT,
        "DB_NAME": DB_NAME,
        "DB_USER": DB_USER,
        "DB_PASSWORD": DB_PASSWORD,
    }

    missing_variables = [
        variable_name
        for variable_name, variable_value in required_variables.items()
        if not variable_value
    ]

    if missing_variables:
        raise RuntimeError(
            "Missing required database configuration: "
            + ", ".join(missing_variables)
        )


# ============================================================
# Database connection
# ============================================================

def create_database_connection(
    register_pgvector=False,
):
    """
    Create a PostgreSQL database connection.

    Args:
        register_pgvector:
            If True, register PostgreSQL vector type support for
            this connection.

    Returns:
        psycopg.Connection:
            An active PostgreSQL database connection.
    """
    validate_database_configuration()

    connection = psycopg.connect(
        host=DB_HOST,
        port=int(DB_PORT),
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    if register_pgvector:
        register_vector(connection)

    return connection