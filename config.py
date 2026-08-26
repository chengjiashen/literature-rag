import os

from dotenv import load_dotenv


# =========================================================
# Environment
# =========================================================

# Load variables from the project's .env file.
load_dotenv()


# =========================================================
# Database configuration
# =========================================================

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# =========================================================
# OpenAI configuration
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "gpt-5.6-terra",
)


# =========================================================
# Chunking configuration
# =========================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# =========================================================
# Embedding pipeline configuration
# =========================================================

EMBEDDING_DIMENSION = 1536

# Number of chunks sent to the embedding API in one request.
EMBEDDING_BATCH_SIZE = 100

# None means process all remaining chunks.
# Set this to 1 if you only want to test one batch first.
MAX_EMBEDDING_BATCHES = None


# =========================================================
# Retrieval configuration
# =========================================================

# Retrieve a broad candidate set before section-aware reranking.
CANDIDATE_K = 30

# Add one neighboring chunk before and after each selected anchor.
NEIGHBOR_RADIUS = 1

# Maximum number of merged evidence windows supplied to the LLM.
MAX_EVIDENCE_WINDOWS = 5

# Maximum total number of evidence characters supplied to the LLM.
MAX_CONTEXT_CHARACTERS = 24000


# =========================================================
# Answer generation configuration
# =========================================================

GENERATION_REASONING_EFFORT = "medium"
MAX_ANSWER_TOKENS = 2500


# =========================================================
# Debugging configuration
# =========================================================

SHOW_ANCHOR_DETAILS = True
SHOW_FULL_EVIDENCE_WINDOWS = False