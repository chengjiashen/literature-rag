-- Literature RAG: initial database schema (baseline 001)
-- Source: schema.sql exported from PostgreSQL 17.10 on 2026-11-13.
-- Target: PostgreSQL 17 with pgvector installed on the server.
-- Run ONCE in a NEW EMPTY database, e.g. literature_rag_dev.
-- Do not run against the existing literature_rag database.
-- Preserves all original tables, columns, sequences, constraints and delete rules.
-- No business data, credentials, new indexes or schema changes are included.
-- This is a baseline SQL script, not an installed migration framework.
-- Successfully executed in literature_rag_dev on 2026-11-13.
-- Plain SQL: compatible with pgAdmin Query Tool (no psql backslash commands).
-- Transaction-local settings revert at COMMIT; on any error, execute ROLLBACK.

BEGIN;

SET LOCAL statement_timeout = 0;
SET LOCAL lock_timeout = 0;
SET LOCAL idle_in_transaction_session_timeout = 0;
SET LOCAL transaction_timeout = 0;
SET LOCAL client_encoding = 'UTF8';
SET LOCAL standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', true);
SET LOCAL check_function_bodies = false;
SET LOCAL xmloption = content;
SET LOCAL client_min_messages = warning;
SET LOCAL row_security = off;

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';

SET LOCAL default_table_access_method = heap;

CREATE TABLE public.authors (
    author_id integer NOT NULL,
    full_name text NOT NULL,
    given_name text,
    family_name text,
    notes text
);

CREATE SEQUENCE public.authors_author_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.authors_author_id_seq OWNED BY public.authors.author_id;

CREATE TABLE public.chunks (
    chunk_id integer NOT NULL,
    paper_id integer,
    document_id integer,
    section_id integer,
    chunk_order integer,
    chunk_text text NOT NULL,
    page_start integer,
    page_end integer,
    notes text,
    embedding public.vector(1536)
);

CREATE SEQUENCE public.chunks_chunk_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.chunks_chunk_id_seq OWNED BY public.chunks.chunk_id;

CREATE TABLE public.documents (
    document_id integer NOT NULL,
    paper_id integer,
    document_type text,
    file_path text NOT NULL,
    file_format text,
    notes text
);

CREATE SEQUENCE public.documents_document_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.documents_document_id_seq OWNED BY public.documents.document_id;

CREATE TABLE public.paper_authors (
    paper_id integer NOT NULL,
    author_id integer NOT NULL,
    author_order integer
);

CREATE TABLE public.papers (
    paper_id integer NOT NULL,
    paper_key text,
    title text NOT NULL,
    publication_year integer,
    publication_type text,
    journal_or_book_title text,
    volume text,
    issue text,
    pages text,
    doi text,
    abstract text,
    language text DEFAULT 'en'::text,
    file_path text,
    notes text
);

CREATE SEQUENCE public.papers_paper_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.papers_paper_id_seq OWNED BY public.papers.paper_id;

CREATE TABLE public.sections (
    section_id integer NOT NULL,
    paper_id integer,
    document_id integer,
    section_title text,
    section_type text,
    section_order integer,
    page_start integer,
    page_end integer,
    notes text
);

CREATE SEQUENCE public.sections_section_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.sections_section_id_seq OWNED BY public.sections.section_id;

ALTER TABLE ONLY public.authors ALTER COLUMN author_id SET DEFAULT nextval('public.authors_author_id_seq'::regclass);

ALTER TABLE ONLY public.chunks ALTER COLUMN chunk_id SET DEFAULT nextval('public.chunks_chunk_id_seq'::regclass);

ALTER TABLE ONLY public.documents ALTER COLUMN document_id SET DEFAULT nextval('public.documents_document_id_seq'::regclass);

ALTER TABLE ONLY public.papers ALTER COLUMN paper_id SET DEFAULT nextval('public.papers_paper_id_seq'::regclass);

ALTER TABLE ONLY public.sections ALTER COLUMN section_id SET DEFAULT nextval('public.sections_section_id_seq'::regclass);

ALTER TABLE ONLY public.authors
    ADD CONSTRAINT authors_pkey PRIMARY KEY (author_id);

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_pkey PRIMARY KEY (chunk_id);

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (document_id);

ALTER TABLE ONLY public.paper_authors
    ADD CONSTRAINT paper_authors_pkey PRIMARY KEY (paper_id, author_id);

ALTER TABLE ONLY public.papers
    ADD CONSTRAINT papers_paper_key_key UNIQUE (paper_key);

ALTER TABLE ONLY public.papers
    ADD CONSTRAINT papers_pkey PRIMARY KEY (paper_id);

ALTER TABLE ONLY public.sections
    ADD CONSTRAINT sections_pkey PRIMARY KEY (section_id);

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(document_id) ON DELETE CASCADE;

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_paper_id_fkey FOREIGN KEY (paper_id) REFERENCES public.papers(paper_id) ON DELETE CASCADE;

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_section_id_fkey FOREIGN KEY (section_id) REFERENCES public.sections(section_id) ON DELETE SET NULL;

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_paper_id_fkey FOREIGN KEY (paper_id) REFERENCES public.papers(paper_id) ON DELETE CASCADE;

ALTER TABLE ONLY public.paper_authors
    ADD CONSTRAINT paper_authors_author_id_fkey FOREIGN KEY (author_id) REFERENCES public.authors(author_id) ON DELETE CASCADE;

ALTER TABLE ONLY public.paper_authors
    ADD CONSTRAINT paper_authors_paper_id_fkey FOREIGN KEY (paper_id) REFERENCES public.papers(paper_id) ON DELETE CASCADE;

ALTER TABLE ONLY public.sections
    ADD CONSTRAINT sections_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(document_id) ON DELETE CASCADE;

ALTER TABLE ONLY public.sections
    ADD CONSTRAINT sections_paper_id_fkey FOREIGN KEY (paper_id) REFERENCES public.papers(paper_id) ON DELETE CASCADE;

COMMIT;
