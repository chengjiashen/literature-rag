"""Create/update one synthetic PDF in literature_rag_dev; no API calls.

Run from the project root:
    python tools/prepare_incremental_demo.py
    python tools/prepare_incremental_demo.py --variant changed
    python tools/prepare_incremental_demo.py --variant broken
    python tools/prepare_incremental_demo.py --variant initial
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("incremental_parser", ROOT / "02_pdf_to_sections_chunks.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def make_pdf(revision):
    with pipeline.pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Introduction", fontsize=14)
        page.insert_text((72, 105), "This synthetic paper tests incremental literature ingestion.")
        page.insert_text((72, 145), "Methods", fontsize=14)
        page.insert_text((72, 178), "The study compares repeated processing of unchanged PDF files.")
        page = pdf.new_page()
        page.insert_text((72, 72), "Results", fontsize=14)
        page.insert_text((72, 105), f"Synthetic revision {revision}: repeated inputs should be skipped.")
        page.insert_text((72, 145), "Conclusion", fontsize=14)
        page.insert_text((72, 178), "Successful metadata and chunks should be stored together.")
        return pdf.tobytes()


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--variant", choices=("initial", "changed", "broken", "blank"), default="initial")
    args = args_parser.parse_args()
    from database import create_database_connection

    with create_database_connection() as conn:
        conn.autocommit = True
        pipeline.validate_target_database(conn, "literature_rag_dev")
        pipeline.acquire_pipeline_lock(conn)
        demo_dir = ROOT / ".incremental_demo"
        demo_dir.mkdir(exist_ok=True)
        original = demo_dir / "original.pdf"
        if not original.exists():
            original.write_bytes(make_pdf(1))
        target = demo_dir / "demo.pdf"
        if args.variant == "initial":
            content = original.read_bytes()
        elif args.variant == "changed":
            content = make_pdf(2)
        elif args.variant == "broken":
            content = b"This is deliberately not a valid PDF."
        else:
            with pipeline.pymupdf.open() as pdf:
                pdf.new_page()
                content = pdf.tobytes()
        # Only our synthetic test file is overwritten. No real PDF is touched.
        target.write_bytes(content)
        with conn.transaction():
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO public.papers (paper_key, title, publication_year)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (paper_key) DO NOTHING""",
                    ("__incremental_demo_v1__", "Synthetic incremental ingestion demo", 2026),
                )
                cursor.execute("SELECT paper_id FROM public.papers WHERE paper_key = %s",
                               ("__incremental_demo_v1__",))
                paper_id = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT document_id FROM public.documents WHERE paper_id = %s AND file_path = %s",
                    (paper_id, str(target)),
                )
                existing = cursor.fetchall()
                if len(existing) > 1:
                    raise RuntimeError("Duplicate demo registrations; inspect before proceeding.")
                if existing:
                    document_id = existing[0][0]
                else:
                    cursor.execute(
                        """INSERT INTO public.documents
                           (paper_id, document_type, file_path, file_format, notes)
                           VALUES (%s, 'demo', %s, 'pdf', 'Synthetic incremental parser test')
                           RETURNING document_id""", (paper_id, str(target)),
                    )
                    document_id = cursor.fetchone()[0]
        print(f"Database: literature_rag_dev; variant: {args.variant}")
        print(f"Demo document_id: {document_id}")
        print("Run this command twice to test processing then skipping:")
        print(f"python 02_pdf_to_sections_chunks.py --document-id {document_id}")


if __name__ == "__main__":
    main()
