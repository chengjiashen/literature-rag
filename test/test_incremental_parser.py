"""Offline regression tests. Real PDFs; mocked database transaction boundaries.

These do not replace the PostgreSQL checks in INCREMENTAL_TEST_GUIDE.md.
"""
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("parser_under_test", ROOT / "02_pdf_to_sections_chunks.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def pdf_bytes(text="Introduction\nSynthetic literature text.\nMethods\nA reproducible experiment."):
    with p.pymupdf.open() as pdf:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        return pdf.tobytes()


class IncrementalParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "paper.pdf"
        self.content = pdf_bytes()
        self.path.write_bytes(self.content)
        self.version = p.build_pipeline_version()
        self.document = {
            "document_id": 1, "paper_id": 7, "file_path": str(self.path),
            "parse_status": "completed",
            "parsed_file_sha256": hashlib.sha256(self.content).hexdigest(),
            "parse_pipeline_version": self.version,
            "has_sections": True, "has_chunks": True,
        }
        self.conn = MagicMock()
        self.conn.autocommit = True

    def test_unchanged_document_does_not_write_or_extract(self):
        with patch.object(p, "prepare_sections") as extract:
            self.assertEqual(p.process_one_document(self.conn, self.document), "skipped")
        extract.assert_not_called()
        self.conn.cursor.assert_not_called()
        self.conn.transaction.assert_not_called()

    def test_changed_file_is_processed(self):
        self.path.write_bytes(pdf_bytes("Results\nA different document."))
        with patch.object(p, "mark_processing") as started, patch.object(p, "replace_document_output") as save:
            self.assertEqual(p.process_one_document(self.conn, self.document), "processed")
        started.assert_called_once()
        self.assertNotEqual(save.call_args.args[3], self.document["parsed_file_sha256"])

    def test_chunk_settings_change_pipeline_version(self):
        with patch.object(p, "CHUNK_SIZE", p.CHUNK_SIZE + 100):
            changed = p.build_pipeline_version()
        self.assertNotEqual(self.version, changed)
        self.assertFalse(p.should_skip_document(self.document, self.document["parsed_file_sha256"], changed))

    def test_algorithm_revision_changes_pipeline_version(self):
        with patch.object(p, "PARSER_RULES_VERSION", "new-rules"):
            self.assertNotEqual(self.version, p.build_pipeline_version())

    def test_failed_or_interrupted_state_is_retried(self):
        for status in ("pending", "failed", "processing"):
            with self.subTest(status=status):
                doc = dict(self.document, parse_status=status)
                self.assertFalse(p.should_skip_document(doc, doc["parsed_file_sha256"], self.version))

    def test_missing_output_is_rebuilt(self):
        doc = dict(self.document, has_chunks=False)
        self.assertFalse(p.should_skip_document(doc, doc["parsed_file_sha256"], self.version))

    def test_force_rebuilds_tracked_document(self):
        self.assertFalse(p.should_skip_document(
            self.document, self.document["parsed_file_sha256"], self.version, force=True
        ))

    def test_legacy_chunks_are_blocked_even_with_force(self):
        doc = dict(self.document, parsed_file_sha256=None, parse_status="pending")
        self.assertEqual(p.process_one_document(self.conn, doc, force=True), "blocked")
        self.conn.cursor.assert_not_called()

    def test_explicit_legacy_rebuild(self):
        doc = dict(self.document, parsed_file_sha256=None, parse_status="pending")
        with patch.object(p, "mark_processing"), patch.object(p, "replace_document_output") as save:
            self.assertEqual(p.process_one_document(self.conn, doc, rebuild_untracked=True), "processed")
        save.assert_called_once()

    def test_blank_pdf_does_not_become_page_marker_chunks(self):
        with self.assertRaisesRegex(ValueError, "OCR"):
            p.prepare_sections(pdf_bytes(""))

    def test_real_pdf_extracts_sections_without_synthetic_markers(self):
        sections = p.prepare_sections(self.content)
        self.assertEqual([s["heading"] for s in sections], ["Introduction", "Methods"])
        self.assertTrue(all("===== Page" not in text for s in sections for text in s["chunks"]))

    def test_corrupt_pdf_records_failure_before_replacement(self):
        self.path.write_bytes(b"not a PDF")
        with patch.object(p, "mark_processing"), patch.object(p, "replace_document_output") as save, patch.object(p, "record_failure") as fail:
            self.assertEqual(p.process_one_document(self.conn, self.document), "failed")
        save.assert_not_called()
        fail.assert_called_once()

    def test_missing_file_records_failure(self):
        self.path.unlink()
        with patch.object(p, "record_failure") as fail, patch.object(p, "replace_document_output") as save:
            self.assertEqual(p.process_one_document(self.conn, self.document), "failed")
        fail.assert_called_once()
        save.assert_not_called()

    def test_hash_and_extraction_use_same_bytes_if_file_changes(self):
        seen = []
        original_prepare = p.prepare_sections
        def replacing_prepare(content):
            self.path.write_bytes(pdf_bytes("Results\nFile changed after it was read."))
            seen.append(content)
            return original_prepare(content)
        with patch.object(p, "mark_processing"), patch.object(p, "prepare_sections", side_effect=replacing_prepare), patch.object(p, "replace_document_output") as save:
            self.assertEqual(p.process_one_document(self.conn, self.document, force=True), "processed")
        self.assertEqual(save.call_args.args[3], hashlib.sha256(seen[0]).hexdigest())
        self.assertNotEqual(seen[0], self.path.read_bytes())

    def test_insert_failure_exits_transaction_before_failure_record(self):
        events = []
        @contextmanager
        def transaction():
            events.append("begin")
            try:
                yield
            except Exception:
                events.append("rollback")
                raise
            else:
                events.append("commit")
        self.conn.transaction.side_effect = transaction
        cursor = self.conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (7, str(self.path))
        with patch.object(p, "mark_processing"), patch.object(p, "delete_old_sections_and_chunks", side_effect=lambda *a: events.append("delete")), patch.object(p, "insert_section", return_value=10), patch.object(p, "insert_chunk", side_effect=RuntimeError("injected insert failure")), patch.object(p, "record_failure", side_effect=lambda *a: events.append("record_failure")):
            self.assertEqual(p.process_one_document(self.conn, self.document, force=True), "failed")
        self.assertEqual(events, ["begin", "delete", "rollback", "record_failure"])
        self.assertFalse(any("SET parse_status = 'completed'" in call.args[0] for call in cursor.execute.call_args_list))

    def test_success_metadata_is_written_inside_output_transaction(self):
        active = [False]
        @contextmanager
        def transaction():
            active[0] = True
            yield
            active[0] = False
        self.conn.transaction.side_effect = transaction
        cursor = self.conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (7, str(self.path))
        cursor.rowcount = 1
        completion_writes = []
        def execute(sql, params):
            if "SET parse_status = 'completed'" in sql:
                completion_writes.append(active[0])
        cursor.execute.side_effect = execute
        with patch.object(p, "delete_old_sections_and_chunks"), patch.object(p, "insert_section", return_value=10), patch.object(p, "insert_chunk"):
            count = p.replace_document_output(self.conn, self.document, [{"heading": "Methods", "chunks": ["abc"]}], self.document["parsed_file_sha256"], self.version)
        self.assertEqual(count, 1)
        self.assertEqual(completion_writes, [True])

    def test_failure_update_preserves_success_snapshot(self):
        cursor = self.conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        p.record_failure(self.conn, 1, "timestamp", ValueError("broken"))
        sql, params = cursor.execute.call_args.args
        for field in ("parsed_file_sha256", "parse_pipeline_version", "parsed_at"):
            self.assertNotIn(field, sql)
        self.assertIn("ValueError: broken", params)

    def test_database_guard_rejects_original_database(self):
        cursor = self.conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = ("literature_rag",)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            p.validate_target_database(self.conn, "literature_rag_dev")

    def test_missing_migration_fails_before_work(self):
        cursor = self.conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = ("literature_rag_dev",)
        cursor.fetchall.return_value = [("document_id",)]
        with self.assertRaisesRegex(RuntimeError, "migration 002"):
            p.validate_target_database(self.conn, "literature_rag_dev")

    def test_concurrent_run_is_rejected(self):
        cursor = self.conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (False,)
        with self.assertRaisesRegex(RuntimeError, "Another"):
            p.acquire_pipeline_lock(self.conn)


if __name__ == "__main__":
    unittest.main()
