"""Unit tests for VectorStore."""

from __future__ import annotations

import pytest

from tests.conftest import make_rows, make_unit_vecs
from vecgrep.store import VectorStore


class TestSearch:
    def test_self_match_score_is_one(self, store):
        vecs = make_unit_vecs(2)
        rows = make_rows(2)
        store.add_chunks(rows, vecs)

        results = store.search(vecs[0], top_k=1)
        assert len(results) == 1
        assert abs(results[0]["score"] - 1.0) < 1e-5
        assert results[0]["content"] == "content_0"

    def test_empty_store_returns_empty(self, store):
        vec = make_unit_vecs(1)[0]
        assert store.search(vec) == []

    def test_top_k_limits_results(self, store):
        vecs = make_unit_vecs(5)
        store.add_chunks(make_rows(5), vecs)
        assert len(store.search(vecs[0], top_k=3)) == 3

    def test_results_have_expected_keys(self, store):
        vecs = make_unit_vecs(1)
        store.add_chunks(make_rows(1), vecs)
        result = store.search(vecs[0], top_k=1)[0]
        assert {"file_path", "start_line", "end_line", "content", "score"} <= result.keys()





class TestDeleteFileChunks:
    def test_removes_chunks(self, store):
        vecs = make_unit_vecs(3)
        store.add_chunks(make_rows(3), vecs)
        assert store.status()["total_chunks"] == 3

        store.delete_file_chunks("a.py")
        assert store.status()["total_chunks"] == 0

    def test_delete_nonexistent_is_noop(self, store):
        store.delete_file_chunks("nonexistent.py")
        assert store.status()["total_chunks"] == 0


class TestReplaceFileChunks:
    def test_only_new_chunks_remain(self, store):
        vecs_old = make_unit_vecs(2)
        store.add_chunks(make_rows(2), vecs_old)

        new_vecs = make_unit_vecs(1, seed=99)
        new_rows = [
            {
                "file_path": "a.py",
                "start_line": 10,
                "end_line": 20,
                "content": "new_content",
                "file_hash": "newhash",
                "chunk_hash": "newchunkhash",
            }
        ]
        store.replace_file_chunks("a.py", new_rows, new_vecs)

        assert store.status()["total_chunks"] == 1
        results = store.search(new_vecs[0], top_k=5)
        assert len(results) == 1
        assert results[0]["content"] == "new_content"

    def test_raises_on_length_mismatch(self, store):
        vecs = make_unit_vecs(2)
        with pytest.raises(ValueError, match="mismatch"):
            store.replace_file_chunks("a.py", make_rows(1), vecs)


class TestContextManager:
    def test_enter_returns_self(self, tmp_path):
        with VectorStore(tmp_path / "cm") as store:
            assert isinstance(store, VectorStore)

    def test_exit_does_not_interfere(self, tmp_path):
        with VectorStore(tmp_path / "cm") as store:
            pass
        # LanceDB doesn't throw if we access it outside the block
        assert store.status()["total_chunks"] == 0


class TestStatus:
    def test_empty_store(self, store):
        s = store.status()
        assert s["total_chunks"] == 0
        assert s["total_files"] == 0
        assert s["last_indexed"] == "never"

    def test_reflects_added_chunks(self, store):
        vecs = make_unit_vecs(3)
        store.add_chunks(make_rows(3), vecs)
        s = store.status()
        assert s["total_chunks"] == 3
        assert s["total_files"] == 1

    def test_touch_last_indexed(self, store):
        store.touch_last_indexed()
        assert store.status()["last_indexed"] != "never"


class TestGetFileHashes:
    def test_empty(self, store):
        assert store.get_file_hashes() == {}

    def test_returns_correct_mapping(self, store):
        vecs = make_unit_vecs(2)
        rows = make_rows(2, file_path="src/foo.py", file_hash="abc123")
        store.add_chunks(rows, vecs)
        hashes = store.get_file_hashes()
        assert hashes == {"src/foo.py": "abc123"}

    def test_multiple_files(self, store):
        vecs = make_unit_vecs(2)
        rows_a = make_rows(1, file_path="a.py", file_hash="hash_a")
        rows_b = make_rows(1, file_path="b.py", file_hash="hash_b")
        store.add_chunks(rows_a, vecs[:1])
        store.add_chunks(rows_b, vecs[1:])
        hashes = store.get_file_hashes()
        assert hashes == {"a.py": "hash_a", "b.py": "hash_b"}


class TestAddChunksValidation:
    def test_length_mismatch_raises(self, store):
        vecs = make_unit_vecs(3)
        rows = make_rows(2)
        with pytest.raises(ValueError, match="mismatch"):
            store.add_chunks(rows, vecs)


class TestBuildIndex:
    def test_returns_false_when_too_few_chunks(self, store):
        vecs = make_unit_vecs(5)
        store.add_chunks(make_rows(5), vecs)
        assert store.build_index() is False

    def test_returns_true_and_builds_when_enough_chunks(self, tmp_path):
        from unittest.mock import MagicMock, patch
        s = VectorStore(tmp_path / "idx")
        vecs = make_unit_vecs(5)
        s.add_chunks(make_rows(5), vecs)
        mock_create = MagicMock()
        with patch.object(s._table, "create_index", mock_create), \
             patch.object(s._table, "count_rows", return_value=300):
            result = s.build_index()
        assert result is True
        mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# Dynamic dims
# ---------------------------------------------------------------------------


class TestDynamicDims:
    def test_default_dims_is_384(self, tmp_path):
        s = VectorStore(tmp_path / "d384")
        assert s._dims == 384

    def test_custom_dims_stored(self, tmp_path):
        s = VectorStore(tmp_path / "d1536", dims=1536)
        assert s._dims == 1536

    def test_stored_dims_used_on_reopen(self, tmp_path):
        idx = tmp_path / "reopen"
        s1 = VectorStore(idx, dims=1024)
        s1._set_meta("dims", "1024")
        # Reopen with a different dims param — stored should win
        s2 = VectorStore(idx, dims=384)
        assert s2._dims == 1024

    def test_chunks_added_with_custom_dims(self, tmp_path):
        idx = tmp_path / "custom"
        s = VectorStore(idx, dims=8)
        vecs = make_unit_vecs(2, dim=8)
        rows = make_rows(2)
        s.add_chunks(rows, vecs)
        assert s.status()["total_chunks"] == 2

    def test_drop_and_recreate_chunks(self, tmp_path):
        idx = tmp_path / "drop"
        s = VectorStore(idx, dims=384)
        vecs = make_unit_vecs(3)
        s.add_chunks(make_rows(3), vecs)
        assert s.status()["total_chunks"] == 3

        s.drop_and_recreate_chunks(dims=1024)
        assert s.status()["total_chunks"] == 0
        assert s._dims == 1024


# ---------------------------------------------------------------------------
# Provider meta
# ---------------------------------------------------------------------------


class TestProviderMeta:
    def test_default_provider_meta(self, tmp_path):
        s = VectorStore(tmp_path / "meta_default")
        meta = s.get_provider_meta()
        assert meta["provider"] == "local"

    def test_set_and_get_provider_meta(self, store):
        store.set_provider_meta("openai", "text-embedding-3-small", 1536)
        meta = store.get_provider_meta()
        assert meta["provider"] == "openai"
        assert meta["model"] == "text-embedding-3-small"
        assert meta["dims"] == 1536

    def test_provider_meta_persists_across_reopen(self, tmp_path):
        idx = tmp_path / "persist"
        s1 = VectorStore(idx)
        s1.set_provider_meta("voyage", "voyage-code-2", 1024)

        s2 = VectorStore(idx)
        meta = s2.get_provider_meta()
        assert meta["provider"] == "voyage"
        assert meta["model"] == "voyage-code-2"
        assert meta["dims"] == 1024

    def test_status_includes_provider_fields(self, store):
        store.set_provider_meta("local", "some-model", 384)
        s = store.status()
        assert "provider" in s
        assert "model" in s
        assert "dims" in s


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------


class TestMetaHelpers:
    def test_get_missing_key_returns_none(self, store):
        assert store._get_meta("nonexistent_key") is None

    def test_set_and_get_meta(self, store):
        store._set_meta("mykey", "myvalue")
        assert store._get_meta("mykey") == "myvalue"

    def test_set_meta_overwrites(self, store):
        store._set_meta("k", "v1")
        store._set_meta("k", "v2")
        assert store._get_meta("k") == "v2"
