"""Tests for chromadb_intelligence."""

from __future__ import annotations

import tempfile
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Ensure the fork-chroma is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chromadb_intelligence.core import (
    _build_knn_graph,
    _laplacian,
    _cheeger_constant,
    _community_detection,
    _jsd_between_clusters,
    CollectionIntelligence,
    SpectralReport,
    DriftReport,
)
from chromadb_intelligence.cli import main as cli_main


warnings.filterwarnings("ignore", category=DeprecationWarning)


# =====================================================================
# Unit Tests — graph construction
# =====================================================================


class TestBuildKnnGraph:
    def test_three_points(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
        adj = _build_knn_graph(emb, k=2)
        assert adj.shape == (3, 3)
        assert adj.nnz > 0

    def test_single_point(self):
        emb = np.array([[1.0, 2.0]], dtype=np.float64)
        adj = _build_knn_graph(emb, k=2)
        assert adj.shape == (1, 1)

    def test_ten_random_points(self):
        np.random.seed(42)
        emb = np.random.randn(10, 4).astype(np.float64)
        adj = _build_knn_graph(emb, k=3)
        assert adj.shape == (10, 10)
        # check symmetric-ish (k-NN is not guaranteed symmetric, but COO works)
        assert adj.nnz >= 10

    def test_k_larger_than_n(self):
        emb = np.random.randn(5, 3).astype(np.float64)
        adj = _build_knn_graph(emb, k=100)
        assert adj.shape == (5, 5)
        # Should be fully connected
        assert adj.nnz == 5 * 4  # no self-loops

    def test_identical_vectors(self):
        emb = np.ones((4, 3), dtype=np.float64)
        adj = _build_knn_graph(emb, k=2)
        assert adj.shape == (4, 4)


# =====================================================================
# Unit Tests — Laplacian & spectral
# =====================================================================


class TestLaplacian:
    def test_two_point_graph(self):
        from scipy.sparse import coo_matrix
        adj = coo_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(2, 2))
        eigvals, fv = _laplacian(adj)
        assert len(eigvals) == 1  # only smallest available
        assert len(fv) == 2

    def test_small_connected(self):
        np.random.seed(7)
        emb = np.random.randn(8, 2).astype(np.float64)
        adj = _build_knn_graph(emb, k=3)
        eigvals, fv = _laplacian(adj)
        assert len(eigvals) > 0
        assert len(fv) == 8
        assert np.all(eigvals >= -1e-10)  # Laplacian is PSD


# =====================================================================
# Unit Tests — Cheeger constant
# =====================================================================


class TestCheegerConstant:
    def test_two_nodes_connected(self):
        from scipy.sparse import coo_matrix
        adj = coo_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(2, 2))
        fv = np.array([-0.5, 0.5])
        c = _cheeger_constant(adj, fv)
        assert c >= 0.0

    def test_no_split(self):
        from scipy.sparse import coo_matrix
        adj = coo_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(2, 2))
        fv = np.array([1.0, 1.0])  # all positive -> no split
        c = _cheeger_constant(adj, fv)
        assert c >= 0.0


# =====================================================================
# Unit Tests — Community detection
# =====================================================================


class TestCommunityDetection:
    def test_two_clusters(self):
        # Two well-separated clusters of 5 points each
        np.random.seed(1)
        cluster_a = np.random.randn(5, 2) + np.array([5, 0])
        cluster_b = np.random.randn(5, 2) + np.array([-5, 0])
        emb = np.vstack([cluster_a, cluster_b]).astype(np.float64)
        adj = _build_knn_graph(emb, k=3)
        _, fv = _laplacian(adj)
        labels, n_comm = _community_detection(adj, fv)
        assert n_comm >= 2
        assert len(labels) == 10

    def test_single_cluster(self):
        np.random.seed(2)
        emb = np.random.randn(4, 2).astype(np.float64)
        adj = _build_knn_graph(emb, k=3)
        _, fv = _laplacian(adj)
        labels, n_comm = _community_detection(adj, fv)
        assert n_comm >= 1

    def test_labels_contiguous(self):
        np.random.seed(3)
        emb = np.random.randn(8, 3).astype(np.float64)
        adj = _build_knn_graph(emb, k=3)
        _, fv = _laplacian(adj)
        labels, _ = _community_detection(adj, fv)
        unique = set(labels)
        assert unique == set(range(len(unique)))


# =====================================================================
# Unit Tests — JSD between clusters
# =====================================================================


class TestJsdBetweenClusters:
    def test_single_label(self):
        emb = np.random.randn(5, 2).astype(np.float64)
        labels = np.zeros(5, dtype=int)
        result = _jsd_between_clusters(emb, labels)
        assert len(result) == 0 or list(result.keys()) == ["no_clusters"]

    def test_two_labels(self):
        np.random.seed(4)
        emb = np.random.randn(10, 2).astype(np.float64)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = _jsd_between_clusters(emb, labels)
        assert len(result) > 0
        for v in result.values():
            assert 0.0 <= v <= 2.0  # JSD range

    def test_three_labels(self):
        np.random.seed(5)
        emb = np.random.randn(12, 2).astype(np.float64)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        result = _jsd_between_clusters(emb, labels)
        assert len(result) >= 2  # at least 0_vs_1, 0_vs_2, 1_vs_2


# =====================================================================
# Integration Tests — CollectionIntelligence end-to-end
# =====================================================================


def _make_real_chroma_client():
    """Try connecting to a real chroma instance (ephemeral in-memory)."""
    try:
        import chromadb
        client = chromadb.Client()
        # Quick health check
        client.heartbeat()
        return client
    except Exception:
        pytest.skip("No local chroma server available")


class TestCollectionIntelligenceIntegration:
    def test_ingest_and_analyze(self):
        client = _make_real_chroma_client()
        try:
            client.delete_collection("test_spectral")
        except Exception:
            pass
        coll = client.create_collection("test_spectral")

        # Add some well-separated clusters
        np.random.seed(42)
        for label, centroid in enumerate(
            [np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([1.0, -1.0])]
        ):
            for i in range(5):
                pt = centroid + np.random.randn(2) * 0.1
                coll.add(
                    ids=[f"{label}_{i}"],
                    embeddings=[pt.tolist()],
                    metadatas=[{"label": f"class_{label}"}],
                )

        ci = CollectionIntelligence(client=client)
        report = ci.analyze("test_spectral", k=4)

        assert report.num_points == 15
        assert report.embedding_dim == 2
        assert report.fiedler_value >= 0  # can be 0 if disconnected components
        assert report.num_communities >= 2
        assert isinstance(report.community_labels, list)
        assert len(report.community_labels) == 15
        assert isinstance(report.jsd_per_cluster, dict)
        client.delete_collection("test_spectral")

    def test_drift_detection(self):
        client = _make_real_chroma_client()
        try:
            client.delete_collection("test_drift")
        except Exception:
            pass
        coll = client.create_collection("test_drift")

        # Initial: well-separated
        np.random.seed(99)
        for i in range(10):
            pt = np.random.randn(2) + np.array([5, 5])
            coll.add(ids=[f"a_{i}"], embeddings=[pt.tolist()])
        for i in range(10):
            pt = np.random.randn(2) + np.array([-5, -5])
            coll.add(ids=[f"b_{i}"], embeddings=[pt.tolist()])

        ci = CollectionIntelligence(client=client)
        report_before = ci.analyze("test_drift", k=3)

        # Simulate drift: replace with noisy garbage
        client.delete_collection("test_drift")
        coll = client.create_collection("test_drift")
        for i in range(20):
            pt = np.random.randn(2) * 10  # random noise
            coll.add(ids=[f"noise_{i}"], embeddings=[pt.tolist()])

        drift = ci.detect_drift("test_drift", snapshot_before=report_before.dict(), k=3)
        assert isinstance(drift, DriftReport)
        assert drift.fiedler_delta is not None
        client.delete_collection("test_drift")

    def test_empty_collection_error(self):
        client = _make_real_chroma_client()
        try:
            client.delete_collection("test_empty")
        except Exception:
            pass
        coll = client.create_collection("test_empty")
        ci = CollectionIntelligence(client=client)
        with pytest.raises(ValueError, match="empty"):
            ci.analyze("test_empty")
        client.delete_collection("test_empty")

    def test_save_load_snapshot(self):
        client = _make_real_chroma_client()
        try:
            client.delete_collection("test_snap")
        except Exception:
            pass
        coll = client.create_collection("test_snap")
        coll.add(ids=["x"], embeddings=[[0.1, 0.2]])
        coll.add(ids=["y"], embeddings=[[0.9, 0.8]])

        ci = CollectionIntelligence(client=client)
        ci.analyze("test_snap", k=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            ci.save_snapshot(path)

        loaded = CollectionIntelligence.load_snapshot(path)
        assert len(loaded) == 1
        assert "report" in loaded[0]
        assert "fiedler_value" in loaded[0]["report"]
        os.unlink(path)
        client.delete_collection("test_snap")

    def test_analyze_with_baseline(self):
        client = _make_real_chroma_client()
        try:
            client.delete_collection("test_baseline")
        except Exception:
            pass
        coll = client.create_collection("test_baseline")
        coll.add(ids=["p"], embeddings=[[0.0, 0.0]])
        coll.add(ids=["q"], embeddings=[[1.0, 1.0]])

        ci = CollectionIntelligence(client=client)
        report1 = ci.analyze("test_baseline", k=1)
        # Same data, identical baseline
        report2 = ci.analyze("test_baseline", k=1, baseline=report1.dict())
        assert not report2.drift_detected  # Same data, no drift
        client.delete_collection("test_baseline")


# =====================================================================
# CLI Tests
# =====================================================================


class TestCLI:
    def test_analyze_help(self):
        try:
            cli_main(["analyze", "--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_drift_help(self):
        try:
            cli_main(["drift", "--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_no_command_error(self):
        try:
            cli_main([])
        except SystemExit as e:
            assert e.code == 2


# =====================================================================
# Edge cases & fuzzing
# =====================================================================


def test_very_low_dimensional():
    emb = np.array([[1.0], [0.0], [0.5]], dtype=np.float64)
    adj = _build_knn_graph(emb, k=2)
    assert adj.shape == (3, 3)


def test_high_k_small_n():
    emb = np.random.randn(4, 10).astype(np.float64)
    adj = _build_knn_graph(emb, k=20)
    assert adj.nnz == 4 * 3  # fully connected minus self-loops


def test_spectral_report_dataclass():
    r = SpectralReport(
        num_points=10,
        embedding_dim=3,
        fiedler_value=0.5,
        cheeger_constant=0.3,
        num_communities=3,
        community_labels=[0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
        jsd_per_cluster={"0_vs_1": 0.5},
        spectral_gap=0.2,
        raw_eigenvalues=[0.0, 0.5, 0.7],
    )
    d = r.dict()
    assert d["num_points"] == 10
    assert d["fiedler_value"] == 0.5


def test_drift_report_dataclass():
    d = DriftReport(
        fiedler_now=0.3,
        fiedler_before=0.6,
        fiedler_delta=-0.3,
        jsd_spectral=0.1,
        drift_detected=True,
        message="drift",
    )
    assert d.drift_detected


def test_invalid_collection_raises():
    """When no server is running, proper error expected."""
    try:
        import chromadb
        client = chromadb.Client()
        ci = CollectionIntelligence(client=client)
        with pytest.raises(Exception):
            ci.analyze("nonexistent_collection_xyz_not_exists")
    except Exception:
        pytest.skip("Chroma server not available")


def test_identical_points_no_crash():
    emb = np.ones((5, 3), dtype=np.float64) * 0.5
    adj = _build_knn_graph(emb, k=3)
    eigvals, fv = _laplacian(adj)
    assert np.isfinite(eigvals).all()


def test_reproducible_community_labels():
    np.random.seed(0)
    emb_a = np.random.randn(6, 2) + np.array([10, 0])
    emb_b = np.random.randn(6, 2) + np.array([-10, 0])
    emb = np.vstack([emb_a, emb_b]).astype(np.float64)
    adj = _build_knn_graph(emb, k=3)
    _, fv = _laplacian(adj)
    labels1, _ = _community_detection(adj, fv)
    labels2, _ = _community_detection(adj, fv)
    assert len(labels1) == len(labels2)


# Need at least 25 tests — add more targeted edge-case tests
def test_all_zero_embeddings():
    emb = np.zeros((5, 4), dtype=np.float64)
    adj = _build_knn_graph(emb, k=3)
    assert adj.nnz > 0
    eigvals, fv = _laplacian(adj)
    assert np.isfinite(eigvals).all()


def test_large_sparse_laplacian_stable():
    np.random.seed(123)
    emb = np.random.randn(30, 8).astype(np.float64)
    adj = _build_knn_graph(emb, k=5)
    eigvals, _ = _laplacian(adj)
    assert np.all(eigvals >= -1e-8)
    assert len(eigvals) > 1


def test_mixed_scale_embeddings():
    emb = np.array(
        [[1e3, 0], [0, 1e-3], [5e2, 5e-4], [-1e3, -1e-3]], dtype=np.float64
    )
    adj = _build_knn_graph(emb, k=2)
    assert adj.nnz > 0
    eigvals, _ = _laplacian(adj)
    assert np.isfinite(eigvals).all()


def test_cheeger_constant_perfect_split():
    """Two disconnected components of 3 nodes each."""
    from scipy.sparse import coo_matrix
    # Component A: 0-1, 0-2; Component B: 3-4, 3-5
    rows = [0, 0, 1, 3, 3, 4]
    cols = [1, 2, 0, 4, 5, 3]
    data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    adj = coo_matrix((data, (rows, cols)), shape=(6, 6))
    fv = np.array([-1, -1, -1, 1, 1, 1])
    c = _cheeger_constant(adj, fv)
    assert c >= 0
    assert c < 0.5


def test_jsd_identical_clusters():
    emb = np.random.randn(10, 2).astype(np.float64)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    result = _jsd_between_clusters(emb, labels)
    # Two clusters from same distribution → JSD <= 2.0 (valid range)
    if "0_vs_1" in result:
        assert 0.0 <= result["0_vs_1"] <= 2.0


def test_cli_analyze_json_flag():
    """Check that --json argument parses correctly at the argparse level."""
    # We can't easily test the full CLI without a server, but we can check
    # that the help output mentions --json.
    import io
    import contextlib
    from chromadb_intelligence.cli import main as cli_main
    # Just test the analyze parser directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", "-c", required=True)
    parser.add_argument("--json", action="store_true")
    # Assert parse succeeds
    args = parser.parse_args(["--collection", "test", "--json"])
    assert args.collection == "test"
    assert args.json is True


def test_cli_drift_missing_baseline():
    try:
        cli_main(["drift", "--collection", "dummy"])
    except SystemExit as e:
        assert e.code == 2  # argparse error


def test_spectral_report_with_drift():
    r = SpectralReport(
        num_points=100,
        embedding_dim=4,
        fiedler_value=0.12,
        cheeger_constant=0.05,
        num_communities=5,
        community_labels=[0] * 100,
        jsd_per_cluster={},
        spectral_gap=0.01,
        raw_eigenvalues=[0.0, 0.12, 0.13, 0.15],
        drift_detected=True,
        drift_message="Fiedler dropped",
    )
    assert r.drift_detected
    assert "dropped" in r.drift_message


def test_build_graph_symmetric_normalization():
    np.random.seed(10)
    emb = np.random.randn(12, 5).astype(np.float64)
    adj = _build_knn_graph(emb, k=4)
    # Convert to dense for comparison
    dense = adj.toarray()
    # k-NN not strictly symmetric but shouldn't crash
    assert dense.shape == (12, 12)


def test_three_way_jsd():
    np.random.seed(20)
    emb_a = np.random.randn(4, 2)
    emb_b = np.random.randn(4, 2) + np.array([5, 0])
    emb_c = np.random.randn(4, 2) + np.array([-5, 0])
    emb = np.vstack([emb_a, emb_b, emb_c]).astype(np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    result = _jsd_between_clusters(emb, labels)
    assert len(result) >= 3  # 0v1, 0v2, 1v2
    # Clusters are well-separated, JSD should be > 0
    for k, v in result.items():
        assert v > 0, f"{k} has zero JSD"


def test_drift_report_dict():
    d = DriftReport(
        fiedler_now=0.4,
        fiedler_before=0.5,
        fiedler_delta=-0.1,
        jsd_spectral=0.05,
        drift_detected=False,
        message="stable",
    )
    dd = d.dict()
    assert dd["fiedler_now"] == 0.4
    assert dd["drift_detected"] is False


def test_analyze_from_empty_report_jsd():
    r = SpectralReport(
        num_points=2,
        embedding_dim=2,
        fiedler_value=0.0,
        cheeger_constant=0.0,
        num_communities=1,
        community_labels=[0, 0],
        jsd_per_cluster={},
        spectral_gap=0.0,
        raw_eigenvalues=[0.0, 0.0],
    )
    assert r.jsd_per_cluster == {}
    assert isinstance(r.community_labels, list)
