"""Core spectral analysis engine for chroma collections."""

from __future__ import annotations

import time
import json
import warnings
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csgraph, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform, jensenshannon

try:
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.api.types import Include
except ImportError:
    chromadb = None  # type: ignore[assignment]

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Dataclasses for reports
# ---------------------------------------------------------------------------


@dataclass
class SpectralReport:
    """Results from spectral graph analysis of a collection."""

    num_points: int
    embedding_dim: int
    fiedler_value: float
    cheeger_constant: float
    num_communities: int
    community_labels: list[int]
    jsd_per_cluster: dict[str, float]
    spectral_gap: float
    raw_eigenvalues: list[float]
    drift_detected: bool = False
    drift_message: str = ""

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    """Cross-temporal drift analysis result."""

    fiedler_now: float
    fiedler_before: float
    fiedler_delta: float
    jsd_spectral: float
    drift_detected: bool
    message: str

    def dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Distance-based graph construction
# ---------------------------------------------------------------------------


def _build_knn_graph(embeddings: NDArray, k: int = 15) -> coo_matrix:
    """Build a k-NN similarity graph from an embedding array.

    Parameters
    ----------
    embeddings : NDArray of shape (n, d)
    k : int, optional
        Number of nearest neighbours per node (default 15).

    Returns
    -------
    coo_matrix
        Sparse adjacency matrix.
    """
    n = embeddings.shape[0]
    if n < 3:
        # fallback: fully connected for tiny sets
        dists = squareform(pdist(embeddings, metric="cosine"))
        adj = np.exp(-dists)
        np.fill_diagonal(adj, 0.0)
        return coo_matrix(adj)

    # Compute cosine distance matrix
    dists = squareform(pdist(embeddings, metric="cosine"))
    # Handle NaN (all-zero vectors produce NaN cosine distance)
    dists = np.nan_to_num(dists, nan=1.0)
    # Convert to similarity
    sim = np.exp(-dists)
    np.fill_diagonal(sim, 0.0)

    # Keep only top-k neighbours per row
    k_actual = min(k, n - 1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n):
        # Get k-th largest value
        row_sim = sim[i]
        threshold = np.partition(row_sim, -k_actual)[-k_actual]
        for j in range(n):
            if i == j:
                continue
            if row_sim[j] >= threshold:
                rows.append(i)
                cols.append(j)
                data.append(row_sim[j])

    return coo_matrix((data, (rows, cols)), shape=(n, n))


def _laplacian(adj: coo_matrix) -> tuple[NDArray, NDArray]:
    """Compute the (normalised) graph Laplacian and return eigenvalues + Fiedler.

    Returns
    -------
    eigenvalues : NDArray
        Sorted eigenvalues (ascending).
    fiedler : NDArray
        Fiedler eigenvector (2nd smallest eigenvalue).
    """
    n = adj.shape[0]
    if n < 2:
        return np.array([0.0]), np.array([0.0])

    lap = csgraph.laplacian(adj, normed=True)
    try:
        eigenvalues, eigenvectors = eigsh(
            lap, k=min(n - 1, 20), which="SM", tol=1e-6, maxiter=n * 100
        )
    except Exception:
        # fallback: dense solve
        eigvals, eigvecs = np.linalg.eigh(lap.toarray())
        eigenvalues = eigvals[: min(n - 1, 20)]
        eigenvectors = eigvecs[:, : min(n - 1, 20)]

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Clamp tiny negative eigenvalues to zero (numeric noise)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # Fiedler = second smallest eigenvalue
    fiedler = eigenvalues[1] if len(eigenvalues) > 1 else eigenvalues[0]
    fiedler_vec = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else eigenvectors[:, 0]

    return eigenvalues, fiedler_vec


def _cheeger_constant(
    adj: coo_matrix, fiedler_vec: NDArray
) -> float:
    """Approximate Cheeger constant via Fiedler vector sign cut."""
    # Convert to CSR for efficient row/col slicing
    adj_csr = adj.tocsr()
    n = adj_csr.shape[0]

    # Sign-based partition
    partition = fiedler_vec >= 0
    s = set(np.where(partition)[0].tolist())
    s_bar = set(np.where(~partition)[0].tolist())
    if not s or not s_bar:
        return 0.0

    # Count edges crossing the cut
    cut_edges = 0
    for i in s:
        row_start = adj_csr.indptr[i]
        row_end = adj_csr.indptr[i + 1]
        neighbours = adj_csr.indices[row_start:row_end]
        cut_edges += sum(1 for nb in neighbours if nb in s_bar)

    vol_s = max(adj_csr[i, :].sum(), 1e-10) if partition.any() else 1e-10
    vol_s_bar = max(adj_csr[list(s_bar), :].sum(), 1e-10) if (~partition).any() else 1e-10

    return cut_edges / min(vol_s, vol_s_bar)


def _community_detection(
    adj: coo_matrix, fiedler_vec: NDArray
) -> tuple[np.ndarray, int]:
    """Recursive spectral bisection for community detection.

    Converts to CSR for efficient submatrix slicing.

    Returns
    -------
    labels : NDArray of shape (n,)
        Integer community label for each node.
    num_communities : int
    """
    adj_csr = adj.tocsr()
    n = adj_csr.shape[0]
    labels = np.zeros(n, dtype=int)

    def _bisect(sub_nodes: list[int], label_offset: int) -> None:
        if len(sub_nodes) < 2:
            return
        # Build sub-adjacency for these nodes
        idx_map = {orig: i for i, orig in enumerate(sub_nodes)}
        m = len(sub_nodes)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for local_i, global_i in enumerate(sub_nodes):
            row_start = adj_csr.indptr[global_i]
            row_end = adj_csr.indptr[global_i + 1]
            for ptr in range(row_start, row_end):
                global_j = adj_csr.indices[ptr]
                if global_j in idx_map:
                    rows.append(local_i)
                    cols.append(idx_map[global_j])
                    data.append(adj_csr.data[ptr])
        if not rows:
            return
        from scipy.sparse import coo_matrix
        sub_adj = coo_matrix((data, (rows, cols)), shape=(m, m))
        try:
            eigenvals, fv = _laplacian(sub_adj)
        except Exception:
            return

        partition = fv >= 0
        left = [sub_nodes[i] for i, p in enumerate(partition) if p]
        right = [sub_nodes[i] for i, p in enumerate(partition) if not p]
        if not left or not right:
            return

        # Assign labels
        new_label = label_offset + 1
        for node in left:
            labels[node] = new_label
        # Recurse
        _bisect(left, new_label * 100)
        _bisect(right, new_label * 100 + 1)

    _bisect(list(range(n)), 0)
    unique = np.unique(labels)
    # Compact labels
    mapping = {old: i for i, old in enumerate(sorted(unique))}
    compact = np.array([mapping[l] for l in labels])
    return compact, len(unique)


def _jsd_between_clusters(
    embeddings: NDArray, labels: np.ndarray
) -> dict[str, float]:
    """Jensen-Shannon divergence between distance distributions of clusters."""
    n = embeddings.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"no_clusters": 1.0}

    dists = squareform(pdist(embeddings, metric="cosine"))
    eps = 1e-10
    results: dict[str, float] = {}
    for i, lab_i in enumerate(unique_labels):
        mask_i = labels == lab_i
        for lab_j in unique_labels[i + 1:]:
            mask_j = labels == lab_j
            intra_i = dists[np.ix_(mask_i, mask_i)][np.triu_indices(mask_i.sum(), k=1)]
            intra_j = dists[np.ix_(mask_j, mask_j)][np.triu_indices(mask_j.sum(), k=1)]

            if len(intra_i) == 0 or len(intra_j) == 0:
                continue

            # Build probability distributions via histogram
            all_d = np.concatenate([intra_i, intra_j])
            bins = np.linspace(0, 1, 50)
            p_i, _ = np.histogram(intra_i, bins=bins, density=True)
            p_j, _ = np.histogram(intra_j, bins=bins, density=True)
            p_i = p_i + eps
            p_j = p_j + eps
            p_i /= p_i.sum()
            p_j /= p_j.sum()

            jsd = jensenshannon(p_i, p_j, base=2)
            results[f"{int(lab_i)}_vs_{int(lab_j)}"] = round(float(jsd), 6)

    return results


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------


class CollectionIntelligence:
    """Spectral graph intelligence for a Chroma collection.

    Parameters
    ----------
    client : chromadb.ClientAPI | None
        Chroma client. If None, uses ``chromadb.Client()``.
    """

    def __init__(self, client: Optional[Any] = None):
        if chromadb is None:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        self._client: ClientAPI = client or chromadb.Client()
        self._history: list[dict] = []

    def _fetch_embeddings(
        self,
        collection_name: str,
        tenant: str = chromadb.config.DEFAULT_TENANT if chromadb else "default_tenant",
        database: str = chromadb.config.DEFAULT_DATABASE if chromadb else "default_database",
    ) -> NDArray:
        """Fetch all embeddings from a collection.

        Works with both local (ephemeral/persistent) and remote (HTTP) clients.
        Local client.get_collection() doesn't accept tenant/database kwargs,
        so we detect and fall back gracefully.
        """
        import inspect
        get_collection_params = inspect.signature(
            self._client.get_collection
        ).parameters
        if "tenant" in get_collection_params:
            coll = self._client.get_collection(
                name=collection_name,
                tenant=tenant,
                database=database,
            )
        else:
            coll = self._client.get_collection(name=collection_name)
        n = coll.count()
        if n == 0:
            raise ValueError(f"Collection '{collection_name}' is empty.")

        batch_size = 1000
        all_embeddings: list[NDArray] = []
        for offset in range(0, n, batch_size):
            result = coll.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings"] if Include else ["embeddings"],
            )
            emb = result["embeddings"]
            if emb is None or len(emb) == 0:
                continue
            all_embeddings.append(np.asarray(emb, dtype=np.float64))

        if not all_embeddings:
            raise ValueError(f"No embeddings retrieved from '{collection_name}'.")

        embeddings = np.concatenate(all_embeddings, axis=0)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        return embeddings

    def analyze(
        self,
        collection_name: str,
        k: int = 15,
        tenant: str = "default_tenant",
        database: str = "default_database",
        baseline: Optional[dict] = None,
    ) -> SpectralReport:
        """Run full spectral intelligence on a collection.

        Parameters
        ----------
        collection_name : str
        k : int
            k-NN graph parameter.
        baseline : dict | None
            Previous ``SpectralReport.dict()`` for drift detection.

        Returns
        -------
        SpectralReport
        """
        embeddings = self._fetch_embeddings(
            collection_name, tenant=tenant, database=database
        )
        n, d = embeddings.shape

        # Build graph
        adj = _build_knn_graph(embeddings, k=k)

        # Spectral analysis
        eigenvalues, fiedler_vec = _laplacian(adj)
        fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else float(eigenvalues[0])
        spectral_gap = float(eigenvalues[2] - eigenvalues[1]) if len(eigenvalues) > 2 else 0.0

        cheeger = float(_cheeger_constant(adj, fiedler_vec))

        # Community detection
        community_labels, num_communities = _community_detection(adj, fiedler_vec)

        # JSD between clusters
        jsd = _jsd_between_clusters(embeddings, community_labels)

        # Drift detection
        drift_detected = False
        drift_message = ""
        if baseline is not None:
            fv_before = baseline.get("fiedler_value", fiedler_value)
            delta = fiedler_value - fv_before
            if delta < -0.05:
                drift_detected = True
                drift_message = (
                    f"Embedding drift detected: Fiedler value dropped from "
                    f"{fv_before:.4f} to {fiedler_value:.4f} (Δ={delta:.4f}). "
                    f"New embeddings produce weaker clustering structure."
                )
            else:
                drift_message = (
                    f"Fiedler value stable: {fv_before:.4f} → {fiedler_value:.4f}"
                )

        report = SpectralReport(
            num_points=n,
            embedding_dim=d,
            fiedler_value=fiedler_value,
            cheeger_constant=cheeger,
            num_communities=num_communities,
            community_labels=community_labels.tolist(),
            jsd_per_cluster=jsd,
            spectral_gap=spectral_gap,
            raw_eigenvalues=[round(float(v), 8) for v in eigenvalues],
            drift_detected=drift_detected,
            drift_message=drift_message,
        )

        # Save snapshot for future drift detection
        self._history.append({
            "timestamp": time.time(),
            "collection": collection_name,
            "report": report.dict(),
        })

        return report

    def detect_drift(
        self,
        collection_name: str,
        snapshot_before: dict,
        k: int = 15,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ) -> DriftReport:
        """Compare current spectral signature against a historical snapshot.

        Parameters
        ----------
        snapshot_before : dict
            Previous ``SpectralReport.dict()`` or one with at least
            ``fiedler_value``, ``raw_eigenvalues``.
        """
        current = self.analyze(
            collection_name, k=k, tenant=tenant, database=database
        )

        fv_before = snapshot_before.get("fiedler_value", current.fiedler_value)
        fv_delta = current.fiedler_value - fv_before

        # JSD between eigenvalue spectra
        eig_before = np.array(
            snapshot_before.get("raw_eigenvalues", current.raw_eigenvalues),
            dtype=np.float64,
        )
        eig_now = np.array(current.raw_eigenvalues, dtype=np.float64)

        # Pad to same length
        max_len = max(len(eig_before), len(eig_now))
        if len(eig_before) < max_len:
            eig_before = np.pad(eig_before, (0, max_len - len(eig_before)))
        if len(eig_now) < max_len:
            eig_now = np.pad(eig_now, (0, max_len - len(eig_now)))

        eps = 1e-10
        p_b = np.abs(eig_before) + eps
        p_n = np.abs(eig_now) + eps
        p_b /= p_b.sum()
        p_n /= p_n.sum()
        jsd_spectral = float(jensenshannon(p_b, p_n, base=2))

        drift_detected = fv_delta < -0.05
        if drift_detected:
            msg = (
                f"Embedding drift detected: Fiedler value dropped from "
                f"{fv_before:.4f} to {current.fiedler_value:.4f} (Δ={fv_delta:.4f}). "
                f"Spectral JSD: {jsd_spectral:.4f}."
            )
        else:
            msg = (
                f"Fiedler value stable: {fv_before:.4f} → {current.fiedler_value:.4f}. "
                f"Spectral JSD: {jsd_spectral:.4f}."
            )

        return DriftReport(
            fiedler_now=current.fiedler_value,
            fiedler_before=fv_before,
            fiedler_delta=fv_delta,
            jsd_spectral=jsd_spectral,
            drift_detected=drift_detected,
            message=msg,
        )

    def save_snapshot(self, path: str) -> None:
        """Save analysis history to disk."""
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)

    @classmethod
    def load_snapshot(cls, path: str) -> list[dict]:
        """Load a previously saved analysis snapshot."""
        with open(path) as f:
            return json.load(f)

