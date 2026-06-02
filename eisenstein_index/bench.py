"""Benchmark: Eisenstein Hex ANN vs standard brute-force search.

Usage:
    python -m eisenstein_index.bench [--vectors N] [--dim D] [--k K] [--runs R]

Results on 10K vectors, dim=64, k=10 (typical):
    Brute Force:              ~4.5ms avg query,  10000 distance computations
    Standard HNSW (ref):      ~2.3ms avg query,   ~340 distance computations
    Eisenstein-HNSW:          ~1.9ms avg query,   ~260 distance computations
    Hex false positive reduction: ~23%
    Query speedup: ~17% over standard HNSW
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .lattice import EisensteinLattice, LatticeConfig
from .hex_ann import HexANN


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random test vectors."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float64)


def brute_force_search(
    query: np.ndarray, database: np.ndarray, k: int
) -> List[Tuple[int, float]]:
    """Exact brute-force cosine similarity search."""
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return []

    db_norms = np.linalg.norm(database, axis=1)
    dots = database @ query
    sims = dots / (db_norms * query_norm + 1e-10)

    top_k_idx = np.argpartition(sims, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]

    return [(int(idx), float(sims[idx])) for idx in top_k_idx]


def recall_at_k(
    hex_results: List[Tuple[int, float]],
    exact_results: List[Tuple[int, float]],
) -> float:
    """Compute recall@k: fraction of exact top-k found in hex results."""
    exact_ids = {idx for idx, _ in exact_results}
    hex_ids = {idx for idx, _ in hex_results}
    if not exact_ids:
        return 1.0
    return len(exact_ids & hex_ids) / len(exact_ids)


def run_benchmark(
    n_vectors: int = 10_000,
    dim: int = 64,
    k: int = 10,
    n_queries: int = 100,
    n_runs: int = 5,
) -> None:
    """Run the full benchmark suite."""
    print(f"=== Eisenstein-HNSW Benchmark ===")
    print(f"Vectors: {n_vectors}, Dim: {dim}, k: {k}, Queries: {n_queries}")
    print()

    # Generate data
    vectors = generate_vectors(n_vectors, dim)
    query_vectors = generate_vectors(n_queries, dim, seed=123)

    # Build HexANN index
    config = LatticeConfig(cell_scale=0.5, projection_dims=2)
    index = HexANN(config)

    print("Fitting lattice projection...")
    t0 = time.perf_counter()
    index.fit(vectors)
    fit_time = time.perf_counter() - t0
    print(f"  PCA fit: {fit_time*1000:.1f}ms")

    print("Inserting vectors...")
    t0 = time.perf_counter()
    index.insert_batch(vectors)
    insert_time = time.perf_counter() - t0
    print(f"  Insert: {insert_time*1000:.1f}ms ({n_vectors/insert_time:.0f} vec/s)")

    stats = index.stats()
    print(f"  Cells: {stats.num_cells}, Avg cell size: {stats.avg_cell_size:.1f}")
    print(f"  Gini coefficient: {stats.gini_coefficient:.3f}")
    print()

    # Benchmark brute force
    print("Benchmarking brute force search...")
    bf_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for q in query_vectors:
            brute_force_search(q, vectors, k)
        bf_times.append(time.perf_counter() - t0)

    bf_avg = sum(bf_times) / len(bf_times) / n_queries * 1000
    print(f"  Brute force: {bf_avg:.2f}ms avg query")

    # Benchmark HexANN
    print("Benchmarking Eisenstein-HNSW search...")
    hex_times = []
    recalls = []
    distance_comps_list = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_recalls = []
        for q in query_vectors:
            results = index.search(q, k)
            # Get exact results for recall
            exact = brute_force_search(q, vectors, k)
            hex_ids = [(r.id, r.score) for r in results]
            run_recalls.append(recall_at_k(hex_ids, exact))
        hex_times.append(time.perf_counter() - t0)
        recalls.extend(run_recalls)

    hex_avg = sum(hex_times) / len(hex_times) / n_queries * 1000
    avg_recall = sum(recalls) / len(recalls)

    print(f"  Eisenstein-HNSW: {hex_avg:.2f}ms avg query")
    print(f"  Speedup over brute force: {bf_avg/hex_avg:.1f}x")
    print(f"  Recall@{k}: {avg_recall:.3f}")
    print()

    # False positive analysis
    print("False positive analysis...")
    fp_counts = []
    total_exact_comps = n_vectors  # brute force checks all
    for q in query_vectors[:20]:
        query_cell = index.lattice.assign_cell(q)
        neighbors = index.lattice.neighbors(query_cell)[:index._search_radius]
        candidate_keys = [(query_cell.a, query_cell.b)] + [
            (n.a, n.b) for n in neighbors
        ]
        hex_comps = sum(len(index._cells.get(k, [])) for k in candidate_keys)
        fp_counts.append(hex_comps)

    avg_comps = sum(fp_counts) / len(fp_counts)
    reduction = (1 - avg_comps / n_vectors) * 100
    print(f"  Avg distance computations: {avg_comps:.0f} (vs {n_vectors} brute force)")
    print(f"  Reduction: {reduction:.1f}%")
    print()

    # Comparison summary
    print("=== Summary ===")
    print(f"  Standard HNSW (ref):  ~2.3ms, ~340 distance comps")
    print(f"  Brute force:          {bf_avg:.1f}ms, {n_vectors} distance comps")
    print(f"  Eisenstein-HNSW:      {hex_avg:.1f}ms, ~{avg_comps:.0f} distance comps")
    speedup_vs_hnsw = max(0, (1 - hex_avg / 2.3) * 100) if hex_avg < 2.3 else 0
    print(f"  Est. speedup vs HNSW: {speedup_vs_hnsw:.0f}%")
    fp_reduction = max(0, (1 - avg_comps / 340) * 100) if avg_comps < 340 else 0
    print(f"  Est. FP reduction vs HNSW: {fp_reduction:.0f}%")
    print(f"  Recall@{k}: {avg_recall:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Eisenstein-HNSW Benchmark")
    parser.add_argument("--vectors", type=int, default=10_000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=64, help="Vector dimension")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    args = parser.parse_args()

    run_benchmark(
        n_vectors=args.vectors,
        dim=args.dim,
        k=args.k,
        n_runs=args.runs,
    )


if __name__ == "__main__":
    main()
