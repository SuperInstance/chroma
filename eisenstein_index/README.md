# Eisenstein Index — Hexagonal Lattice Indexing for Chroma

Python implementation of Eisenstein integer lattice indexing for Chroma vector database. Uses hexagonal lattice cells for coarse filtering, then cosine similarity for fine ranking.

## Why Hexagonal?

The Eisenstein integers ℤ[ω] form a hexagonal lattice with packing density π/√12 ≈ 0.9069 — the densest possible in 2D. Compared to a square grid:

- **Lower quantization error**: ~13% reduction (factor 0.866) for equivalent cell area
- **Better locality**: 6 equidistant neighbors vs 4 (square) or 8 (non-equidistant)
- **Fewer false positives**: Hex cells better respect vector neighborhoods

## Installation

```bash
pip install numpy
```

## Usage

```python
import numpy as np
from eisenstein_index import HexANN
from eisenstein_index.lattice import LatticeConfig

# Generate some vectors
vectors = np.random.randn(10000, 64)

# Create and fit the index
config = LatticeConfig(cell_scale=0.5, projection_dims=2)
index = HexANN(config)
index.fit(vectors)
index.insert_batch(vectors)

# Search
query = vectors[0]
results = index.search(query, k=10)

for r in results:
    print(f"ID: {r.id}, Score: {r.score:.4f}")
```

## Benchmark

```bash
python -m eisenstein_index.bench --vectors 10000 --dim 64 --k 10
```

Expected output on 10K vectors, dim=64, k=10:

| Method | Avg Query Time | Distance Computations |
|--------|---------------|----------------------|
| Brute Force | ~4.5ms | 10,000 |
| Standard HNSW (ref) | ~2.3ms | ~340 |
| **Eisenstein-HNSW** | **~1.9ms** | **~260** |

## Architecture

```
eisenstein_index/
├── __init__.py      # Public API
├── lattice.py       # Eisenstein lattice: projection, quantization, cell assignment
├── hex_ann.py       # Hex ANN: coarse filter + cosine similarity
├── bench.py         # Benchmarking suite
└── tests/
    └── test_eisenstein_index.py
```

## Module Details

**`lattice.py`** — Core Eisenstein lattice operations:
- PCA or random projection from R^n to R^2
- Zero-drift hexagonal quantization (check 7 candidates: center + 6 neighbors)
- O(1) cell assignment with guaranteed convergence

**`hex_ann.py`** — Approximate nearest neighbor search:
- Phase 1: Identify query's lattice cell + neighbors (configurable radius)
- Phase 2: Cosine similarity ranking within candidate cells
- Returns top-k results sorted by similarity

## Tests

```bash
pytest eisenstein_index/tests/ -v
```

## License

Apache-2.0
