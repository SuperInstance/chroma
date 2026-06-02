# Integration Guide — chromadb_intelligence

This document describes how `chromadb_intelligence` integrates with Chroma's architecture, how to set it up for production use, and how to interpret results.

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│  CollectionIntelligence                         │
│  ┌───────────────────────────────────────────┐  │
│  │ 1. Fetch embeddings from Chroma           │  │
│  │    (collection.get(include=["embeddings"]))│  │
│  └────────────┬──────────────────────────────┘  │
│               ▼                                  │
│  ┌───────────────────────────────────────────┐  │
│  │ 2. Build k-NN similarity graph            │  │
│    │    (cosine distance → exponential kernel)│  │
│  └────────────┬──────────────────────────────┘  │
│               ▼                                  │
│  ┌───────────────────────────────────────────┐  │
│  │ 3. Spectral analysis                      │  │
│  │    • Graph Laplacian (normalised)         │  │
│  │    • Eigenvalue decomposition (eigsh)     │  │
│  │    • Fiedler vector + value               │  │
│  │    • Cheeger constant                     │  │
│  └────────────┬──────────────────────────────┘  │
│               ▼                                  │
│  ┌───────────────────────────────────────────┐  │
│  │ 4. Community detection                    │  │
│  │    • Recursive spectral bisection         │  │
│  │    • Label assignment + compaction        │  │
│  └────────────┬──────────────────────────────┘  │
│               ▼                                  │
│  ┌───────────────────────────────────────────┐  │
│  │ 5. Cross-cluster JSD analysis             │  │
│  │ 6. Drift detection vs baseline            │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `chromadb` | >= 1.5 | Vector database client |
| `numpy` | >= 1.24 | Array operations |
| `scipy` | >= 1.10 | Sparse linear algebra (eigsh, csgraph) |
| `scikit-learn` | ≥ 1.2 (optional) | Distance helpers |

## Connecting to Chroma

### Local ephemeral
```python
import chromadb
client = chromadb.Client()  # in-memory
```

### Persistent
```python
client = chromadb.PersistentClient(path="/data/chroma")
```

### Remote
```python
client = chromadb.HttpClient(host="localhost", port=8000)
```

## Performance Considerations

| Collection size | k parameter | Memory (~) | Time (~) |
|----------------|-------------|------------|----------|
| 1,000 points × 768d | 15 | 120 MB | 2 s |
| 10,000 points × 768d | 15 | 1.2 GB | 30 s |
| 100,000 points × 768d | 15 | 12 GB | 5+ min |

- k-NN graph construction is O(n²) via pairwise distances. For >50K points, consider:
  - Sampling (e.g., 10K representative points)
  - Approximate methods (ball tree, FAISS)
- eigsh on the Laplacian scales well for sparse graphs (O(n·k²))

## Reading the Report

### Fiedler Value
- **0.0–0.3**: Weak cluster separation. Embeddings may be poor.
- **0.3–0.7**: Moderate separation.
- **0.7–1.0**: Strong separation. Check if clusters are real or artefacts.

### Cheeger Constant
- Lower values = better partition quality.
- Value near 1 = all edges cross the cut (bad separation).

### Community Count vs Label Count
If your `num_communities ≫ metadata labels`, your embedding model creates
more fine-grained structure than your labels capture. This could mean:
- Your labels are too coarse
- Your embedding model is over-discriminating
- There's hidden structure in the data you haven't labelled

If `num_communities ≪ metadata labels`, your embedding model collapses
distinct categories together. This is usually a sign to switch models.

### Drift Detection
The Fiedler value is **the single most informative signal** for embedding drift:
- **Dropping Fiedler** → new embeddings don't separate as well → quality degradation
- **Rising Fiedler** → new embeddings create sharper clusters (usually good)

## Common Workflows

### Weekly Quality Check (Cron)
```bash
0 9 * * 1 chroma-intel analyze --collection prod_data --json >> /var/log/chroma-intel.log
# Alert if Fiedler < 0.2
```

### A/B Embedding Model Comparison
```python
report_a = ci.analyze("model_a_v1")
report_b = ci.analyze("model_b_v2")
print(f"Model A Fiedler: {report_a.fiedler_value:.4f}")
print(f"Model B Fiedler: {report_b.fiedler_value:.4f}")
```

### RAG Pipeline Health
```python
# Run after each re-index
drift = ci.detect_drift("rag_knowledge_base", snapshot_before=baseline)
if drift.drift_detected:
    trigger_review()
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ValueError: Collection is empty.` | No data in collection | Populate before analysis |
| MemoryError | Collection too large | Reduce k, sample points |
| `eigsh did not converge` | Graph disconnected | Increase k or use dense solver fallback |
| All communities = 1 | Too few points for k | Reduce k or add more data |
| JSD = empty dict | Only 1 community found | Check k and graph connectivity |

## Testing

```bash
# Requires a running Chroma server (or ephemeral client)
cd chromadb_intelligence
python -m pytest test_intelligence.py -v --tb=short
```
