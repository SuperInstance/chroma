![Chroma](./docs/assets/chroma-wordmark-color.png#gh-light-mode-only)
![Chroma](./docs/assets/chroma-wordmark-white.png#gh-dark-mode-only)

<p align="center">
    <b>Chroma - the open-source data infrastructure for AI</b>. <br />
</p>

<p align="center">
  <a href="https://discord.gg/MMeYNTmh3x" target="_blank">
      <img src="https://img.shields.io/discord/1073293645303795742?cacheSeconds=3600" alt="Discord">
  </a> |
  <a href="https://github.com/chroma-core/chroma/blob/master/LICENSE" target="_blank">
      <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a> |
  <a href="https://docs.trychroma.com/" target="_blank">
      Docs
  </a> |
  <a href="https://www.trychroma.com/" target="_blank">
      Homepage
  </a>
</p>

```bash
pip install chromadb # python client
# for javascript, npm install chromadb!
# for client-server mode, chroma run --path /chroma_db_path
```

## Chroma Cloud

Our hosted service, Chroma Cloud, powers serverless vector, hybrid, and full-text search. It's extremely fast, cost-effective, scalable and painless. Create a DB and try it out in under 30 seconds with $5 of free credits.

[Get started with Chroma Cloud](https://trychroma.com/signup)

## API

The core API is only 4 functions (run our [💡 Google Colab](https://colab.research.google.com/drive/1QEzFyqnoFxq7LUGyP1vzR4iLt9PpCDXv?usp=sharing)):

```python
import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["This is document1", "This is document2"], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
    ids=["doc1", "doc2"], # unique for each doc
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)
```

Learn about all features on our [Docs](https://docs.trychroma.com)

## Get involved

Chroma is a rapidly developing project. We welcome PR contributors and ideas for how to improve the project.
- [Join the conversation on Discord](https://discord.com/invite/chromadb) - `#contributing` channel
- [Review the 🛣️ Roadmap and contribute your ideas](https://docs.trychroma.com/docs/overview/oss#roadmap)
- [Grab an issue and open a PR](https://github.com/chroma-core/chroma/issues) - [`Good first issue tag`](https://github.com/chroma-core/chroma/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Read our contributing guide](https://docs.trychroma.com/docs/overview/oss#contributing)

**Release Cadence**
We currently release new tagged versions of the `pypi` and `npm` packages on Mondays. Hotfixes go out at any time during the week.

## chromadb-intelligence — Spectral Analysis for Chroma Collections

`chromadb_intelligence` is a Python package that extracts the **vector index graph** from a Chroma collection and runs **spectral graph analysis** to answer:

> *How many categories does your embedding space actually have?*

Key capabilities:

- **Fiedler vector** — identifies natural clusters in embedding space
- **Cheeger constant** — quantifies cluster separation quality
- **Spectral community detection** — compare labeled categories vs geometric clusters
- **Jensen-Shannon divergence** — distance distribution differences between clusters
- **Embedding drift detector** — compare spectral signature now vs 30 days ago

### Quick Start

```bash
pip install chromadb chromadb-intelligence
```

```python
from chromadb_intelligence import CollectionIntelligence

ci = CollectionIntelligence()
report = ci.analyze("my_docs")

# Your embeddings say 12 categories. The geometry says 19.
print(f"Your labels: 12 | Geometry: {report.num_communities}")
print(f"Fiedler value: {report.fiedler_value:.4f}")
```

### CLI

```bash
chroma-intel analyze --collection my_docs --json
chroma-intel drift --collection my_docs --baseline my_docs_snapshot.json
```

### Drift Detection

```python
# Today
snapshot = ci.analyze("my_docs")
ci.save_snapshot("snapshot.json")

# 30 days later
drift = ci.detect_drift("my_docs", snapshot_before=ci.load_snapshot("snapshot.json")[-1]["report"])
drift.message
# "Embedding drift detected: Fiedler value dropped from 0.42 to 0.18..."
```

### Insight Engine

```bash
# Your embeddings say 12 categories. The geometry says 19. Your embedding model needs work.
chroma-intel analyze --collection my_docs
```

```
============================================================
  CHROMA COLLECTION INTELLIGENCE REPORT
============================================================
  Points:             8543
  Embedding dim:      768
  Fiedler value:      0.241234
  Spectral gap:       0.087656
  Cheeger constant:   0.312345
  Communities found:  19
  JSD (per cluster):  {'0_vs_1': 0.543, '0_vs_2': 0.621, ...}
  Drift detected:     True
============================================================

  INSIGHT:
  Your embeddings say 12 categories. The geometry says 19.
  ⚠ Low Fiedler value — clusters are poorly separated.
    Consider improving your embedding model.
```

## License

[Apache 2.0](./LICENSE)
