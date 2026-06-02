"""chromadb_intelligence — spectral graph analysis for chroma collections.

Extract the vector index graph from a chroma collection, run spectral analysis
(Fiedler vector, Cheeger constant, community detection), Jensen-Shannon divergence
between distance distributions per cluster, and embedding drift detection over time.
"""

from .core import CollectionIntelligence
from .cli import main as cli_main

__all__ = ["CollectionIntelligence", "cli_main"]
__version__ = "0.1.0"
