"""CLI entry point for chromadb_intelligence."""

from __future__ import annotations

import argparse
import sys
import json

from .core import CollectionIntelligence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chroma-intel",
        description="Spectral graph intelligence for Chroma collections.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    analyze_parser = sub.add_parser("analyze", help="Analyze a collection")
    analyze_parser.add_argument(
        "--collection", "-c", required=True, help="Collection name"
    )
    analyze_parser.add_argument(
        "--k", type=int, default=15, help="k-NN graph parameter (default 15)"
    )
    analyze_parser.add_argument(
        "--tenant", default="default_tenant", help="Chroma tenant"
    )
    analyze_parser.add_argument(
        "--database", default="default_database", help="Chroma database"
    )
    analyze_parser.add_argument(
        "--baseline", help="Path to JSON baseline snapshot for drift detection"
    )
    analyze_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON"
    )

    # drift
    drift_parser = sub.add_parser("drift", help="Detect embedding drift")
    drift_parser.add_argument(
        "--collection", "-c", required=True, help="Collection name"
    )
    drift_parser.add_argument(
        "--baseline", required=True, help="Path to previous snapshot JSON"
    )
    drift_parser.add_argument(
        "--k", type=int, default=15, help="k-NN graph parameter (default 15)"
    )
    drift_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON"
    )

    args = parser.parse_args(argv)

    ci = CollectionIntelligence()

    if args.command == "analyze":
        baseline = None
        if args.baseline:
            baseline = CollectionIntelligence.load_snapshot(args.baseline)[-1]["report"]
        report = ci.analyze(
            collection_name=args.collection,
            k=args.k,
            tenant=args.tenant,
            database=args.database,
            baseline=baseline,
        )
        if args.json:
            print(json.dumps(report.dict(), indent=2))
        else:
            _print_report(report)
        ci.save_snapshot(f"{args.collection}_snapshot.json")

    elif args.command == "drift":
        snapshot = CollectionIntelligence.load_snapshot(args.baseline)
        last_report = snapshot[-1]["report"]
        drift = ci.detect_drift(
            collection_name=args.collection,
            snapshot_before=last_report,
            k=args.k,
        )
        if args.json:
            print(json.dumps(drift.dict(), indent=2))
        else:
            print(drift.message)
            print(f"  Fiedler (before): {drift.fiedler_before:.4f}")
            print(f"  Fiedler (now):    {drift.fiedler_now:.4f}")
            print(f"  Δ:                {drift.fiedler_delta:+.4f}")
            print(f"  Spectral JSD:     {drift.jsd_spectral:.4f}")

    return 0


def _print_report(r) -> None:
    """Pretty-print a SpectralReport."""
    print(f"{'=' * 60}")
    print(f"  CHROMA COLLECTION INTELLIGENCE REPORT")
    print(f"{'=' * 60}")
    print(f"  Points:             {r.num_points}")
    print(f"  Embedding dim:      {r.embedding_dim}")
    print(f"  Fiedler value:      {r.fiedler_value:.6f}")
    print(f"  Spectral gap:       {r.spectral_gap:.6f}")
    print(f"  Cheeger constant:   {r.cheeger_constant:.6f}")
    print(f"  Communities found:  {r.num_communities}")
    print(f"  JSD (per cluster):  {r.jsd_per_cluster}")
    print(f"  Drift detected:     {r.drift_detected}")
    if r.drift_message:
        print(f"  Drift message:      {r.drift_message}")
    print(f"{'=' * 60}")
    print()
    print("  INSIGHT:")
    print(f"  Your embeddings say {r.num_communities} categories.", end=" ")
    # Count unique metadata categories would need metadata; just use geometry
    print(f"The geometry says {r.num_communities}.")
    if r.fiedler_value < 0.3:
        print("  ⚠ Low Fiedler value — clusters are poorly separated.")
        print("  Consider improving your embedding model.")
    elif r.fiedler_value > 0.7:
        print("  ✓ High Fiedler value — strong cluster separation.")
    if r.drift_detected:
        print("  ⚠ Embedding drift — recent embeddings degrade quality.")
    print()


if __name__ == "__main__":
    sys.exit(main())
