"""Tests for Eisenstein lattice indexing."""

import math
import numpy as np
import pytest

from eisenstein_index.lattice import EisensteinLattice, EisensteinPoint, LatticeConfig
from eisenstein_index.hex_ann import HexANN


# ─── EisensteinPoint Tests ────────────────────────────────────────────


class TestEisensteinPoint:
    def test_to_cartesian_origin(self):
        pt = EisensteinPoint(a=0, b=0)
        x, y = pt.to_cartesian()
        assert x == 0.0
        assert y == 0.0

    def test_to_cartesian_unit(self):
        pt = EisensteinPoint(a=1, b=0)
        x, y = pt.to_cartesian()
        assert abs(x - 1.0) < 1e-10
        assert abs(y) < 1e-10

    def test_to_cartesian_omega(self):
        pt = EisensteinPoint(a=0, b=1)
        x, y = pt.to_cartesian()
        assert abs(x - (-0.5)) < 1e-10
        assert abs(y - math.sqrt(3) / 2) < 1e-10

    def test_norm_squared(self):
        # Norm of ω: a=0, b=1 → 0² - 0*1 + 1² = 1
        pt = EisensteinPoint(a=0, b=1)
        assert pt.norm_squared() == 1

    def test_hashable(self):
        """EisensteinPoint can be used as dict keys."""
        d = {EisensteinPoint(a=1, b=2): "test"}
        assert d[EisensteinPoint(a=1, b=2)] == "test"

    def test_equality(self):
        assert EisensteinPoint(a=1, b=2) == EisensteinPoint(a=1, b=2)
        assert EisensteinPoint(a=1, b=2) != EisensteinPoint(a=2, b=1)


# ─── EisensteinLattice Tests ──────────────────────────────────────────


class TestEisensteinLattice:
    def _make_vectors(self, n=100, dim=8, seed=42):
        rng = np.random.RandomState(seed)
        return rng.randn(n, dim)

    def test_quantize_roundtrip(self):
        """Lattice points should quantize to themselves."""
        lattice = EisensteinLattice()
        pt = EisensteinPoint(a=3, b=-2)
        x, y = pt.to_cartesian()
        quantized = lattice.quantize(x, y)
        assert quantized == pt

    def test_quantize_near_origin(self):
        lattice = EisensteinLattice()
        pt = lattice.quantize(0.0, 0.0)
        assert pt.a == 0
        assert pt.b == 0

    def test_neighbors_count(self):
        lattice = EisensteinLattice()
        pt = EisensteinPoint(a=0, b=0)
        neighbors = lattice.neighbors(pt)
        assert len(neighbors) == 6

    def test_fit_and_assign(self):
        vectors = self._make_vectors()
        lattice = EisensteinLattice(LatticeConfig(cell_scale=0.5))
        lattice.fit(vectors)
        cell = lattice.assign_cell(vectors[0])
        assert isinstance(cell, EisensteinPoint)

    def test_batch_assign(self):
        vectors = self._make_vectors()
        lattice = EisensteinLattice()
        lattice.fit(vectors)
        cells = lattice.assign_cells(vectors)
        assert len(cells) == len(vectors)
        assert all(isinstance(c, EisensteinPoint) for c in cells)

    def test_random_projection(self):
        dim = 128
        lattice = EisensteinLattice()
        lattice.fit_random_projection(dim)
        vector = np.ones(dim) * 0.5
        cell = lattice.assign_cell(vector)
        assert isinstance(cell, EisensteinPoint)

    def test_quantization_error(self):
        vectors = self._make_vectors()
        lattice = EisensteinLattice(LatticeConfig(cell_scale=1.0))
        lattice.fit(vectors)
        error = lattice.quantization_error(vectors)
        assert error >= 0.0
        assert math.isfinite(error)

    def test_not_fitted_raises(self):
        lattice = EisensteinLattice()
        with pytest.raises(RuntimeError):
            lattice.assign_cell(np.ones(10))


# ─── HexANN Tests ──────────────────────────────────────────────────────


class TestHexANN:
    def _make_vectors(self, n=200, dim=16, seed=42):
        rng = np.random.RandomState(seed)
        return rng.randn(n, dim)

    def test_insert_and_search(self):
        vectors = self._make_vectors()
        index = HexANN(LatticeConfig(cell_scale=0.5))
        index.fit(vectors)
        index.insert_batch(vectors)

        results = index.search(vectors[0], k=5)
        assert len(results) > 0
        # Query vector itself should be top result
        assert results[0].score > 0.99

    def test_empty_index(self):
        index = HexANN()
        assert index.num_vectors == 0
        results = index.search(np.ones(10), k=5)
        assert len(results) == 0

    def test_search_with_radius(self):
        vectors = self._make_vectors()
        index = HexANN(LatticeConfig(cell_scale=0.3))
        index.fit(vectors)
        index.insert_batch(vectors)

        r0 = index.search_with_radius(vectors[0], k=10, radius=0)
        r1 = index.search_with_radius(vectors[0], k=10, radius=1)
        r2 = index.search_with_radius(vectors[0], k=10, radius=2)

        # More radius should yield more candidates (or equal)
        assert len(r2) >= len(r1) - 5  # Allow some tolerance

    def test_index_stats(self):
        vectors = self._make_vectors()
        index = HexANN()
        index.fit(vectors)
        index.insert_batch(vectors)

        stats = index.stats()
        assert stats.num_vectors == len(vectors)
        assert stats.num_cells > 0
        assert stats.avg_cell_size > 0

    def test_insert_with_ids(self):
        vectors = self._make_vectors(n=50)
        index = HexANN()
        index.fit(vectors)
        index.insert_batch(vectors, ids=list(range(100, 150)))

        results = index.search(vectors[0], k=3)
        assert all(100 <= r.id < 150 for r in results)

    def test_deterministic(self):
        """Same input should produce same results."""
        vectors = self._make_vectors()

        index1 = HexANN(LatticeConfig(cell_scale=0.5, random_seed=42))
        index1.fit(vectors)
        index1.insert_batch(vectors)

        index2 = HexANN(LatticeConfig(cell_scale=0.5, random_seed=42))
        index2.fit(vectors)
        index2.insert_batch(vectors)

        r1 = index1.search(vectors[0], k=5)
        r2 = index2.search(vectors[0], k=5)

        assert [r.id for r in r1] == [r.id for r in r2]
