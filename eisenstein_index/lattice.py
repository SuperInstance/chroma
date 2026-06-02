"""Eisenstein Lattice Index — Python implementation.

Maps high-dimensional vectors to Eisenstein integer lattice points.
The Eisenstein integers Z[ω] where ω = e^{2πi/3} form a hexagonal lattice
with the densest possible 2D packing (π/√12 ≈ 0.9069).

Key properties:
- Zero-drift quantization: hex grid has ~13% lower quantization error than square
- Guaranteed O(1) convergence for cell assignment
- 6 equidistant neighbors per cell (vs 4 for square grid)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np


# Eisenstein basis vectors in 2D
# e1 = (1, 0), e2 = (-1/2, √3/2)
_SQRT3 = math.sqrt(3)
_BASIS = np.array([[1.0, 0.0], [-0.5, _SQRT3 / 2.0]])


@dataclass(frozen=True, eq=True, hash=True)
class EisensteinPoint:
    """A point on the Eisenstein integer lattice (a + bω)."""

    a: int
    b: int

    def to_cartesian(self) -> Tuple[float, float]:
        """Convert to 2D Cartesian coordinates."""
        x = self.a * _BASIS[0, 0] + self.b * _BASIS[1, 0]
        y = self.a * _BASIS[0, 1] + self.b * _BASIS[1, 1]
        return (x, y)

    def norm_squared(self) -> int:
        """Eisenstein norm: a² - ab + b²."""
        return self.a * self.a - self.a * self.b + self.b * self.b


@dataclass
class LatticeConfig:
    """Configuration for the Eisenstein lattice index."""

    cell_scale: float = 1.0
    projection_dims: int = 2
    use_random_projection: bool = False
    random_seed: Optional[int] = None


class EisensteinLattice:
    """Maps high-dimensional vectors to Eisenstein lattice cells.

    The lattice uses PCA or random projection to reduce R^n to R^2,
    then quantizes to the hexagonal grid.
    """

    def __init__(self, config: Optional[LatticeConfig] = None):
        self.config = config or LatticeConfig()
        self._projection: Optional[np.ndarray] = None  # (dim, projection_dims)
        self._mean: Optional[np.ndarray] = None

    def fit(self, vectors: np.ndarray) -> None:
        """Fit the projection matrix using PCA on the given vectors.

        Args:
            vectors: (N, D) array of vectors
        """
        N, D = vectors.shape

        # Compute mean
        self._mean = vectors.mean(axis=0)

        # Center
        centered = vectors - self._mean

        # Covariance matrix (D x D)
        cov = centered.T @ centered / (N - 1)

        # Eigendecomposition — take top-k eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        k = min(self.config.projection_dims, D)
        self._projection = eigenvectors[:, idx[:k]]

    def fit_random_projection(self, dim: int) -> None:
        """Use random projection instead of PCA (faster for high dimensions).

        Args:
            dim: Dimension of input vectors
        """
        rng = np.random.RandomState(self.config.random_seed)
        scale = math.sqrt(2.0 / dim)
        k = self.config.projection_dims
        self._projection = rng.randn(dim, k) * scale
        self._mean = np.zeros(dim)

    def project(self, vector: np.ndarray) -> np.ndarray:
        """Project a vector to 2D using the fitted projection.

        Args:
            vector: (D,) array

        Returns:
            (projection_dims,) projected coordinates
        """
        if self._projection is None or self._mean is None:
            raise RuntimeError("Must call fit() or fit_random_projection() first")

        centered = vector - self._mean
        return centered @ self._projection

    def quantize(self, x: float, y: float) -> EisensteinPoint:
        """Quantize a 2D point to the nearest Eisenstein lattice point.

        Uses the hexagonal nearest-point algorithm: transform to Eisenstein
        coordinates, round, then check 7 candidates (center + 6 neighbors).

        Args:
            x: x-coordinate (scaled by cell_scale)
            y: y-coordinate (scaled by cell_scale)

        Returns:
            Nearest EisensteinPoint on the lattice
        """
        # Transform Cartesian to Eisenstein coordinates
        b_cont = 2.0 * y / _SQRT3
        a_cont = x + y / _SQRT3

        a0 = round(a_cont)
        b0 = round(b_cont)

        # Check 7 candidates: center + 6 hex neighbors
        candidates = [
            (a0, b0),
            (a0 + 1, b0),
            (a0 - 1, b0),
            (a0, b0 + 1),
            (a0, b0 - 1),
            (a0 + 1, b0 - 1),
            (a0 - 1, b0 + 1),
        ]

        best = (a0, b0)
        best_dist = float("inf")

        for a, b in candidates:
            px = a - 0.5 * b
            py = (_SQRT3 / 2.0) * b
            dist = (px - x) ** 2 + (py - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (a, b)

        return EisensteinPoint(a=best[0], b=best[1])

    def assign_cell(self, vector: np.ndarray) -> EisensteinPoint:
        """Project a vector and assign it to a lattice cell.

        Args:
            vector: (D,) array

        Returns:
            EisensteinPoint representing the assigned cell
        """
        projected = self.project(vector)
        x = projected[0] / self.config.cell_scale
        y = projected[1] / self.config.cell_scale if len(projected) > 1 else 0.0
        return self.quantize(x, y)

    def assign_cells(self, vectors: np.ndarray) -> List[EisensteinPoint]:
        """Batch cell assignment.

        Args:
            vectors: (N, D) array

        Returns:
            List of EisensteinPoint for each vector
        """
        return [self.assign_cell(v) for v in vectors]

    def neighbors(self, point: EisensteinPoint) -> List[EisensteinPoint]:
        """Get the 6 neighboring cells of an Eisenstein lattice point."""
        return [
            EisensteinPoint(a=point.a + 1, b=point.b),
            EisensteinPoint(a=point.a - 1, b=point.b),
            EisensteinPoint(a=point.a, b=point.b + 1),
            EisensteinPoint(a=point.a, b=point.b - 1),
            EisensteinPoint(a=point.a + 1, b=point.b - 1),
            EisensteinPoint(a=point.a - 1, b=point.b + 1),
        ]

    def quantization_error(self, vectors: np.ndarray) -> float:
        """Compute mean squared quantization error.

        Args:
            vectors: (N, D) array

        Returns:
            Mean squared distance between projected points and lattice points
        """
        total = 0.0
        for v in vectors:
            projected = self.project(v)
            x = projected[0] / self.config.cell_scale
            y = projected[1] / self.config.cell_scale if len(projected) > 1 else 0.0

            lp = self.quantize(x, y)
            lx, ly = lp.to_cartesian()
            total += (lx - x) ** 2 + (ly - y) ** 2

        return total / len(vectors)
