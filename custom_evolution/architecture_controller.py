"""
ArchitectureController
======================
Enforces structural constraints on quantum circuit architectures during
Neural Architecture Search (NAS).

Constraint
----------
If layer n uses a CNOT on wire pair (i, j), then layer n+1 is forbidden
from using the same CNOT pair (i, j).  The first layer is unconstrained.

The controller is search-space-aware: it reads the CNOT configuration
directly from each block's metadata, so no hard-coded assumptions about
gate types or qubit counts are needed.

Usage
-----
    controller = ArchitectureController(search_space)

    # Build a valid random architecture from scratch
    arch = controller.random_architecture(n_layers)

    # Get all valid block indices that may follow a given block
    valid = controller.valid_next_blocks(prev_block_idx)

    # Repair an arbitrary architecture so it satisfies the constraint
    arch = controller.repair(arch)

    # Constrained versions of the EA operators
    arch  = controller.repair(arch)          # repair single
    pop   = controller.repair_population(pop) # repair entire population
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import sys


class ArchitectureController:
    """
    Enforces the no-repeated-CNOT constraint across consecutive layers.

    Parameters
    ----------
    search_space : SearchSpace
        The NAS search space instance.  Each element ``search_space[i]``
        must be a tuple ``(Rs_config, CNOT_config)`` where
        ``CNOT_config`` is either ``None`` or a tuple ``(ctrl, tgt)``.
    """

    def __init__(self, search_space):
        self.search_space = search_space
        self.n_blocks = len(search_space)

        # Pre-compute the CNOT wire pairs for every block index.
        # _cnot_of[i] is a frozenset of (ctrl, tgt) pairs (empty = no CNOT).
        self._cnot_of: list[frozenset] = []
        for idx in range(self.n_blocks):
            _, cnot_cfg = search_space[idx]
            self._cnot_of.append(frozenset(cnot_cfg))

        self._valid_after = []

        bar_width = 40

        for idx in range(self.n_blocks):
            # ----- progress bar -----
            progress = (idx + 1) / self.n_blocks
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(f"\rComputing valid followers [{bar}] {idx+1}/{self.n_blocks}")
            sys.stdout.flush()
            # ------------------------

            forbidden_cnots = self._cnot_of[idx]

            valid = [
                j for j in range(self.n_blocks)
                if not (forbidden_cnots and self._cnot_of[j] & forbidden_cnots)
            ]

            self._valid_after.append(valid)

        print()  # move to next line when done

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------

    def cnot_of(self, block_idx: int) -> frozenset:
        """Return the frozenset of CNOT wire pairs for *block_idx* (empty = no CNOT)."""
        return self._get_cnot(block_idx)

    def valid_next_blocks(self, prev_block_idx: int) -> list[int]:
        """
        Return all block indices that are valid directly after *prev_block_idx*.

        Parameters
        ----------
        prev_block_idx : int
            The block index of the preceding layer.

        Returns
        -------
        list[int]
            Non-empty list of valid block indices for the next layer.
        """
        return self._get_valid_after(prev_block_idx)

    # ------------------------------------------------------------------
    # Private helpers — handle both static and dynamic block indices
    # ------------------------------------------------------------------

    def _get_cnot(self, block_idx: int) -> frozenset:
        """Return CNOT frozenset for any index, static or dynamic."""
        if block_idx < self.n_blocks:
            return self._cnot_of[block_idx]
        _, cnots, _ = self.search_space._dynamic_extensions[block_idx]
        return frozenset(cnots)

    def _get_valid_after(self, block_idx: int) -> list[int]:
        """Return valid static follower indices for any index, static or dynamic."""
        if block_idx < self.n_blocks:
            return self._valid_after[block_idx]
        forbidden = self._get_cnot(block_idx)
        if not forbidden:
            return list(range(self.n_blocks))
        return [j for j in range(self.n_blocks) if not (self._cnot_of[j] & forbidden)]

    def is_valid(self, arch: np.ndarray) -> bool:
        """
        Check whether *arch* satisfies the constraint for all consecutive pairs.

        Parameters
        ----------
        arch : 1-D array of int
            Architecture represented as a sequence of block indices.

        Returns
        -------
        bool
        """
        for i in range(len(arch) - 1):
            cnot_now  = self._get_cnot(int(arch[i]))
            cnot_next = self._get_cnot(int(arch[i + 1]))
            if cnot_now and cnot_now & cnot_next:
                return False
        return True

    # ------------------------------------------------------------------
    # Architecture generation
    # ------------------------------------------------------------------

    def random_architecture(self, n_layers: int) -> np.ndarray:
        """
        Sample a fully valid random architecture of length *n_layers*.

        Layer 0 is chosen uniformly from all blocks.
        Each subsequent layer is chosen uniformly from the valid followers
        of the preceding layer.

        Parameters
        ----------
        n_layers : int

        Returns
        -------
        1-D np.ndarray of dtype int
        """
        arch = np.empty(n_layers, dtype=int)
        arch[0] = np.random.randint(0, self.n_blocks)
        for i in range(1, n_layers):
            choices = self._valid_after[int(arch[i - 1])]
            arch[i] = choices[np.random.randint(len(choices))]
        return arch

    # ------------------------------------------------------------------
    # Repair operator
    # ------------------------------------------------------------------

    def repair(self, arch: np.ndarray) -> np.ndarray:
        """
        Repair a single architecture so it satisfies the constraint.

        The repair is performed left-to-right:
        - Layer 0 is kept as-is (always valid).
        - For each subsequent layer, if the block is forbidden given the
          previous layer, it is replaced by a uniformly random valid block
          that shares the same block index modulo the number of valid choices
          (to preserve "genetic material" where possible), falling back to a
          fully random valid choice.

        Parameters
        ----------
        arch : 1-D array of int
            Architecture to repair (not modified in place).

        Returns
        -------
        1-D np.ndarray — repaired architecture satisfying the constraint.
        """
        arch = arch.copy().astype(int)
        for i in range(1, len(arch)):
            forbidden = self._get_cnot(int(arch[i - 1]))
            cnot_i    = self._get_cnot(int(arch[i]))
            if forbidden and forbidden & cnot_i:
                valid   = self._get_valid_after(int(arch[i - 1]))
                arch[i] = valid[int(arch[i]) % len(valid)]
        return arch

    def repair_population(self, population: np.ndarray) -> np.ndarray:
        """
        Repair every individual in *population*.

        Parameters
        ----------
        population : 2-D array of shape (pop_size, n_layers)

        Returns
        -------
        2-D np.ndarray — repaired population (new array, input unchanged).
        """
        return np.array([self.repair(ind) for ind in population])

    # ------------------------------------------------------------------
    # Constrained EA operators
    # ------------------------------------------------------------------

    def constrained_mutation(
        self,
        offspring: np.ndarray,
        mutation_prob: Optional[float] = None,
    ) -> np.ndarray:
        """
        Mutate *offspring* while guaranteeing the constraint is preserved.

        Each gene is mutated with probability *mutation_prob* (default
        1 / n_layers, matching the original polynomial mutation).  Mutated
        genes are replaced with a random valid block **given the preceding
        layer** rather than a fully random block index.

        Layer 0 mutates freely (no predecessor constraint).

        Parameters
        ----------
        offspring : 2-D array of shape (n_offspring, n_layers)
        mutation_prob : float, optional

        Returns
        -------
        2-D np.ndarray — mutated offspring satisfying the constraint.
        """
        mutated = offspring.copy().astype(int)
        n_individuals, n_layers = mutated.shape

        if mutation_prob is None:
            mutation_prob = 1.0 / n_layers

        for i in range(n_individuals):
            for j in range(n_layers):
                if np.random.rand() < mutation_prob:
                    if j == 0:
                        # Layer 0: unconstrained
                        mutated[i, j] = np.random.randint(0, self.n_blocks)
                    else:
                        # Layer j: must be a valid follower of layer j-1
                        valid = self._get_valid_after(int(mutated[i, j - 1]))
                        mutated[i, j] = valid[np.random.randint(len(valid))]
            # A mutated gene at position j may invalidate an unmutated gene at j+1.
            mutated[i] = self.repair(mutated[i])

        return mutated

    def constrained_crossover_and_repair(
        self,
        parents: np.ndarray,
        n_layers: int,
    ) -> np.ndarray:
        """
        Two-point crossover followed by constraint repair.

        The crossover itself is identical to the original two-point operator.
        After the swap, each offspring is repaired with :meth:`repair` so
        that the constraint is always satisfied before further processing.

        Parameters
        ----------
        parents : 2-D array of shape (n_parents, n_layers)
        n_layers : int

        Returns
        -------
        2-D np.ndarray — offspring satisfying the constraint.
        """
        n_parents = len(parents)
        offspring = np.empty_like(parents)

        for i in range(0, n_parents - 1, 2):
            p1 = parents[i].copy()
            p2 = parents[i + 1].copy()

            cut1, cut2 = sorted(np.random.choice(n_layers + 1, size=2, replace=False))

            c1 = np.concatenate([p1[:cut1], p2[cut1:cut2], p1[cut2:]])
            c2 = np.concatenate([p2[:cut1], p1[cut1:cut2], p2[cut2:]])

            offspring[i]     = self.repair(c1)
            offspring[i + 1] = self.repair(c2)

        if n_parents % 2 == 1:
            offspring[-1] = self.repair(parents[-1].copy())

        return offspring

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def constraint_summary(self) -> str:
        """
        Return a human-readable summary of the constraint graph.

        Shows, for every block, which CNOT (if any) it carries and how many
        valid successors it has.
        """
        lines = [
            "ArchitectureController — constraint summary",
            f"  Search space size : {self.n_blocks} blocks",
            "",
            "  Block | CNOT wires | # valid next blocks",
            "  " + "-" * 44,
        ]
        for idx in range(self.n_blocks):
            cnot = self._cnot_of[idx]
            cnot_str = ",".join(f"({c},{t})" for c, t in sorted(cnot)) if cnot else "  None  "
            n_valid = len(self._valid_after[idx])
            lines.append(f"  {idx:>5} | {cnot_str:^10} | {n_valid}")
        return "\n".join(lines)