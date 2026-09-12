"""Controlled diagnostic scenario specifications for MISDA.

The historical ``Case``/``MOP`` split is retained only in generator names while
R5 migrates the implementation. This module provides one conceptual catalogue:
each scenario declares its generating variables, clean problem map, observation
model, truth dimensions, legacy generating-family partition, optional structural
units, and diagnostic tags.

Generating families are descriptive groupings and are deliberately not used to
infer structural dimension or structural units. Case 5 (one cumulative family
but 20 structural units) and MOP-B (three functional families but no unambiguous
structural partition) make this distinction explicit.

No truth in this catalogue is inferred from observed data or MISDA output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .cases import (
    make_case1_independence,
    make_case2_total_redundancy,
    make_case3_block_structure,
    make_case4_two_big_blocks,
    make_case5_chain_structure,
    make_case6_mixed_structure,
    make_case7_pure_conflict_groups,
)
from .mop import (
    mopA_monotonic_redundancy,
    mopB_tradeoff_with_redundancies,
    mopC_latent_blocks_4x5,
    mopD_pure_conflict_groups,
    mopE_partial_redundancy_noisy,
    mopF_regime_switching,
)


@dataclass(frozen=True)
class DiagnosticScenario:
    """Scientific specification for one controlled diagnostic problem."""

    id: str
    historical_name: str
    generator: Callable
    variables: tuple[str, ...]
    clean_map: str
    observation: str
    latent_expected: int
    structural_expected: int
    family_sizes: tuple[int, ...]
    tags: frozenset[str]
    structural_unit_sizes: tuple[int, ...] | None = None

    def validate_legacy_contract(self, *, N: int = 32, seed: int = 123) -> None:
        """Check legacy output against the explicit scenario declaration."""
        frame, truth = self.generator(N=N, seed=seed)
        if frame.shape[1] != sum(self.family_sizes):
            raise AssertionError(
                f"{self.id}: objective count does not match generating families"
            )
        if truth["latent_expected"] != self.latent_expected:
            raise AssertionError(f"{self.id}: latent truth mismatch")
        if truth["structural_expected"] != self.structural_expected:
            raise AssertionError(f"{self.id}: structural truth mismatch")
        observed_sizes = tuple(len(block) for block in truth["blocks_expected"])
        if observed_sizes != self.family_sizes:
            raise AssertionError(f"{self.id}: generating-family declaration mismatch")
        if self.structural_unit_sizes is not None:
            if sum(self.structural_unit_sizes) != frame.shape[1]:
                raise AssertionError(
                    f"{self.id}: structural units must partition all objectives"
                )
            if len(self.structural_unit_sizes) != self.structural_expected:
                raise AssertionError(
                    f"{self.id}: structural-unit count must match structural truth"
                )


DIAGNOSTIC_SCENARIOS = (
    DiagnosticScenario(
        id="independence",
        historical_name="Case 1 - Total independence",
        generator=make_case1_independence,
        variables=tuple(f"x{i}" for i in range(1, 21)),
        clean_map="f_i = x_i for i=1..20",
        observation="identity",
        latent_expected=20,
        structural_expected=20,
        family_sizes=(1,) * 20,
        structural_unit_sizes=(1,) * 20,
        tags=frozenset({"independence", "linear"}),
    ),
    DiagnosticScenario(
        id="total_redundancy",
        historical_name="Case 2 - Total redundancy",
        generator=make_case2_total_redundancy,
        variables=("x",),
        clean_map="20 copies of x",
        observation="legacy additive Gaussian perturbation on each copy",
        latent_expected=1,
        structural_expected=1,
        family_sizes=(20,),
        structural_unit_sizes=(20,),
        tags=frozenset({"total_redundancy", "linear", "noisy_observation"}),
    ),
    DiagnosticScenario(
        id="blocks_4x5",
        historical_name="Case 3 - Blocks (4 x 5)",
        generator=make_case3_block_structure,
        variables=("x1", "x2", "x3", "x4"),
        clean_map="five copies of each independent factor x1..x4",
        observation="legacy additive Gaussian perturbation on each copy",
        latent_expected=4,
        structural_expected=4,
        family_sizes=(5, 5, 5, 5),
        structural_unit_sizes=(5, 5, 5, 5),
        tags=frozenset({"block_redundancy", "linear", "noisy_observation"}),
    ),
    DiagnosticScenario(
        id="blocks_2x10",
        historical_name="Case 4 - Blocks (2 x 10)",
        generator=make_case4_two_big_blocks,
        variables=("x1", "x2"),
        clean_map="ten copies of each independent factor x1,x2",
        observation="legacy additive Gaussian perturbation on each copy",
        latent_expected=2,
        structural_expected=2,
        family_sizes=(10, 10),
        structural_unit_sizes=(10, 10),
        tags=frozenset({"block_redundancy", "linear", "noisy_observation"}),
    ),
    DiagnosticScenario(
        id="transitive_chain",
        historical_name="Case 5 - Chain",
        generator=make_case5_chain_structure,
        variables=tuple(f"x{i}" for i in range(1, 21)),
        clean_map=(
            "f1=x1; fj=x1+0.2*x2+...+0.2*xj for j=2..20 "
            "(triangular cumulative map)"
        ),
        observation="identity; innovations are generating degrees of freedom, not noise",
        latent_expected=20,
        structural_expected=20,
        family_sizes=(20,),
        structural_unit_sizes=(1,) * 20,
        tags=frozenset({"transitive_chaining", "linear", "known_failure_mode"}),
    ),
    DiagnosticScenario(
        id="mixed_independent_and_blocks",
        historical_name="Case 6 - Mixed (indep + latents)",
        generator=make_case6_mixed_structure,
        variables=tuple(f"x{i}" for i in range(1, 13)),
        clean_map="10 independent objectives plus two factors replicated five times each",
        observation="legacy additive Gaussian perturbation on replicated factors",
        latent_expected=12,
        structural_expected=12,
        family_sizes=(1,) * 10 + (5, 5),
        structural_unit_sizes=(1,) * 10 + (5, 5),
        tags=frozenset({"mixed_structure", "block_redundancy", "noisy_observation"}),
    ),
    DiagnosticScenario(
        id="antagonistic_linear_groups",
        historical_name="Case 7 - Structural conflict (anti-corr) 2-groups",
        generator=make_case7_pure_conflict_groups,
        variables=("x",),
        clean_map="10 copies of x and 10 copies of -x",
        observation="legacy additive Gaussian perturbation on each objective",
        latent_expected=1,
        structural_expected=2,
        family_sizes=(10, 10),
        structural_unit_sizes=(10, 10),
        tags=frozenset({"antagonistic_conflict", "linear", "noisy_observation"}),
    ),
    DiagnosticScenario(
        id="monotonic_redundancy",
        historical_name="MOP-A — Monotonic redundancy",
        generator=mopA_monotonic_redundancy,
        variables=("x",),
        clean_map="20 monotone nonlinear transformations of x",
        observation="optional additive Gaussian perturbation after each transform",
        latent_expected=1,
        structural_expected=1,
        family_sizes=(20,),
        structural_unit_sizes=(20,),
        tags=frozenset({"total_redundancy", "nonlinear", "monotonic"}),
    ),
    DiagnosticScenario(
        id="tradeoff_redundancies",
        historical_name="MOP-B — Trade-off + redundancies",
        generator=mopB_tradeoff_with_redundancies,
        variables=("a", "b"),
        clean_map="cost C(a,b), consumption E(a,b), and performance Q(a,b) families",
        observation="legacy small perturbation on selected family replicas",
        latent_expected=2,
        structural_expected=2,
        family_sizes=(7, 7, 6),
        structural_unit_sizes=None,
        tags=frozenset({"tradeoff", "nonlinear", "family_redundancy"}),
    ),
    DiagnosticScenario(
        id="nonlinear_blocks_4x5",
        historical_name="MOP-C — Latent blocks",
        generator=mopC_latent_blocks_4x5,
        variables=("u", "v", "w", "z"),
        clean_map="five nonlinear transforms of each independent factor u,v,w,z",
        observation="legacy perturbation on one w-family replica",
        latent_expected=4,
        structural_expected=4,
        family_sizes=(5, 5, 5, 5),
        structural_unit_sizes=(5, 5, 5, 5),
        tags=frozenset({"block_redundancy", "nonlinear"}),
    ),
    DiagnosticScenario(
        id="antagonistic_nonlinear_groups",
        historical_name="MOP-D — Structural conflict (anti-corr) 2-groups",
        generator=mopD_pure_conflict_groups,
        variables=("x",),
        clean_map="10 monotone transforms of x and 10 of 1-x",
        observation="optional additive Gaussian perturbation after each transform",
        latent_expected=1,
        structural_expected=2,
        family_sizes=(10, 10),
        structural_unit_sizes=(10, 10),
        tags=frozenset({"antagonistic_conflict", "nonlinear", "tradeoff"}),
    ),
    DiagnosticScenario(
        id="overlapping_factors",
        historical_name="MOP-E — Partial redundancy + noise",
        generator=mopE_partial_redundancy_noisy,
        variables=("a", "b"),
        clean_map="families on a, b, and the overlapping compound a+b",
        observation="legacy perturbation on selected a-family replicas",
        latent_expected=2,
        structural_expected=2,
        family_sizes=(10, 4, 6),
        structural_unit_sizes=None,
        tags=frozenset({"overlapping_factors", "nonlinear", "partial_redundancy"}),
    ),
    DiagnosticScenario(
        id="regime_switching",
        historical_name="MOP-F — Regimes (mixture, M=20)",
        generator=mopF_regime_switching,
        variables=("a", "b"),
        clean_map="nonlinear transforms of regime mixture L(a,b) plus transforms of b",
        observation="optional additive Gaussian perturbation after clean transforms",
        latent_expected=2,
        structural_expected=2,
        family_sizes=(10, 10),
        structural_unit_sizes=(10, 10),
        tags=frozenset({"regime_switching", "nonlinear", "known_failure_mode"}),
    ),
)

DIAGNOSTIC_BY_ID = {scenario.id: scenario for scenario in DIAGNOSTIC_SCENARIOS}

if len(DIAGNOSTIC_BY_ID) != len(DIAGNOSTIC_SCENARIOS):
    raise RuntimeError("Diagnostic scenario ids must be unique")
