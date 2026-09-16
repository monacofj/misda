<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Contributing to MISDA

Thank you for your interest in MISDA!

This project is open source, and we welcome contributions from the community to help improve the framework.

## How to Contribute

If you are interested in contributing to the source code, documentation, or proposing new features, please, feel free to contact the authors.

See the [AUTHORS](AUTHORS) file for contact details.

## Public capability preservation

Refactors are behavior-preserving by default. User-visible scientific capabilities are part of the project contract even when their exact textual formatting is allowed to evolve.

Before deleting, replacing, or classifying an implementation as legacy, contributors must:

1. identify the public behavior and diagnostics currently provided;
2. preserve equivalent behavior in the replacement implementation, or document an intentional breaking change in a superseding ADR;
3. add replacement contract tests before removing existing regression tests;
4. never delete a public-behavior test merely because the implementation it protects is being removed;
5. run the named acceptance gates and the complete test suite;
6. verify benchmark outputs separately from MISDA-native outputs so benchmark truth cannot mask a loss of native capability.

For reporting specifically, `tests/test_reporting_contract.py` is a minimum-capability gate. Changes to `MISSet.report()` must preserve its information families unless ADR 0016 is explicitly superseded.
