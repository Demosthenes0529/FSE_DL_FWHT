# Vendored OCP subset

This directory provides the SAT modeling dependencies from the [Open Cryptanalysis Platform](https://github.com/Open-CP/OCP), based on commit `a3c8d32f813275643bb8af98eb5160d04921baa7`. See `LICENSE` for the license and `UPSTREAM_README.md` for the upstream documentation.

Only the SAT modeling components needed to reproduce the DL-FWHT experiments are included:

| Directory | Purpose |
| --- | --- |
| `attacks/` | Model configuration, endpoint constraints, and result parsing for differential and linear trails. |
| `operators/` | XOR, AND, modular addition, S-boxes, matrices, and basic operators. |
| `primitives/` | Existing OCP cipher models and the base classes required by the custom SIMON-like models. |
| `tools/` | SAT constraint generation, objective functions, search scheduling, and resource monitoring. |
| `solving/` | The common OCP solver interface, which DL-FWHT workers bind to external CryptoMiniSat. |
| `variables/` | OCP bit-variable and word-variable definitions. |
| `visualisations/` | Visualization interfaces required by the OCP attack modules; plotting libraries are loaded only when plotting is requested. |
| `files/sbox_modeling/` | SAT S-box constraint templates for PRESENT/KNOT. |

DL-FWHT loads these files from `OCP/` at the project root by default. Set `OCP_ROOT` to override this location.

## File descriptions

| File | Function |
| --- | --- |
| `attacks/attack_trace.py` | Result data structures for differential and linear trails. |
| `attacks/differential_cryptanalysis.py` | Differential-trail model configuration, endpoint constraints, and solution parsing. |
| `attacks/linear_cryptanalysis.py` | Linear-trail model configuration, endpoint constraints, and solution parsing. |
| `operators/operators.py` | OCP operator base classes and basic operators such as identity mappings and rotations. |
| `operators/boolean_operators.py` | Models for XOR, AND, AND-XOR, and XOR with a constant. |
| `operators/modular_operators.py` | Differential and linear models for modular addition and addition of a constant. |
| `operators/Sbox.py` | DDT/LAT SAT models for generic S-boxes and the PRESENT/KNOT S-boxes. |
| `operators/matrix.py` | Linear matrix operators required by the OCP primitive base classes. |
| `primitives/primitives.py` | Model base classes for permutations and block ciphers. |
| `primitives/speck.py` | SPECK primitives for differential and linear trails. |
| `primitives/simon.py` | SIMON primitives for differential and linear trails. |
| `primitives/present.py` | The internal PRESENT permutation model shared by differential and linear trails. |
| `tools/model_constraints.py` | Converts operators and fixed endpoints into CNF constraints. |
| `tools/model_objective.py` | Generates and parses trail-weight objective functions. |
| `tools/sat_search.py` | SAT search strategies and the solver interface. |
| `tools/milp_search.py` | MILP compatibility interfaces required by imports in the attack modules; this project does not invoke MILP solving. |
| `tools/minimize_logic.py` | Logic minimization helpers for generating constraint templates. |
| `tools/polyhedron.py` | Convex-hull helpers for generating constraint templates. |
| `tools/resource_monitor.py` | Solver runtime and resource statistics. |
| `solving/solving.py` | The common SAT/MILP solver entry point; this project redirects SAT calls to external CryptoMiniSat. |
| `variables/variables.py` | Bit variables, word variables, and their connections. |
| `visualisations/visualisations.py` | Optional plotting interfaces required by the OCP attack modules, with lazy imports. |
| `files/sbox_modeling/constraints_sat_*_Sbox_*_PR_*.txt` | Four weighted differential/linear S-box SAT templates, two each for PRESENT and KNOT; only the modes used by the shared worker are included. |

The symbolic round structure for PRESENT is defined in `primitives/present.py`, and its fixed-key encryption implementation is in `Cipher/Present/`. The KNOT round structure is implemented in `Framework/sat/ocp_factory.py`, using the `KNOT_Sbox` provided here. The shared OCP worker solves both differential and linear weights. Independent trail replay for both ciphers is implemented in `Framework/search/adapter.py` to check that model outputs agree with the actual S-boxes and bit ordering.
