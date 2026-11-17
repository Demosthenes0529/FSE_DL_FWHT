# FSE_DL_FWHT

Differential-linear search code for SPECK, SIMON, SIMECK, CHAM, PRESENT, and KNOT. The package includes middle-segment spectrum evaluation, three-segment search, round-split exploration, fixed-key validation, and trail aggregation.

## Environment and conventions

Run all commands from `FSE_DL_FWHT/`. Python dependencies include NumPy, PyTorch, psutil, and tqdm. CUDA spectrum evaluation requires a working NVIDIA driver and Triton. Outer-segment search invokes an external CryptoMiniSat executable; trail aggregation also requires `pycryptosat` in the active Python environment. Native programs use g++ with C++17 and GCC with OpenMP.

```bash
python -m pip install -r Framework/requirements.txt
make -C Framework native
```

`Framework/requirements.txt` lists the Python dependencies. `Framework/Makefile` builds the validators, window dynamic programming (DP) program, and reverse sampler. Build outputs default to `Framework/build/`. To use another location:

```bash
make -C Framework native BUILD_DIR=/tmp/dl-fwht-build
export DL_FWHT_BUILD_DIR=/tmp/dl-fwht-build
```

Specify CryptoMiniSat through `cryptominisat5` on PATH, the `CRYPTOMINISAT_BIN` environment variable, or `--cms-bin`. OCP defaults to the bundled `OCP/` directory; `OCP_ROOT` or the search option `--ocp-root` selects another location. Experiment output directories are set with `--output-dir` or `--output-root`, depending on the entry point.

Each `__init__.py` declares a Python package or exports its public interface; it is not a standalone experiment entry point.

## Cipher: implementations and representations

| Program | Purpose |
| --- | --- |
| `Cipher/registry.py` | Cipher names, block sizes, key sizes, and round metadata. `get_cipher_spec`, `get_reference`, and `get_gpu_primitive` provide common selection interfaces. |
| `Cipher/tensor_ops.py` | Tensor word operations, logical shifts, rotations, and projection helpers. |
| `reference.py` in each of the six family directories | CPU round functions, actual master-key schedules, and segment encryption. KNOT is an unkeyed permutation. |
| `gpu.py` in each family directory | Parameterized tensor encryption, difference injection, output packing, and bit-order conversion. |
| `native/validation_traits.hpp` in each family directory | Round functions, key schedules, and word-width adapters used by the common C++ validator. |
| `Cipher/Present/adapter.py` | PRESENT metadata, S-box cells, and P-layer mappings for the shared OCP search. |
| `Cipher/Present/triton.py` | Triton kernels for batched PRESENT encryption. |
| `Cipher/Present/native/present_fast.hpp` | Native PRESENT encryption. |
| `Cipher/KNOT/adapter.py` | KNOT state mappings, row rotations, and nonce/rate endpoint constraints. |
| `Cipher/KNOT/native/knot256_native.hpp` | Native KNOT-256 round functions, constants, and state representation. |

| Family | Names and supported key sizes |
| --- | --- |
| Speck | `speck32/64`, `speck48/{72,96}`, `speck64/{96,128}`, `speck96/{96,144}`, `speck128/{128,192,256}`. |
| Simon | The same block/key-size combinations as SPECK, with the `simon` prefix. |
| Simeck | `simeck32/64`, `simeck48/96`, `simeck64/128`. |
| Cham | `cham64/128`, `cham128/128`. |
| Present | `present80`, `present128`. |
| KNOT | `knot256`; no master key, key size `0`. The version selector also accepts `256` for compatibility. |

A typical command-line selection is `--cipher speck128 --version 256`, specifying the block and key sizes. Default key sizes are defined in `registry.py`. `available_ciphers()` lists 18 default names; `available_variants()` lists 28 concrete configurations.

Adapters convert differences and masks between integers and paper-bit lists. SPECK/SIMON/SIMECK number bits from the least significant bit of the right word, followed by the left word. CHAM starts at v0 and uses least-significant-bit-first order within each word. PRESENT uses most-significant-bit-first order. KNOT starts at row0, proceeds through row3, and uses most-significant-bit-first order within each row. Aggregation endpoints use `LEFT_HEX,RIGHT_HEX`. The main search accepts a full-state integer for `--fixed-delta`; the validation CLI also provides `--delta-bits`, `--input-mask-bits`, and `--mask-bits`.

## Framework/search: three-segment search for a fixed round split

| Program | Purpose |
| --- | --- |
| `search/__main__.py` | Main search CLI and argument parsing. |
| `search/engine.py` | E1 candidates, Em scans, E2 scheduling, the candidate leaderboard, dynamic pruning, independent validation, and recovery. |
| `search/adapter.py` | Adapter selection for all six families, OCP calls, endpoint constraints, trail replay, and common result checks. |
| `search/models.py` | `SearchConfig`, cipher metadata, panels, and spectrum result structures. The combined weight is `w_E1 + w_Em + 2*w_E2`. |

```bash
python -m Framework.search --cipher speck32 \
  --r1 0 --rm 2 --r2 0 --fixed-delta 0x1 \
  --backend cpu --panel-size 4 \
  --scout-samples-power 8 --eval-samples-power 10 \
  --parallel-workers 1 --output-dir /tmp/dl-search
```

`--r1/--rm/--r2` specify the lengths of E1, Em, and E2. Their starting rounds are `round_start`, `round_start+r1`, and `round_start+r1+rm`, respectively. Nonzero outer segments require CryptoMiniSat. The default enumerates single-bit middle-input differences; PRESENT/KNOT use a single active S-box cell. `--delta-source pre-linear-single-cell` maps these cell differences through the linear layer. `--fixed-delta` selects a fixed middle-input difference.

`--delta-candidates 0` retains every candidate that passes E1. Middle masks first pass the strict threshold `abs(rho)>rho_threshold` and the Hamming-weight limit. The search then keeps up to `--em-top-per-hw` masks per weight before applying the overall `--em-top-k` cap. `--leaderboard-size` sets the number of complete candidates retained; E2 bounds tighten dynamically once the leaderboard is full. `--e1-weight-bound`, `--e2-weight-bound`, and the optional `--total-weight-bound` specify explicit search bounds.

Enable independent fixed-key validation of Em with:

```text
--validation-enabled true --validation-num-keys 50
--validation-samples-power 34 --validation-backend cuda
```

The main search uses the mean absolute Em correlation to update its three-segment estimate with single-trail outer segments. Per-key E2 measurements and pairing are performed separately by `Experiments.validation`. Search outputs include `summary.json`, candidate leaderboards, and stage caches. `--resume true` checks the configuration and source identity before resuming unfinished work. Timeouts, resource limits, and replay failures are recorded as incomplete states.

## Framework/split_ladder.py: round-split exploration

`split_ladder.py` probes Em at four representative input coordinates, extends the middle-segment length, and selects candidate round splits. `em-only` evaluates only the middle segment; `quick-split` also searches E1/E2 frontiers. Both use the main search adapters and the same spectrum interface.

```bash
python -m Framework.split_ladder --ciphers speck32,simon32,simeck32 \
  --mode em-only --rm-values 2 --samples 256 --panel-samples 128 \
  --backend cpu --panel-size 4 --output-root /tmp/dl-splits
```

`--samples` and `--panel-samples` specify actual sample counts. `--max-rm-steps` limits middle-segment extension; `--max-e1-rounds`, `--max-e2-rounds`, or `--max-side-rounds-from-total` bound the outer-segment search. CHAM, PRESENT, and KNOT use `--em-round-start-values` to specify absolute round offsets. SPECK/SIMON/SIMECK may reuse an Em spectrum at a standard offset during preliminary split exploration; the main search places each segment at its actual offset.

Outputs include per-cipher candidates, frontiers, execution states, and the combined `all_recommended_splits.json`. Results distinguish replay-verified feasible trails from proven optima. Uncertified frontiers are marked `uncertified_frontier`; a timeout does not prove that a longer trail is absent.

## Framework/fwht and common: middle-segment spectrum evaluation

| Program | Purpose |
| --- | --- |
| `fwht/scan.py` | Shared CPU/CUDA scan entry point and result conversion for main search and round-split exploration. |
| `fwht/candidates.py` | HW1 coordinate panels, spectrum conversion, strict threshold and weight filtering, deduplication of actual boundary masks, and two-stage candidate quotas. |
| `fwht/cpu.py` | Small-panel sampling, projected histograms, and integer FWHT using NumPy. |
| `fwht/cuda.py` | GPU HW1 ranking, small-panel spectrum evaluation, four-GPU scheduling, and caching. |
| `fwht/distributed.py` | Four-GPU histogram sampling, communication, local/cross-device FWHT, and candidate extraction by Hamming weight. |
| `fwht/kernels.py` | Triton kernels for bucketing, prefix sums, sample routing, and histogram accumulation. |
| `common/gpu.py` | Tensor cipher configuration, key-word widths, device selection, random-word generation, and integer FWHT. |
| `common/path_utils.py` | Bit/word conversion, weight conversion, atomic JSON/JSONL I/O, and fingerprints of source code and constraint templates. |

With CUDA, `k<n` uses independent sample batches for HW1 coordinate selection and spectrum evaluation; `k=n` directly evaluates the full spectrum. `--scout-samples-power` and `--eval-samples-power` specify the sample-count exponents for the two stages. The CPU backend is intended for small reference checks, supports panel dimensions 1-24, and reuses a scout prefix of the evaluation samples. Its sampling protocol therefore differs from the two independent CUDA batches.

`--backend cuda --panel-size 32 --gpus 0,1,2,3` uses four GPUs, each holding `2^30` histogram/spectrum entries. The base array requires about 4 GiB per GPU for `N<=2^30`, or 8 GiB for `N=2^34`, plus space for encryption, routing, and FWHT. Smaller panels use the small-panel backend: each GPU holds a full histogram, and the integer FWHT arrays on the reduction GPU peak at approximately `16*2^k` bytes, excluding sampling buffers.

`--gpus` and `DL_FWHT_DISABLED_GPUS` use logical device indices after `CUDA_VISIBLE_DEVICES` remapping. A 32-bit panel requires four available GPUs. If fewer than four are specified, the backend fills the selection from visible, non-disabled devices. Main-search and split-exploration caches check the configuration and source identity; internal CUDA caches additionally check the actual devices, batch size, and candidate-file SHA256.

The common paper-scale parameters map to the options below. These are not the budgets used by the small examples above.

| Parameter | 32-bit blocks | Larger states |
| --- | --- | --- |
| Total discovery samples per stage | `--scout-samples-power 30 --eval-samples-power 30` | Set both exponents to `34`. |
| Filtering threshold | Set `--rho-threshold` to the decimal value of `2^-12.29`. | Use the decimal value of `2^-14.29`. |
| Panel cap | `--panel-size 32` | `--panel-size 32` |
| Per-weight / overall quota | `--em-top-per-hw 1000 --em-top-k 2000` | Same. |
| Complete candidates retained | `--leaderboard-size 10` | Same. |
| Fixed-key validation | `--validation-num-keys 50 --validation-samples-power 30` | Set the sample-count exponent to `34`. |

For instances requiring low-weight suffix masks, explicitly set `--mask-hw-max 6`. To compare panel dimensions, run different `--panel-size` values while keeping the round split and other parameters fixed.

## Framework/sat: outer-segment modeling and solving

| Program | Purpose |
| --- | --- |
| `search.py` | E1/E2 subprocess scheduling, fixed endpoints, resource limits, and status normalization. |
| `differential_worker.py` | SAT search for a differential outer segment with fixed endpoints. |
| `linear_worker.py` | SAT search for a linear outer segment with a fixed input mask. |
| `ocp_factory.py` | OCP primitives and round structures for all six families, including the KNOT S-box wrapper. |
| `solver.py` | CryptoMiniSat execution, temporary files, and SAT result parsing. |
| `replay.py` | Differential trail replay and bit-representation conversion. |
| `trail_replay.py` | Independent differential/linear transition-weight evaluation for modular addition and SIMON-like functions. |

Outer-segment results use `status/results`. Trails contain `weight`, `states`, `round_weights`, `path_verified`, and `optimum_proven`. PRESENT/KNOT replay uses the DDT/LAT of the actual S-box. For SIMON/SIMECK, the bitwise OCP objective can differ from the exact whole-round weight. Replay verifies a feasible trail but does not establish global optimality under the whole-round metric; such results are not marked as proven optima.

For KNOT, `--knot-mode aead` restricts input differences to the nonce and output masks to the rate; `permutation` leaves the permutation endpoints unrestricted. These constraints do not change the full-state sampling distribution used by validation.

## Framework/aggregation: fixed-endpoint trail aggregation

| Program | Purpose |
| --- | --- |
| `run.py` | Aggregation CLI, cipher capability checks, parameter normalization, run manifests, and backend selection. |
| `models.py` | Differential/linear SAT models, exact-weight constraints, projected states, replay, and solver construction. |
| `allsat.py` | Projected enumeration partitioned by weight and intermediate state, probing, timeout-driven subdivision, recovery, and contribution accumulation. |
| `pipeline.py` | Staged SAT enumeration, trail replay, window initialization, same-width replacement, and final union. |
| `window.py` | Window bit order, replay caches, inside/outside classification, replacement candidates, and native DP calls. |
| `window_dp.c` | Fixed-endpoint window DP for differential probabilities or sums of squared linear trail correlations. |
| `markov_sampler.c` | Reverse differential Markov sampling and per-round active-bit ranking. |
| `simon_transition.h` | Parameterized affine differential transitions and probabilities for SIMON/SIMECK. |

```bash
python -m Framework.aggregation.run --cipher simon32 --trail-kind differential \
  --rounds 1 --input 0x1,0x0 --output 0x4,0x1 \
  --method sat-guided-window --weight-range 2-3 --best-trail-weight 2 \
  --window-size 2 --samples-power 6 --workers 1 --shard-bits 1 \
  --max-shard-bits 2 --memory-limit-gb 1 --output-dir /tmp/dl-window
```

`--method sat` enumerates trails only; `sat-guided-window` adds dynamic-window DP. SIMON/SIMECK support both methods for differential aggregation. SPECK supports fixed-endpoint linear SAT enumeration. CHAM/PRESENT/KNOT have no fixed-endpoint AllSAT aggregation backend. Use `--list-ciphers` to view the capability table.

The SIMON/SIMECK OCP linear model does not provide exact whole-round weights, so linear AllSAT and the dependent window pipeline reject these families. E2 in the paper uses fixed-master-key measurements through `Experiments.validation --measurement linear`. The standalone linear DP in `window_dp.c` computes the squared-correlation sum over the trails admitted by a supplied window and can be used to evaluate and check that window.

`--weight-range` specifies integer trail-weight layers. `--best-trail-weight` specifies a feasible single-trail weight; differential windows protect the supports of trails in that layer. `--window-size`, `--top-families`, `--max-guided-iterations`, and `--min-guided-improvement` control window iteration. `--timeout`, probe budgets, and partition parameters control AllSAT. `--memory-limit-gb` limits window DP memory.

The window pipeline selects one final window's DP value and adds the contribution of enumerated trails outside that window. AllSAT blocks only state sequences, and counts from a subdivided parent are not added to its children's counts. The result is a lower bound contributed by a restricted trail set. `complete` means that the requested weight layers or stages finished; it does not mean that the entire hull was enumerated. Complete layers and partial counts are recorded separately. `--resume` reuses completed partitions and stages. Window recovery requires the same configuration; SAT-only recovery supports extending the upper weight bound and retrying unfinished partitions with a larger timeout.

A differential trail of weight `w` contributes `2^-w`; a linear trail of weight `w=-log2|c|` contributes `2^(-2w)`. Linear aggregation reports the ELP-model sum of squared trail correlations, not the signed linear hull for an individual fixed key. Window JSON contains `semantics`, `mass`, and `mass_weight`. Summaries record the window, stopping reason, weight-layer statuses, and `strict_union_mass_weight`. Zero mass is represented by a `null` weight.

## Experiments: fixed-key validation and program checks

| Program | Purpose |
| --- | --- |
| `validation.py` | Validation CLI/API, configuration checks, per-key statistics, ordinary linear suffix measurement, and pairing by master key. |
| `backends.py` | Master-key generation, state conversion, CPU/CUDA sampling, device scheduling, C++ compilation/execution, and result loading. |
| `native/validate.cpp` | Shared multithreaded C++ sampling and counting, master-key-list loading, and DL or ordinary linear parity evaluation. |
| `checks.py` | Check suites for cipher interfaces, main search and round splits, model constraints, validation statistics, and trail aggregation. |

### Em and E2 validation

```bash
python -m Experiments.validation --cipher speck32 \
  --rounds 3 --round-start 0 --delta 0x40 --mask 0x100 \
  --samples-power 8 --num-keys 2 --backend cpp --threads 2 \
  --output-dir /tmp/dl-validation

python -m Experiments.validation --cipher speck32 --measurement linear \
  --rounds 1 --round-start 3 --input-mask 0x100 --mask 0x100 \
  --samples-power 8 --num-keys 2 --backend cpp --threads 2 \
  --pair-with-em /tmp/dl-validation --output-dir /tmp/dl-e2-validation
```

The default `--measurement dl` estimates `mean((-1)^(Gamma·(E_K(x) xor E_K(x xor Delta))))`; `linear` estimates `mean((-1)^(Gamma_in·x xor Gamma_out·E_K(x)))`. Inputs are sampled uniformly over the full segment state. A zero-round linear suffix is treated as the identity.

`--seed` controls master keys; `--sample-seed` separately controls the input sample stream. Linear mode defaults to a different sampling seed from DL mode. `--keys-json` accepts lists of master-key words, such as `[["0x1","0x2","0x3","0x4"]]`. `--pair-with-em` loads the actual master keys from an Em output directory and checks the boundary mask, key contents, consecutive segment offsets, and independent sampling seeds. In a complete split, Em starts at `R1` and E2 at `R1+Rm`; pass these offsets explicitly with `--round-start`.

`config.json` records parameters and the sampling protocol. `summary.json` contains per-key signed correlations, Walsh sums, parity counts, master keys, mean magnitudes, and distribution statistics. `per_key_noise=1/sqrt(N)` is the noise scale. CPU, CUDA, and C++ share the master-key list but use different input streams, so their finite-sample counts need not match.

Pairing writes `paired_summary.json`. By default, it computes `mean_k(|r_hat_K|*q_hat_K²)` and also reports the two marginal means. Linear records additionally contain a noise-corrected square and the product of estimates from two sample halves. `--q2-estimator empirical|noise-corrected|split-half` selects the estimator used for pairing. The latter two estimates may be negative and are not clipped to zero. Multiplying the paired mean by the aggregated prefix contribution gives the three-segment composition estimate.

KNOT has no master key; `--num-keys` specifies independent sample batches. Full-state measurement differs from nonce-only AEAD initialization sampling, and endpoint constraints do not change that sampling distribution.

### Check entry points

```bash
python -m Experiments.checks
python -m Experiments.checks --suite all --cms-bin /path/to/cryptominisat5
python -m Experiments.checks --suite interfaces --device cuda:0
```

`interfaces` checks reference/tensor agreement for 28 cipher configurations, known-answer vectors, bit order, and FWHT. `framework` checks main search, round splits, fixed-key statistics, cache recovery, and S-box templates; providing `--cms-bin` also enables real E1/E2 solver checks. `aggregation` checks real AllSAT, independent short-round enumeration, window DP, union accounting, and recovery; it requires `pycryptosat` and a compiler. The default runs the first two suites; `all` runs all three. `--spn-only --cms-bin ...` selects the PRESENT/KNOT checks. Generated files use temporary directories.

## OCP: modeling dependencies

`OCP/` provides cipher primitives, variables, operators, SAT constraint generation, solver interfaces, and four PRESENT/KNOT S-box constraint templates for outer-segment search and linear aggregation. Per-file descriptions are in [OCP/README.md](OCP/README.md); the license is in [OCP/LICENSE](OCP/LICENSE).
