"""Shared fixed-key DL/linear measurement, statistics, pairing and CLI."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from Cipher.registry import available_ciphers, get_cipher_spec
from Framework.common.path_utils import atomic_write_json


@dataclass(frozen=True)
class ValidationConfig:
    rounds: int
    samples_power: int = 20
    num_keys: int = 50
    round_start: int = 0
    seed: int = 987654321
    backend: str = "cpu"
    batch_size: int = 65536
    gpus: str = "0"
    threads: int = 1
    measurement: str = "dl"
    sample_seed: int | None = None

    @property
    def sampling_seed(self):
        if self.sample_seed is not None:
            return self.sample_seed
        # Preserve the original DL stream. E2 uses another input stream while
        # the master-key seed stays unchanged.
        return self.seed ^ (0xE2A55EED20261104 if self.measurement == "linear" else 0)

    @classmethod
    def from_search(cls, config):
        if isinstance(config, cls):
            return config
        return cls(rounds=config.rm, samples_power=config.validation_samples_power,
                   num_keys=config.validation_num_keys, round_start=config.em_start,
                   seed=config.validation_seed, backend=getattr(config, "validation_backend", "cpu"),
                   batch_size=getattr(config, "validation_batch_size", 65536),
                   gpus=getattr(config, "gpus", "0"), threads=getattr(config, "validation_threads", 1))

    def check(self):
        if self.rounds < 0 or self.round_start < 0:
            raise ValueError("rounds and round_start must be nonnegative")
        if not 0 <= self.samples_power <= 62 or (self.measurement == "linear" and self.samples_power < 1):
            raise ValueError("samples_power must be in 0..62 (1..62 for linear measurement)")
        if min(self.num_keys, self.batch_size, self.threads) < 1:
            raise ValueError("num_keys, batch_size and threads must be positive")
        if self.backend not in {"cpu", "cuda", "cpp"} or self.measurement not in {"dl", "linear"}:
            raise ValueError("invalid backend or measurement")
        if any(not 0 <= seed < 1 << 64 for seed in (self.seed, self.sampling_seed)):
            raise ValueError("seeds must be unsigned 64-bit integers")


def summarize(adapter, masks, by_key, samples_per_key, measurement="dl"):
    results = []
    # Version selectors may describe a state size; registry metadata gives
    # the actual key size and distinguishes an unkeyed permutation.
    unkeyed = get_cipher_spec(adapter.gpu_name, adapter.metadata.key_bits).key_size == 0
    for index, mask in enumerate(masks):
        rows = sorted(by_key[index], key=lambda row: int(row["key_index"]))
        values = sorted(abs(float(row["rho"])) for row in rows)
        mean = sum(values) / len(values)
        per_key = [dict(row, abs_rho=abs(float(row["rho"]))) for row in rows]
        result = {
            "mask": adapter.format_state(mask), "measurement": measurement, "rho_valid": mean,
            "mean_abs_rho": mean, "median_abs_rho": (values[(len(values)-1)//2] + values[len(values)//2]) / 2,
            "mean_signed_rho": sum(float(row["rho"]) for row in rows) / len(rows),
            "std_abs_rho": math.sqrt(sum((value-mean)**2 for value in values) / len(values)),
            "min_abs_rho": min(values), "max_abs_rho": max(values),
            "samples_per_key": samples_per_key, "num_keys": 0 if unkeyed else len(rows),
            "num_independent_batches": len(rows) if unkeyed else None, "unkeyed": unkeyed,
            "replicate_semantics": "independent sample batches" if unkeyed else "fixed master keys",
            "per_key_noise": 1.0 / math.sqrt(samples_per_key), "per_key": per_key,
        }
        if measurement == "linear":
            for row in per_key:
                squared = row["rho"] ** 2
                row["q_squared"] = squared
                row["q_squared_noise_corrected"] = (samples_per_key * squared - 1) / (samples_per_key - 1)
                row["q_squared_split_half"] = (row["Walsh_first_half"] * row["Walsh_second_half"] /
                                                (samples_per_key // 2) ** 2)
            for key in ("q_squared", "q_squared_noise_corrected", "q_squared_split_half"):
                result["mean_" + key] = sum(row[key] for row in per_key) / len(rows)
        results.append(result)
    return results


def validate(adapter, delta, masks, config, output_dir, *, input_mask=None, master_keys=None):
    """Measure DL pairs, or a linear suffix with ``delta=None, input_mask=...``.

    Samples are uniform over the whole segment state. KNOT repetitions are
    unkeyed batches, not nonce-only AEAD experiments. A zero-round linear
    suffix is the identity, including for PRESENT (no extra whitening).
    """
    from Experiments.backends import key_widths, key_words_for, run
    config = ValidationConfig.from_search(config)
    config.check()
    spec = get_cipher_spec(adapter.gpu_name, adapter.metadata.key_bits)
    if config.round_start + config.rounds > spec.max_rounds:
        raise ValueError("segment exceeds the cipher's supported rounds")
    if config.measurement == "linear":
        if delta is not None or input_mask is None:
            raise ValueError("linear measurement requires input_mask and delta=None")
        input_value = input_mask
    else:
        if delta is None or input_mask is not None:
            raise ValueError("DL measurement requires delta, not an input mask")
        input_value = delta
    masks = list(masks)
    if not masks or any(int(value) < 0 or int(value) >= 1 << spec.block_size for value in [input_value, *masks]):
        raise ValueError("at least one valid output mask and an in-range input difference/mask are required")
    supplied_keys = master_keys is not None
    widths = key_widths(spec)
    if master_keys is None:
        master_keys = [key_words_for(config.seed, index, widths) for index in range(config.num_keys)]
    else:
        master_keys = [[int(word, 0) if isinstance(word, str) else int(word) for word in key] for key in master_keys]
    if len(master_keys) != config.num_keys or any(len(key) != len(widths) or
            any(not 0 <= word < 1 << width for word, width in zip(key, widths)) for key in master_keys):
        raise ValueError("master keys must match num_keys, word count and word widths")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped = run(adapter, input_value, masks, config, output, master_keys)
    samples = 1 << config.samples_power
    if set(grouped) != set(range(len(masks))):
        raise ValueError("validation returned missing or extra masks")
    for rows in grouped.values():
        if sorted(row["key_index"] for row in rows) != list(range(config.num_keys)):
            raise ValueError("validation returned incomplete or duplicate per-key observations")
        for row in rows:
            if not 0 <= row["odd_parity_count"] <= samples or row["Walsh"] != samples - 2 * row["odd_parity_count"]:
                raise ValueError("inconsistent parity count")
            if row["rho"] != row["Walsh"] / samples:
                raise ValueError("inconsistent correlation")
    results = summarize(adapter, masks, grouped, samples, config.measurement)
    metadata = {
        "cipher": spec.name, "key_bits": spec.key_size, "config": asdict(config),
        "delta" if config.measurement == "dl" else "input_mask": adapter.format_state(input_value),
        "sampling": "uniform full-state inputs", "sample_seed": config.sampling_seed,
        "plaintext_generator": {"cpu": "NumPy default_rng", "cuda": "PyTorch CUDA generator",
                                "cpp": "counter-based SplitMix64"}[config.backend],
        "master_key_generator": "supplied words" if supplied_keys else "SplitMix64; shared across backends",
        "aggregation": "mean_k(abs(rho_k))" if config.measurement == "dl" else "per-key signed q and squared estimators",
    }
    atomic_write_json(output / "config.json", metadata)
    atomic_write_json(output / "summary.json", results)
    return results


def paired_summary(em, suffix, estimator="empirical"):
    """Pair actual master-key words; never replace a mean of products by two means."""
    fields = {"empirical": "q_squared", "noise-corrected": "q_squared_noise_corrected",
              "split-half": "q_squared_split_half"}
    if estimator not in fields or em.get("unkeyed") or suffix.get("unkeyed"):
        raise ValueError("paired key statistics require a keyed cipher and a supported estimator")
    if em.get("measurement", "dl") != "dl" or suffix.get("measurement") != "linear":
        raise ValueError("pairing requires an Em DL result and an E2 linear result")
    left = {row["key_index"]: row for row in em["per_key"]}
    right = {row["key_index"]: row for row in suffix["per_key"]}
    if not left or left.keys() != right.keys() or len(left) != len(em["per_key"]) or len(right) != len(suffix["per_key"]):
        raise ValueError("paired observations must have identical, unique key indices")
    rows = []
    for index in sorted(left):
        a, b = left[index], right[index]
        keys = [[int(word, 0) for word in record["master_key_words_hex"]] for record in (a, b)]
        if not keys[0] or keys[0] != keys[1]:
            raise ValueError(f"master-key mismatch at index {index}")
        r, q2 = abs(a["rho"]), b[fields[estimator]]
        rows.append({"key_index": index, "master_key_words_hex": a["master_key_words_hex"],
                     "abs_r": r, "q_squared": q2, "abs_r_times_q_squared": r*q2})
    mean_r, mean_q2 = (sum(row[key] for row in rows)/len(rows) for key in ("abs_r", "q_squared"))
    return {"q_squared_estimator": estimator, "num_keys": len(rows), "per_key": rows,
            "mean_abs_r": mean_r, "mean_q_squared": mean_q2,
            "mean_paired_abs_r_times_q_squared": sum(row["abs_r_times_q_squared"] for row in rows)/len(rows),
            "product_of_separate_means": mean_r*mean_q2,
            "interpretation": "measured segment factors; not a direct full-distinguisher correlation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cipher", required=True, help=", ".join(available_ciphers()))
    parser.add_argument("--version", type=int)
    parser.add_argument("--measurement", choices=("dl", "linear"), default="dl")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-start", type=int, default=0)
    inputs = parser.add_mutually_exclusive_group(required=True)
    for name in ("delta", "delta-bits", "input-mask", "input-mask-bits"):
        inputs.add_argument("--"+name)
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument("--mask", action="append", help="output mask; repeat for multiple masks")
    masks.add_argument("--masks-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cpp", "cuda"), default="cpu")
    parser.add_argument("--samples-power", type=int, default=20)
    parser.add_argument("--num-keys", type=int, default=50, help="fixed keys, or independent KNOT batches")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=lambda text: int(text, 0), default=987654321, help="master-key seed")
    parser.add_argument("--sample-seed", type=lambda text: int(text, 0), help="independent plaintext seed")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--keys-json", type=Path, help="list of master-key word lists")
    source.add_argument("--pair-with-em", type=Path, help="Em output directory; reuse its exact keys and pair E2")
    parser.add_argument("--q2-estimator", choices=("empirical", "noise-corrected", "split-half"), default="empirical")
    args = parser.parse_args(argv)
    from Framework.search.adapter import get_adapter
    try:
        adapter = get_adapter(args.cipher, args.version)
        if (args.measurement == "dl") != (args.delta is not None or args.delta_bits is not None):
            raise ValueError("dl uses --delta/--delta-bits; linear uses --input-mask/--input-mask-bits")
        if args.measurement != "linear" and (args.pair_with_em or args.q2_estimator != "empirical"):
            raise ValueError("E2 pairing and squared estimators require --measurement linear")
        def decode(record):
            if isinstance(record, dict):
                if "paper_bits" in record:
                    value = adapter.from_paper_bits(record["paper_bits"])
                    if "mask" in record and value != decode(record["mask"]):
                        raise ValueError("mask and paper_bits disagree")
                    return value
                return decode(record["mask"])
            return int(record, 0) if isinstance(record, str) else int(record)
        text = args.delta if args.measurement == "dl" else args.input_mask
        bits = args.delta_bits if args.measurement == "dl" else args.input_mask_bits
        input_value = int(text, 0) if text is not None else adapter.from_paper_bits(
            [int(bit) for bit in bits.split(",") if bit.strip()])
        records = args.mask if args.mask is not None else json.loads(args.masks_json.read_text())
        if isinstance(records, dict):
            records = records.get("unique_masks", records.get("results", records.get("masks")))
        selected = [decode(record) for record in records]
        config = ValidationConfig(args.rounds, args.samples_power, args.num_keys, args.round_start,
                                  args.seed, args.backend, args.batch_size, args.gpus, args.threads,
                                  args.measurement, args.sample_seed)
        master_keys = json.loads(args.keys_json.read_text()) if args.keys_json else None
        em, em_meta = None, None
        if args.pair_with_em:
            if args.output_dir.resolve() == args.pair_with_em.resolve():
                raise ValueError("E2 output must differ from its Em input directory")
            em_meta = json.loads((args.pair_with_em / "config.json").read_text())
            em_rows = json.loads((args.pair_with_em / "summary.json").read_text())
            # Stored states are hexadecimal; PRESENT/KNOT use fixed-width
            # strings without an 0x prefix, unlike the two-word adapters.
            em = next((row for row in em_rows if int(row["mask"], 16) == input_value), None)
            spec = get_cipher_spec(adapter.gpu_name, adapter.metadata.key_bits)
            if em is None or em_meta["cipher"] != spec.name or em_meta.get("key_bits", spec.key_size) != spec.key_size:
                raise ValueError("Em cipher/key size or boundary mask does not match E2")
            if em.get("unkeyed") or em_meta["config"].get("measurement", "dl") != "dl":
                raise ValueError("Em must be a fixed-key DL measurement")
            if config.round_start != em_meta["config"]["round_start"] + em_meta["config"]["rounds"]:
                raise ValueError("E2 round_start must immediately follow the recorded Em segment")
            if config.sampling_seed == em_meta.get("sample_seed", em_meta["config"]["seed"]):
                raise ValueError("E2 must use an independent sample seed")
            master_keys = [row["master_key_words_hex"] for row in sorted(em["per_key"], key=lambda row: row["key_index"])]
        results = validate(adapter, input_value if config.measurement == "dl" else None, selected,
                           config, args.output_dir, input_mask=input_value if config.measurement == "linear" else None,
                           master_keys=master_keys)
        if em is not None:
            paired = [dict(paired_summary(em, result, args.q2_estimator),
                           input_mask=adapter.format_state(input_value), output_mask=result["mask"],
                           em_round_start=em_meta["config"]["round_start"], em_rounds=em_meta["config"]["rounds"],
                           e2_round_start=config.round_start, e2_rounds=config.rounds,
                           em_samples_per_key=em["samples_per_key"], e2_samples_per_key=1 << config.samples_power)
                      for result in results]
            atomic_write_json(args.output_dir / "paired_summary.json", paired)
        print(json.dumps(results, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
