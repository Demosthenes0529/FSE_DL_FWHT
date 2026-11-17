"""Shared sampling/counting loop; cipher implementations come from Cipher.registry."""

import json
import csv
import hashlib
import subprocess
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from Cipher.registry import get_cipher_spec, get_gpu_primitive, get_reference

MASK64 = (1 << 64) - 1


def splitmix64(state):
    state = (int(state) + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def key_widths(config):
    if config.key_word_bits is not None:
        return tuple(config.key_word_bits)
    return (config.word_size,) * (config.key_size // config.word_size)


def key_words_for(seed, key_index, widths):
    state = (int(seed) ^ (0xD6E8FEB86659FD93 * (key_index + 1))) & MASK64
    words = []
    for width in widths:
        state, word = splitmix64(state)
        words.append(word & ((1 << width) - 1))
    return words


def stream_seed(seed, key_index):
    _, value = splitmix64(int(seed) ^ (0x94D049BB133111EB * (key_index + 1)))
    return value & ((1 << 63) - 1)


def _signed64(value):
    value = int(value) & MASK64
    return value if value < (1 << 63) else value - (1 << 64)


class ArrayRuntime:
    """Array allocation differs by backend; the experiment itself does not."""

    def __init__(self, backend, seed, gpu=None):
        self.cuda = backend == "cuda"
        if self.cuda:
            import torch
            self.torch = torch
            self.device = torch.device(f"cuda:{gpu}")
            torch.cuda.set_device(self.device)
            self.rng = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.rng = np.random.default_rng(seed)

    def word(self, count, width):
        if not self.cuda:
            return self.rng.integers(0, (1 << width) - 1, size=count,
                                     dtype=np.uint64, endpoint=True)
        torch = self.torch
        if width < 63:
            return torch.randint(0, 1 << width, (count,), generator=self.rng,
                                 device=self.device, dtype=torch.int64)
        low = torch.randint(0, 1 << 32, (count,), generator=self.rng,
                            device=self.device, dtype=torch.int64)
        high = torch.randint(0, 1 << (width - 32), (count,), generator=self.rng,
                             device=self.device, dtype=torch.int64)
        return low | (high << 32)

    def scalar(self, value):
        return _signed64(value) if self.cuda else np.uint64(int(value) & MASK64)

    def constant(self, count, value):
        if self.cuda:
            return self.torch.full((count,), _signed64(value), dtype=self.torch.int64,
                                   device=self.device)
        return np.full(count, value, dtype=np.uint64)

    def selected(self, state, mask):
        selected = state[0] & self.scalar(mask[0])
        for word, selected_mask in zip(state[1:], mask[1:]):
            selected = selected ^ (word & self.scalar(selected_mask))
        return selected

    def odd_count(self, parity):
        for shift in (32, 16, 8, 4, 2, 1):
            parity = parity ^ (parity >> shift)
        return int((parity & 1).sum().item())

    def synchronize(self):
        if self.cuda:
            self.torch.cuda.synchronize(self.device)


def worker(name, version, config, key_indices, difference, word_masks, master_keys, output_path, gpu=None):
    """Process complete fixed keys on one device, retaining exact parity counts."""
    primitive = get_gpu_primitive(name, version)
    encryptor = primitive if config.backend == "cuda" else get_reference(name, version)
    samples = 1 << config.samples_power
    records = []
    linear = config.measurement == "linear"
    for key_index in key_indices:
        runtime = ArrayRuntime(config.backend, stream_seed(config.sampling_seed, key_index), gpu)
        master_key = master_keys[key_index]
        ones = [0] * len(word_masks)
        half_ones = [0] * len(word_masks)
        started = time.perf_counter()
        offset = 0
        while offset < samples:
            count = min(config.batch_size, samples - offset)
            if linear and offset < samples // 2:
                count = min(count, samples // 2 - offset)
            state = tuple(runtime.word(count, primitive.config.word_size)
                          for _ in range(primitive.config.state_words))
            keys = tuple(runtime.constant(count, word) for word in master_key)
            if linear:
                output = (encryptor.encrypt_batch(state, keys, config.rounds, round_start=config.round_start)
                          if config.rounds else state)
                input_parity = runtime.selected(state, difference)
            else:
                pair_encrypt = getattr(encryptor, "encrypt_pair_difference_batch", None)
                if pair_encrypt is not None:
                    output = pair_encrypt(state, keys, difference, config.rounds,
                                          round_start=config.round_start)
                else:
                    paired = tuple(word ^ runtime.scalar(delta) for word, delta in zip(state, difference))
                    left = encryptor.encrypt_batch(state, keys, config.rounds, round_start=config.round_start)
                    right = encryptor.encrypt_batch(paired, keys, config.rounds, round_start=config.round_start)
                    output = tuple(a ^ b for a, b in zip(left, right))
                input_parity = 0
            for i, mask in enumerate(word_masks):
                odd = runtime.odd_count(runtime.selected(output, mask) ^ input_parity)
                ones[i] += odd
                if linear and offset < samples // 2:
                    half_ones[i] += odd
            offset += count
        runtime.synchronize()
        elapsed = time.perf_counter() - started
        records.append({
            "key_index": key_index, "samples": samples,
            "master_key_words_hex": [hex(word) for word in master_key],
            "stream_seed": stream_seed(config.sampling_seed, key_index),
            "elapsed_sec": elapsed, "gpu": gpu,
            "masks": [{"mask_index": i, "odd_parity_count": odd,
                       "Walsh": samples - 2 * odd,
                       "rho": (samples - 2 * odd) / samples,
                       "Walsh_first_half": samples // 2 - 2 * half_ones[i] if linear else None,
                       "Walsh_second_half": samples // 2 - 2 * (odd - half_ones[i]) if linear else None}
                      for i, odd in enumerate(ones)],
        })
    Path(output_path).write_text(json.dumps(records, indent=2) + "\n")


def run(adapter, input_value, masks, config, output_dir, master_keys):
    name = adapter.gpu_name
    version = adapter.metadata.key_bits or None
    primitive = get_gpu_primitive(name, version)
    word_mask = (1 << primitive.config.word_size) - 1
    def words(value):
        return tuple(int(word) & word_mask for word in
                     primitive.paper_bits_to_difference(adapter.paper_bits(value)))
    difference = words(input_value)
    word_masks = [words(mask) for mask in masks]
    if config.backend == "cpp":
        return run_native(primitive.config, difference, word_masks, config, output_dir, master_keys)
    if config.backend == "cuda":
        from Framework.common.gpu import resolve_gpu_ids
        devices = resolve_gpu_ids(config.gpus)
        if not devices:
            raise ValueError("no CUDA devices selected")
        context = mp.get_context("spawn")
        jobs, paths = [], []
        for rank, gpu in enumerate(devices):
            indices = list(range(rank, config.num_keys, len(devices)))
            if not indices:
                continue
            path = Path(output_dir) / f"device_{gpu}.json"
            job = context.Process(target=worker, args=(name, version, config, indices,
                                  difference, word_masks, master_keys, str(path), gpu))
            job.start()
            jobs.append(job)
            paths.append(path)
        for job in jobs:
            job.join()
        if any(job.exitcode for job in jobs):
            raise RuntimeError(f"CUDA validator worker failed: {[job.exitcode for job in jobs]}")
    else:
        paths = [Path(output_dir) / "observations.json"]
        worker(name, version, config, range(config.num_keys), difference, word_masks, master_keys, str(paths[0]))
    grouped = {index: [] for index in range(len(masks))}
    for path in paths:
        for key in json.loads(path.read_text()):
            for row in key["masks"]:
                grouped[row["mask_index"]].append({
                    "key_index": key["key_index"], "rho": row["rho"],
                    "Walsh": row["Walsh"], "odd_parity_count": row["odd_parity_count"],
                    "master_key_words_hex": key["master_key_words_hex"],
                    "stream_seed": key["stream_seed"],
                    "Walsh_first_half": row["Walsh_first_half"],
                    "Walsh_second_half": row["Walsh_second_half"],
                })
    return grouped


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"speck": "Speck", "simon": "Simon", "simeck": "Simeck",
            "cham": "Cham", "present": "Present", "knot": "KNOT"}


def ensure_binary(name, version, build_dir):
    spec = get_cipher_spec(name, version)
    header = ROOT / "Cipher" / FAMILIES[spec.family] / "native" / "validation_traits.hpp"
    source = Path(__file__).parent / "native" / "validate.cpp"
    dependencies = [source, *header.parent.glob("*.hpp")]
    digest = hashlib.sha256()
    for path in sorted(dependencies):
        digest.update(path.read_bytes())
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / f"validate_{spec.family}_{digest.hexdigest()[:12]}"
    if not binary.exists():
        subprocess.run(["g++", "-O3", "-std=c++17", "-pthread", "-I", str(ROOT),
                        f'-DCIPHER_HEADER="{header.relative_to(ROOT)}"',
                        str(source), "-o", str(binary)], check=True)
    return binary



def run_native(spec, difference, masks, config, output_dir, master_keys):
    output = Path(output_dir).resolve()
    mask_file, key_file, per_key = (output / "native_masks.txt", output / "native_keys.txt",
                                  output / "native_per_key.csv")
    for path, rows in ((mask_file, masks), (key_file, master_keys)):
        path.write_text("".join(",".join(hex(int(word)) for word in row) + "\n" for row in rows))
    binary = ensure_binary(spec.name, spec.key_size, output / "build")
    command = [str(binary), "--block", str(spec.block_size), "--key-bits", str(spec.key_size),
               "--round-start", str(config.round_start), "--rounds", str(config.rounds),
               "--keys", str(config.num_keys), "--threads", str(config.threads),
               "--samples", str(1 << config.samples_power), "--seed", str(config.sampling_seed),
               "--measurement", config.measurement, "--keys-file", str(key_file),
               "--difference", ",".join(hex(int(word)) for word in difference),
               "--masks", str(mask_file), "--output", str(per_key)]
    subprocess.run(command, check=True)
    grouped = {index: [] for index in range(len(masks))}
    with per_key.open() as handle:
        for row in csv.DictReader(handle):
            key_index = int(row["key_index"])
            grouped[int(row["mask_index"])].append({
                "key_index": key_index, "rho": float(row["rho"]),
                "Walsh": int(row["Walsh"]), "odd_parity_count": int(row["odd_parity_count"]),
                "master_key_words_hex": [hex(word) for word in master_keys[key_index]],
                "Walsh_first_half": int(row["Walsh_first_half"]) if row["Walsh_first_half"] else None,
                "Walsh_second_half": int(row["Walsh_second_half"]) if row["Walsh_second_half"] else None,
            })
    return grouped
