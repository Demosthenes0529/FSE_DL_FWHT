from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CipherMetadata:
    name: str
    variant: str
    block_bits: int
    key_bits: int
    max_rounds: int
    cell_bits: int
    cells: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class Panel:
    domain: str
    bits: Tuple[int, ...]
    method: str
    panel_id: str


@dataclass
class EmScanResult:
    """Spectrum observations and sampling metadata returned by the shared scanner."""
    domains: Dict[str, Dict[str, Any]]
    samples_generated: int
    data_generation_passes: int
    cache_reused_for_fwht: bool
    domains_derived_from_same_ciphertexts: bool
    backend: str


@dataclass
class SearchConfig:
    cipher: str
    version: int
    r1: int
    rm: int
    r2: int
    round_start: int
    output_dir: str
    delta_source: str = "single-cell"
    delta_cells: Tuple[int, ...] = ()
    fixed_deltas: List[int] = field(default_factory=list)
    delta_candidates: int = 0
    e1_weight_bound: int = 64
    e2_weight_bound: int = 64
    total_weight_bound: Optional[float] = None
    timeout_seconds: float = 300.0
    solver_threads: int = 1
    parallel_workers: int = 48
    scout_samples_power: int = 20
    eval_samples_power: int = 20
    validation_threads: int = 48
    panel_size: int = 24
    rho_threshold: float = 0.0002
    em_top_k: int = 100
    em_top_per_hw: int = 200
    leaderboard_size: int = 10
    seed: int = 42
    cms_bin: str = ""
    validation_enabled: bool = False
    validation_samples_power: int = 22
    validation_num_keys: int = 16
    validation_top_n: int = 10
    validation_seed: int = 2026181401
    validation_backend: str = "cpu"
    validation_batch_size: int = 65536
    resume: bool = True
    backend: str = "cpu"
    gpus: str = "0,1,2,3"
    cuda_batch_size: int = 1 << 20
    # KNOT-256 uses one permutation primitive for two experiment interfaces.
    # Other ciphers ignore this field.
    knot_mode: str = "aead"
    ocp_root: str = ""
    eval_samples: Optional[int] = None
    scout_samples: Optional[int] = None
    panel_seed: Optional[int] = None
    mask_hw_min: int = 1
    mask_hw_max: int = 32

    @property
    def sample_count(self):
        return int(self.eval_samples if self.eval_samples is not None else 1 << self.eval_samples_power)

    @property
    def scout_count(self):
        return int(self.scout_samples if self.scout_samples is not None else 1 << self.scout_samples_power)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def e1_start(self) -> int: return int(self.round_start)
    @property
    def em_start(self) -> int: return int(self.round_start+self.r1)
    @property
    def e2_start(self) -> int: return int(self.round_start+self.r1+self.rm)
    @property
    def end_round(self) -> int: return int(self.round_start+self.r1+self.rm+self.r2)
    @property
    def em_last_round(self) -> int: return int(self.em_start+self.rm-1)


def compose_weight(w_e1: float, w_em: float, w_e2: float) -> float:
    return float(w_e1) + float(w_em) + 2.0 * float(w_e2)
