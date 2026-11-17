"""Shared cipher interface, OCP adapters and independent trail validation."""
from functools import lru_cache
import math

from Cipher.registry import available_ciphers, get_cipher_spec, get_reference
from Framework.sat.search import run_worker, search_e1_dc_to_delta, search_e2_lc
from Framework.sat.trail_replay import round_weights
from .models import CipherMetadata

def require_version(adapter, version):
    if get_cipher_spec(adapter.metadata.name, version).key_size != int(adapter.metadata.key_bits):
        raise ValueError(f"{adapter.metadata.variant} requires version {adapter.metadata.key_bits}")

def validate_payload(payload,stage,rounds):
    if not isinstance(payload,dict) or "status" not in payload or "results" not in payload:
        raise ValueError(f"{stage} backend returned a non-normalized payload")
    required={"states","round_weights","weight","path_verified","optimum_proven","backend"}
    required.add("delta_in" if stage=="E1" else "gamma_out")
    for row in payload["results"]:
        missing=required-set(row)
        if missing:raise ValueError(f"{stage} result missing fields: {sorted(missing)}")
        if not row["path_verified"]:raise ValueError(f"{stage} result failed independent replay")
        if len(row["states"])!=int(rounds)+1:raise ValueError(f"{stage} state count mismatch")
        if len(row["round_weights"])!=int(rounds):raise ValueError(f"{stage} round-weight count mismatch")
        if not math.isclose(sum(map(float,row["round_weights"])),float(row["weight"]),abs_tol=1e-9):
            raise ValueError(f"{stage} replay weight mismatch")
    return payload

class BaseAdapter:
    """Override representation and trail models; experiment flow is shared."""
    def _check_version(self, version):
        require_version(self, version)
    def _identity(self, value, stage):
        state=self.format_state(value)
        row={"weight":0.0,"states":[state],"round_weights":[],
             "path_verified":True,"optimum_proven":True,"backend":"identity"}
        row.update({"delta_in":state,"delta_m":state} if stage=="E1" else {"gamma_out":state})
        return {"status":"ok","results":[row],"runtime_seconds":0.0}
    def format_state(self, value):
        return f"0x{int(value):0{(self.metadata.block_bits+3)//4}x}"
    def paper_bits(self, value):
        return [i for i in range(self.metadata.block_bits) if (int(value) >> (self.metadata.block_bits-1-i)) & 1]
    def from_paper_bits(self, bits):
        value = 0
        for bit in bits:
            if not 0 <= int(bit) < self.metadata.block_bits: raise ValueError(f"invalid paper bit: {bit}")
            value |= 1 << (self.metadata.block_bits-1-int(bit))
        return value
    def encrypt_segment(self, plaintext, key, round_start, rounds, version=None):
        from Cipher.registry import get_reference
        return get_reference(self.metadata.name, version or self.metadata.key_bits).encrypt_segment(plaintext, key, round_start, rounds)
    def set_cell(self, value, cell, local_value):
        bits = self.metadata.cells[int(cell)]
        chosen = [bit for pos, bit in enumerate(bits) if (int(local_value) >> pos) & 1]
        return (int(value) & ~self.from_paper_bits(bits)) | self.from_paper_bits(chosen)
    def linear(self, value): return int(value)
    def inverse_linear(self, value): return int(value)
    def transpose_linear(self, value): return int(value)
    def scan_em(self, delta_m, config, output_dir):
        require_version(self, config.version)
        from Framework.fwht.scan import scan
        return scan(self, delta_m, config, output_dir)
    def validate_masks(self, delta_m, masks, config, output_dir):
        require_version(self, config.version)
        from Experiments.validation import validate
        return validate(self, delta_m, masks, config, output_dir)

class OCPAdapter(BaseAdapter):
    def __init__(self, name, version=None):
        self.cipher_spec = get_cipher_spec(name, version)
        self.reference = get_reference(name, version)
        self.word_size = self.cipher_spec.word_size
        self.gpu_name = self.cipher_spec.name
        self.version = self.cipher_spec.key_size
        self.metadata = CipherMetadata(
            name=self.cipher_spec.name, variant=f"{self.cipher_spec.family.upper()}-{self.cipher_spec.block_size}/{self.cipher_spec.key_size}",
            block_bits=self.cipher_spec.block_size, key_bits=self.cipher_spec.key_size,
            max_rounds=self.cipher_spec.max_rounds, cell_bits=1,
            cells=tuple((bit,) for bit in range(self.cipher_spec.block_size)))

    def paper_bits(self, value):
        return [bit for bit in range(self.metadata.block_bits) if (int(value) >> bit) & 1]

    def from_paper_bits(self, bits):
        return sum(1 << int(bit) for bit in set(bits))

    def _config(self, config, bound):
        return {"cipher": self.cipher_spec.name, "block_size": self.cipher_spec.block_size, "word_size": self.word_size,
                "version": self.cipher_spec.block_size, "cms_bin": config.cms_bin,
                "ocp_root": getattr(config, "ocp_root", ""),
                "threads": config.solver_threads, "timeout_sec": config.timeout_seconds,
                "seed": config.seed, "max_weight": int(bound),
                "e1_start": config.e1_start, "e2_start": config.e2_start}

    def _normalize(self, rows, stage, rounds, start):
        normalized = []
        for row in rows:
            if row.get("status") != "ok":
                continue
            trail = row["trail" if stage == "E1" else "linear_trail"]
            bit_states = trail["states_paper_bits"]
            integers = [self.from_paper_bits(bits) for bits in bit_states]
            mask = (1 << self.word_size) - 1
            words = [tuple((value >> (index * self.word_size)) & mask for index in range(self.cipher_spec.state_words)) for value in integers]
            if self.cipher_spec.state_words == 2:
                words = [tuple(reversed(state)) for state in words]
            try:
                weights = round_weights(self.cipher_spec.family, words, self.word_size,
                                        "differential" if stage == "E1" else "linear", start)
            except ValueError as error:
                return {"status": "replay_failed", "results": [], "error": str(error)}
            if len(weights) != rounds:
                return {"status": "replay_failed", "results": [], "error": "Incorrect number of round boundaries"}
            model_weights = trail.get("model_round_weights") or trail.get("round_weights") or []
            exact_model = self.cipher_spec.family in ("speck", "cham")
            same_weights = len(model_weights) == len(weights) and all(
                math.isclose(float(model), exact, abs_tol=1e-9)
                for model, exact in zip(model_weights, weights))
            value = {"weight": sum(weights), "states": [self.format_state(state) for state in integers],
                     "round_weights": weights, "path_verified": True,
                     "optimum_proven": bool(exact_model and same_weights and row.get("model_optimum_proven")),
                     "model_optimum_proven": bool(row.get("model_optimum_proven")),
                     "optimality_model": row.get("optimality_model"),
                     "model_weight": row["w_E1" if stage == "E1" else "w_E2"],
                     "backend": row["solver"]}
            if stage == "E1":
                value.update(delta_in=value["states"][0], delta_m=value["states"][-1])
            else:
                value["gamma_out"] = value["states"][-1]
            normalized.append(value)
        payload = {"status": "ok" if normalized else rows[0].get("status", "no_path") if rows else "no_path",
                   "results": normalized, "error": rows[0].get("warning") if rows else None,
                   "runtime_seconds": sum(float(row.get("runtime_sec", 0)) for row in rows)}
        if rows:
            for key in ("model_unsat_proven", "model_weight_bound", "optimality_model"):
                if key in rows[0]:
                    payload[key] = rows[0][key]
        return payload

    def search_e1(self, delta_m, config):
        self._check_version(config.version)
        if config.e1_weight_bound < 0:
            return {"status": "no_path", "results": [], "model_unsat_proven": True}
        if not config.r1:
            return self._identity(delta_m, "E1")
        rows = search_e1_dc_to_delta(self.paper_bits(delta_m), config.r1, 1, self._config(config, config.e1_weight_bound))
        return self._normalize(rows, "E1", config.r1, config.e1_start)

    def search_e2(self, gamma_m, bound, config):
        self._check_version(config.version)
        if bound < 0:
            return {"status": "no_path", "results": [], "model_unsat_proven": True}
        if not config.r2:
            return self._identity(gamma_m, "E2")
        # search_e2_lc uses a strict product budget; the adapter receives an
        # inclusive integer trail-weight bound from the shared engine.
        rows = search_e2_lc(self.paper_bits(gamma_m), config.r2, 2.0 * (int(bound) + 1), 0.0, 1,
                            self._config(config, bound))
        payload = self._normalize(rows, "E2", config.r2, config.e2_start)
        payload["results"] = [row for row in payload["results"] if row["weight"] <= bound]
        if not payload["results"] and payload["status"] == "ok":
            payload["status"] = "replay_bound_unresolved"
            payload["error"] = "The model optimum exceeds the budget after independent replay; no exact-metric UNSAT proof is available"
        return payload

    def e2_lower_bound(self, gamma_m, config):
        return 0.0

@lru_cache(maxsize=None)
def _transition_tables(sbox):
    size = len(sbox)
    ddt = tuple(tuple(sum(sbox[x] ^ sbox[x ^ a] == b for x in range(size))
                      for b in range(size)) for a in range(size))
    lat = tuple(tuple(sum(1 if bin((a & x) ^ (b & sbox[x])).count('1') % 2 == 0 else -1
                          for x in range(size)) for b in range(size)) for a in range(size))
    return ddt, lat

class SBoxOCPAdapter(BaseAdapter):
    """Cipher adapters provide only their S-box, cell layout and linear map."""

    def endpoint_constraints(self, stage, config):
        return {}

    def replay(self, states, stage):
        """Check each local transition without trusting the solver's weights."""
        if stage not in {"E1", "E2"}:
            raise ValueError(f"unknown trail stage {stage!r}")
        ddt, lat = _transition_tables(tuple(int(value) for value in self.sbox))
        table = ddt if stage == "E1" else lat
        weights = []
        for before, after in zip(states, states[1:]):
            target = self.inverse_linear(after) if stage == "E1" else self.transpose_linear(after)
            weight = 0.0
            for cell in range(len(self.metadata.cells)):
                count = abs(table[self.get_cell(before, cell)][self.get_cell(target, cell)])
                if not count:
                    raise ValueError(f"{self.metadata.variant} {stage}: invalid transition in round {len(weights)}, cell {cell}")
                weight -= math.log2(count / len(self.sbox))
            weights.append(weight)
        return weights

    def e2_lower_bound(self, gamma_m, config):
        if not int(config.r2):
            return 0.0
        _, lat = _transition_tables(tuple(int(value) for value in self.sbox))
        return sum(-math.log2(max(abs(value) for value in lat[self.get_cell(gamma_m, cell)]) / len(self.sbox))
                   for cell in range(len(self.metadata.cells)))

    def _check_endpoints(self, states, constraints):
        for name, state in (("input_zero_bits", states[0]), ("output_zero_bits", states[-1])):
            if set(self.paper_bits(state)).intersection(constraints.get(name, ())):
                raise ValueError(f"{name} constraint violated")
        if constraints.get("input_not_zero") and not states[0]:
            raise ValueError("nonzero input constraint violated")

    def _search(self, fixed, stage, bound, config):
        require_version(self, config.version)
        rounds = int(config.r1 if stage == "E1" else config.r2)
        start = int(config.e1_start if stage == "E1" else config.e2_start)
        fixed = int(fixed)
        if fixed < 0 or fixed >= 1 << self.metadata.block_bits:
            raise ValueError("trail endpoint is outside the cipher state")
        if rounds < 0 or start < 0 or rounds + start > self.metadata.max_rounds:
            raise ValueError("trail interval is outside the cipher round range")
        if float(bound) < 0:
            return {"status": "no_path", "results": [], "runtime_seconds": 0.0, "model_unsat_proven": True}
        constraints = self.endpoint_constraints(stage, config)
        if not rounds:
            try:
                self._check_endpoints([fixed], constraints)
            except ValueError:
                return {"status": "no_path", "results": [], "runtime_seconds": 0.0, "model_unsat_proven": True}
            return self._identity(fixed, stage)

        request = {
            "output_difference" if stage == "E1" else "input_mask": self.paper_bits(fixed),
            "r1" if stage == "E1" else "r2": rounds,
            "max_weight": math.floor(bound), "start_round": start, "top_k": 1,
            **constraints,
        }
        solver = {"cipher": self.gpu_name, "version": self.metadata.key_bits,
                  "block_size": self.metadata.block_bits, "word_size": 1,
                  "cms_bin": config.cms_bin, "ocp_root": config.ocp_root,
                  "threads": config.solver_threads, "timeout_sec": config.timeout_seconds,
                  "seed": config.seed}
        payload = run_worker("differential" if stage == "E1" else "linear", request, solver)
        result = {"status": payload.get("status", "solver_error"), "results": [],
                  "error": payload.get("error"), "runtime_seconds": payload.get("runtime_sec", 0.0)}
        for name in ("model_unsat_proven", "model_weight_bound", "optimality_model"):
            if name in payload:
                result[name] = payload[name]
        if result["status"] != "ok":
            return result
        try:
            for source in payload.get("results", []):
                states = [self.from_paper_bits(bits) for bits in source["states_paper_bits"]]
                if len(states) != rounds + 1:
                    raise ValueError("incorrect number of round boundaries")
                if states[-1 if stage == "E1" else 0] != fixed:
                    raise ValueError("fixed endpoint mismatch")
                self._check_endpoints(states, constraints)
                endpoints = (("delta_in_paper_bits", states[0]), ("delta_m_paper_bits", states[-1])) if stage == "E1" else (("gamma_out_paper_bits", states[-1]),)
                for field, state in endpoints:
                    if field in source and self.from_paper_bits(source[field]) != state:
                        raise ValueError(f"{field} differs from the returned trail")
                weights = self.replay(states, stage)
                model_weights = source["round_weights"]
                if len(model_weights) != rounds or any(not math.isclose(actual, float(model), abs_tol=1e-8)
                                                      for actual, model in zip(weights, model_weights)):
                    raise ValueError("OCP/replay round-weight mismatch")
                weight = sum(weights)
                if not math.isclose(weight, float(source["w_E1" if stage == "E1" else "w_E2"]), abs_tol=1e-8):
                    raise ValueError("OCP/replay total-weight mismatch")
                if weight > float(bound) + 1e-8:
                    raise ValueError("trail weight exceeds the requested bound")
                row = {"weight": weight, "states": [self.format_state(state) for state in states],
                       "round_weights": weights, "path_verified": True,
                       "optimum_proven": bool(source.get("model_optimum_proven")),
                       "model_optimum_proven": bool(source.get("model_optimum_proven")),
                       "optimality_model": source.get("optimality_model"), "backend": "ocp_sat_cryptominisat"}
                if stage == "E1":
                    row.update(delta_in=row["states"][0], delta_m=row["states"][-1])
                else:
                    row["gamma_out"] = row["states"][-1]
                result["results"].append(row)
            if not result["results"]:
                raise ValueError("solver reported success without a trail")
        except (KeyError, TypeError, ValueError) as error:
            result.update(status="replay_failed", results=[], error=str(error))
        return result

    def search_e1(self, delta_m, config):
        return self._search(delta_m, "E1", config.e1_weight_bound, config)

    def search_e2(self, gamma_m, bound, config):
        return self._search(gamma_m, "E2", bound, config)

@lru_cache(maxsize=None)
def get_adapter(name, version=None):
    """Use the same canonical cipher/key-size parser as CPU/GPU validation."""
    spec = get_cipher_spec(name, version)
    if spec.family == "present":
        from Cipher.Present.adapter import PRESENTAdapter
        return PRESENTAdapter(spec.key_size)
    if spec.family == "knot":
        from Cipher.KNOT.adapter import KNOT256Adapter
        return KNOT256Adapter()
    return OCPAdapter(spec.name, spec.key_size)
