#!/usr/bin/env python3
import argparse, json, os, shutil, sys
from pathlib import Path

from .engine import run
from .models import SearchConfig
from .adapter import get_adapter


def _bool(text):
    if isinstance(text, bool): return text
    if str(text).lower() in {"1","true","yes","on"}: return True
    if str(text).lower() in {"0","false","no","off"}: return False
    raise argparse.ArgumentTypeError(text)


def _default_cms():
    configured = os.environ.get("CRYPTOMINISAT_BIN")
    if configured:
        return configured
    found = shutil.which("cryptominisat5")
    if found:
        return found
    return "cryptominisat5"


def parser():
    p=argparse.ArgumentParser(description="Shared E1 + Em/FWHT + E2 search")
    p.add_argument("--cipher",required=True);p.add_argument("--version",type=int,default=None,help="key size; inferred from cipher if omitted")
    p.add_argument("--r1",type=int,required=True);p.add_argument("--rm",type=int,required=True);p.add_argument("--r2",type=int,required=True)
    p.add_argument("--round-start",type=int,default=0);p.add_argument("--output-dir",required=True)
    p.add_argument("--delta-source",choices=("single-cell","pre-linear-single-cell"),default="single-cell")
    p.add_argument("--delta-cells",default="",
        help="comma-separated source-cell indices for single-cell delta proposers; empty means all cells")
    p.add_argument("--fixed-delta",action="append",default=[]);p.add_argument("--delta-candidates",type=int,default=0,help="E1 candidate limit, 0 keeps all")
    p.add_argument("--e1-weight-bound",type=int,default=64);p.add_argument("--e2-weight-bound",type=int,default=64)
    p.add_argument("--total-weight-bound",type=float,default=None,
        help="optional complete-path weight cap; default waits until the retained set is full")
    p.add_argument("--timeout-seconds",type=float,default=300)
    p.add_argument("--solver-threads",type=int,default=1);p.add_argument("--parallel-workers",type=int,default=48)
    p.add_argument("--scout-samples-power",type=int,default=20);p.add_argument("--eval-samples-power",type=int,default=20)
    p.add_argument("--validation-threads",type=int,default=48)
    p.add_argument("--panel-size",type=int,default=24)
    p.add_argument("--rho-threshold",type=float,default=0.0002);p.add_argument("--em-top-k",type=int,default=100)
    p.add_argument("--em-top-per-hw",type=int,default=200,
        help="per-HW candidate cap before the global --em-top-k cap (0 selects none)")
    p.add_argument("--mask-hw-min",type=int,default=1)
    p.add_argument("--mask-hw-max",type=int,default=32)
    p.add_argument("--leaderboard-size",type=int,default=10);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--cms-bin",default=_default_cms())
    p.add_argument("--ocp-root",default="")
    p.add_argument("--resume",type=_bool,default=True)
    p.add_argument("--backend",choices=("cpu","cuda"),default="cpu",
        help="Em sampling/histogram/FWHT backend; E1/E2 remain unchanged")
    p.add_argument("--gpus",default="0,1,2,3",
        help="CUDA device ids; k=32 requires four usable devices")
    p.add_argument("--cuda-batch-size",type=int,default=1<<20)
    p.add_argument("--knot-mode",choices=("aead","permutation"),default="aead",
        help="KNOT endpoint constraints: AEAD key/rate interface or unconstrained permutation")
    p.add_argument("--validation-enabled",type=_bool,default=False,
        help="run independent multi-key correlation validation (opt-in; default: false)")
    p.add_argument("--validation-samples-power",type=int,default=22)
    p.add_argument("--validation-num-keys",type=int,default=16)
    p.add_argument("--validation-top-n",type=int,default=10)
    p.add_argument("--validation-seed",type=int,default=2026181401)
    p.add_argument("--validation-backend",choices=("cpu","cpp","cuda"),default="cpu")
    p.add_argument("--validation-batch-size",type=int,default=65536)
    return p


def main(argv=None):
    a=parser().parse_args(argv); adapter=get_adapter(a.cipher,a.version)
    a.version=adapter.metadata.key_bits
    if a.backend == "cuda":
        from Framework.common.gpu import resolve_gpu_ids
        a.gpus=",".join(str(device) for device in
                         resolve_gpu_ids(a.gpus,required_count=4 if a.panel_size==32 else None))
    if a.scout_samples_power>a.eval_samples_power:raise ValueError("single-cache Em requires scout samples <= evaluation samples")
    delta_cells=tuple(int(x.strip(),0) for x in a.delta_cells.split(",") if x.strip())
    config=SearchConfig(cipher=a.cipher,version=a.version,r1=a.r1,rm=a.rm,r2=a.r2,round_start=a.round_start,output_dir=a.output_dir,delta_source=a.delta_source,delta_cells=delta_cells,fixed_deltas=[int(x,0) for x in a.fixed_delta],delta_candidates=a.delta_candidates,e1_weight_bound=a.e1_weight_bound,e2_weight_bound=a.e2_weight_bound,total_weight_bound=a.total_weight_bound,timeout_seconds=a.timeout_seconds,solver_threads=a.solver_threads,parallel_workers=a.parallel_workers,scout_samples_power=a.scout_samples_power,eval_samples_power=a.eval_samples_power,validation_threads=a.validation_threads,panel_size=a.panel_size,rho_threshold=a.rho_threshold,em_top_k=a.em_top_k,em_top_per_hw=a.em_top_per_hw,leaderboard_size=a.leaderboard_size,seed=a.seed,cms_bin=a.cms_bin,validation_enabled=a.validation_enabled,validation_samples_power=a.validation_samples_power,validation_num_keys=a.validation_num_keys,validation_top_n=a.validation_top_n,validation_seed=a.validation_seed,resume=a.resume,backend=a.backend,gpus=a.gpus,cuda_batch_size=a.cuda_batch_size,knot_mode=a.knot_mode,ocp_root=a.ocp_root)
    config.mask_hw_min=a.mask_hw_min
    config.mask_hw_max=a.mask_hw_max
    config.validation_backend=a.validation_backend
    config.validation_batch_size=a.validation_batch_size
    print(json.dumps(run(adapter,config),indent=2,sort_keys=True))


if __name__=="__main__":main()
