"""One CPU/CUDA middle-spectrum interface for search and round allocation."""
from pathlib import Path
from Framework.search.models import EmScanResult, Panel


def scan(adapter, delta, config, output_dir):
    if config.backend == 'cpu':
        from .cpu import scan as cpu_scan
        return cpu_scan(adapter, delta, config, output_dir)
    if config.backend != 'cuda':
        raise ValueError('backend must be cpu or cuda')
    from .cuda import run_cuda_em
    out = Path(output_dir) / 'em_cuda'
    out.mkdir(parents=True, exist_ok=True)
    class Cache:
        root = str(out)
    panel_seed = config.panel_seed if config.panel_seed is not None else (int(config.seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    if adapter.metadata.block_bits > config.panel_size and panel_seed == config.seed:
        raise ValueError('coordinate selection and spectrum evaluation require different seeds')
    result = run_cuda_em(adapter.paper_bits(delta), config.rm, Cache(), config.sample_count,
        config.seed, config.rho_threshold, adapter.gpu_name, config.gpus, config.cuda_batch_size,
        resume=config.resume, round_start=config.em_start, panel_size=config.panel_size,
        panel_samples=config.scout_count, panel_seed=panel_seed,
        per_hw_limit=config.em_top_per_hw, extraction_hw_min=config.mask_hw_min,
        extraction_hw_max=min(config.mask_hw_max, config.panel_size), version=config.version)
    summary = result['summary']
    panel = Panel('direct', tuple(map(int, summary['panel'])), 'top-bits', 'direct_top_bits')
    ranking = [{**row, 'bit': int(row.get('paper_bit', row.get('bit')))}
               for row in summary.get('singleton_ranking', [])]
    rows = []
    for rank, item in enumerate(result['candidates'], 1):
        coefficient = int(item['panel_mask'])
        bits = [bit for position, bit in enumerate(panel.bits) if (coefficient >> position) & 1]
        rows.append({'rank': rank, 'mask_index': coefficient,
                     'mask_hex': adapter.format_state(adapter.from_paper_bits(bits)),
                     'rho': float(item['rho_search']), 'Walsh': int(item['Walsh']),
                     'source': 'cuda_projected_fwht'})
    domains = {'direct': {'singletons': ranking, 'panels': [{'panel': panel, 'rows': rows}]}}
    full = adapter.metadata.block_bits == config.panel_size
    return EmScanResult(domains, config.sample_count, 1 if full else 2, full, True, 'cuda_projected_fwht')
