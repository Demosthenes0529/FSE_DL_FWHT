"""Paper HW1 coordinate selection and one shared middle-candidate schema."""
import math
from Framework.search.models import Panel


def build_panel(singletons, panel_size):
    ranked = sorted(singletons, key=lambda row: (-abs(float(row.get('abs_rho', row.get('rho', 0.0)))), int(row['bit'])))
    bits = tuple(int(row['bit']) for row in ranked[:int(panel_size)])
    return Panel('direct', bits, 'top-bits', 'direct_top_bits')


def row_to_candidate(adapter, row, panel):
    gamma = int(row['mask_hex'], 16)
    rho = float(row['rho'])
    return {'gamma_m': adapter.format_state(gamma),
            'gamma_m_paper_bits': adapter.paper_bits(gamma),
            'rhoEm_search': rho, 'wEm_search': -math.log2(abs(rho)) if rho else math.inf,
            'source': row.get('source', 'projected_fwht'),
            'candidate_rank_by_abs_rho': int(row.get('rank', 0) or 0),
            'panel_id': panel.panel_id, 'panel_bits': list(panel.bits),
            'panel_coefficient': int(row['mask_index'])}


def select_candidates(rows, *, adapter, rho_threshold, per_hw_limit, global_limit,
                      mask_hw_min=1, mask_hw_max=None):
    """Strict threshold, allowed HWs, deduplication, then per-HW and total caps."""
    per_hw_limit, global_limit = int(per_hw_limit), int(global_limit)
    if min(per_hw_limit, global_limit) < 0:
        raise ValueError('Em quotas must be nonnegative')
    maximum = adapter.metadata.block_bits if mask_hw_max is None else int(mask_hw_max)
    if not 1 <= int(mask_hw_min) <= maximum <= adapter.metadata.block_bits:
        raise ValueError('invalid candidate Hamming-weight interval')
    unique = {}
    for source in rows:
        gamma = int(source['gamma_m'], 16)
        rho = float(source['rhoEm_search'])
        if gamma == 0 or not math.isfinite(rho) or abs(rho) <= float(rho_threshold):
            continue
        bits = adapter.paper_bits(gamma)
        if not int(mask_hw_min) <= len(bits) <= maximum:
            continue
        previous = unique.get(gamma)
        if previous is None or abs(rho) > abs(float(previous['rhoEm_search'])):
            unique[gamma] = dict(source, hw=len(bits), gamma_m_paper_bits=bits)
    ranked = sorted(unique.values(), key=lambda row: (
        -abs(float(row['rhoEm_search'])), int(row['hw']), row['gamma_m_paper_bits']))
    kept, counts = [], {}
    for row in ranked:
        hw = row['hw']
        if counts.get(hw, 0) >= per_hw_limit:
            continue
        kept.append(row)
        counts[hw] = counts.get(hw, 0) + 1
    return kept[:global_limit] if global_limit else kept


def scan_candidates(adapter, scan_result, config):
    rows = [row_to_candidate(adapter, row, item['panel'])
            for item in scan_result.domains['direct']['panels'] for row in item['rows']]
    return select_candidates(rows, adapter=adapter, rho_threshold=config.rho_threshold,
                             per_hw_limit=config.em_top_per_hw, global_limit=config.em_top_k,
                             mask_hw_min=config.mask_hw_min,
                             mask_hw_max=min(config.mask_hw_max, adapter.metadata.block_bits))
