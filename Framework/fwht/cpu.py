"""Small-panel CPU reference scanner; the large paper scans use CUDA."""
import numpy as np
from Cipher.registry import get_reference
from Framework.search.models import EmScanResult
from .candidates import build_panel

def fwht(values):
    values=np.asarray(values,dtype=np.int64).copy()
    if len(values)==0 or len(values)&(len(values)-1):raise ValueError("FWHT length must be a power of two")
    half=1
    while half<len(values):
        view=values.reshape(-1,2*half)
        left,right=view[:,:half].copy(),view[:,half:].copy()
        view[:,:half]=left+right;view[:,half:]=left-right
        half*=2
    return values

def scan(adapter,delta,config,output_dir):
    ref=get_reference(adapter.gpu_name,config.version)
    spec=ref.config
    samples,scout=config.sample_count,config.scout_count
    if not 0<scout<=samples:raise ValueError("CPU scout count must be in 1..sample count")
    if not 1<=config.panel_size<=min(24,spec.block_size):raise ValueError("CPU panel must be in 1..min(B,24)")
    rng=np.random.default_rng(config.seed)
    def words(width,count=samples):
        return rng.bit_generator.random_raw(count)&np.uint64((1<<width)-1)
    state=tuple(words(spec.word_size) for _ in range(spec.state_words))
    key_widths=spec.key_word_bits or (spec.word_size,)*spec.key_words
    keys=tuple(words(width) for width in key_widths)
    difference=ref._state_words(delta)
    paired=tuple(x^np.uint64(d) for x,d in zip(state,difference))
    left=ref.encrypt_batch(state,keys,config.rm,config.em_start)
    right=ref.encrypt_batch(paired,keys,config.rm,config.em_start)
    outputs=tuple(a^b for a,b in zip(left,right))
    def plane(mask):
        parity=np.zeros(samples,dtype=np.uint8)
        for data,word in zip(outputs,ref._state_words(mask)):
            while word:
                bit=(word&-word).bit_length()-1
                parity^=((data>>np.uint64(bit))&np.uint64(1)).astype(np.uint8)
                word&=word-1
        return parity
    domains={}
    for domain in ("direct",):
        def raw_mask(bits):
            mask=adapter.from_paper_bits(bits)
            return mask
        singletons=[]
        for bit in range(spec.block_size):
            rho=1-2*int(plane(raw_mask([bit]))[:scout].sum())/scout
            singletons.append({"bit":bit,"rho":rho,"abs_rho":abs(rho)})
        panels=[build_panel(singletons, config.panel_size)]
        items=[]
        for panel in panels:
            projected=np.zeros(samples,dtype=np.int64)
            for position,bit in enumerate(panel.bits):
                projected|=plane(raw_mask([bit])).astype(np.int64)<<position
            spectrum=fwht(np.bincount(projected,minlength=1<<len(panel.bits)))
            indices=np.flatnonzero(np.abs(spectrum)>float(config.rho_threshold)*samples)
            rows=[]
            for coefficient in indices:
                if not coefficient:continue
                bits=[bit for position,bit in enumerate(panel.bits) if (int(coefficient)>>position)&1]
                rho=int(spectrum[coefficient])/samples
                if rho==0:continue
                rows.append({"mask_index":int(coefficient),"mask_hex":adapter.format_state(adapter.from_paper_bits(bits)),
                    "rho":rho,"Walsh":int(spectrum[coefficient]),"source":"cpu_projected_fwht"})
            rows.sort(key=lambda row:(-abs(row["rho"]),row["mask_index"]))
            for rank,row in enumerate(rows,1):row["rank"]=rank
            items.append({"panel":panel,"rows":rows})
        domains[domain]={"singletons":singletons,"panels":items}
    return EmScanResult(domains,samples,1,True,True,"cpu_reference")
