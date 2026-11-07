#!/usr/bin/env python3
"""Summarize the public-SPX tables and diagnostics used in the manuscript."""
import hashlib
import json
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_public_spx import DATA, OUT, DATES, metrics, black_otm

LABELS={'rough_heston':'Rough Heston','rough_hawkes_heston':'Rough Hawkes-Heston'}


def main():
    summaries=[json.loads((OUT/f'{d}_summary.json').read_text()) for d in DATES]
    frames=[pd.read_csv(OUT/f'{d}_predictions.csv',dtype={'date':str}) for d in DATES]
    all_data=pd.concat(frames,ignore_index=True)
    rows=[]
    params=[]
    for summary in summaries:
        for model,info in summary['models'].items():
            params.append(dict(date=summary['date'],model=model,**info['parameters']))
            for split,values in info['metrics'].items():
                rows.append(dict(date=summary['date'],model=model,split=split,**values))
    for split in ['train','strike_holdout','maturity_holdout','combined_holdout','left_tail_holdout']:
        if split=='combined_holdout':
            d=all_data[all_data.split!='train']
        elif split=='left_tail_holdout':
            d=all_data[(all_data.split!='train')&(all_data.k<-.1)]
        else:
            d=all_data[all_data.split==split]
        for model in LABELS:
            rows.append(dict(date='pooled',model=model,split=split,**metrics(d,d[model+'_iv'].to_numpy())))
    tab=pd.DataFrame(rows)
    tab.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(params).to_csv(OUT/'parameters.csv',index=False)
    all_data.to_csv(OUT/'all_predictions.csv',index=False)

    audit=[]
    for df in frames:
        for t,d in df.groupby('Texp'):
            d=d.sort_values('k'); k=d.k.to_numpy(); tt=d.Texp.to_numpy()
            prices=black_otm(k,tt,d.mid_iv.to_numpy())+np.maximum(1-np.exp(k),0)
            slopes=np.diff(prices)/np.diff(np.exp(k))
            audit.append(dict(date=str(d.date.iloc[0]),Texp=t,n=len(d),
                midpoint_vertical_violations=int(np.sum((slopes>1e-8)|(slopes<-1-1e-8))),
                midpoint_butterfly_violations=int(np.sum(np.diff(slopes)<-1e-8))))
    pd.DataFrame(audit).to_csv(OUT/'data_arbitrage_diagnostics.csv',index=False)
    manifest=json.loads((DATA/'source_manifest.json').read_text())
    for f in manifest['files']:
        assert hashlib.sha256((DATA/f['file']).read_bytes()).hexdigest()==f['sha256']
    root=Path(__file__).resolve().parents[1]
    tracked=[str(f.relative_to(root)) for f in sorted((root/'numerics').glob('*.py'))]
    tracked+=['arxiv/main.tex','Bondi.pdf']
    env=dict(python=platform.python_version(),packages={m:importlib.metadata.version(m)
             for m in ['numpy','scipy','pandas','matplotlib','numba']},
             sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in tracked})
    (OUT/'reproducibility.json').write_text(json.dumps(env,indent=2)+'\n')

    print(tab[tab.date=='pooled'].to_string(index=False))


if __name__=='__main__':
    main()
