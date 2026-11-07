#!/usr/bin/env python3
"""Regenerate the manuscript figure from frozen public-SPX predictions."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'numerics/output/public_spx_calibration'
FIGURES = ROOT/'arxiv/figures'
DATES = ['20241219','20250416','20250620','20250702']


def main():
    plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(6.45,4.65),layout='constrained')
    for ax,date in zip(axes.flat,DATES):
        d=pd.read_csv(DATA/f'{date}_predictions.csv')
        d=d[d.Texp==.032].sort_values('k')
        ax.fill_between(d.k,100*d.Bid,100*d.Ask,color='#bfc7cf',alpha=.75,label='Market bid-ask')
        ax.plot(d.k,100*d.mid_iv,'.',color='#555b63',ms=2.5,label='Market midpoint')
        ax.plot(d.k,100*d.rough_heston_iv,color='#bf593c',lw=1.25,label='Rough Heston')
        ax.plot(d.k,100*d.rough_hawkes_heston_iv,color='#146d91',lw=1.25,label='Rough Hawkes-Heston')
        h=d[d.split=='strike_holdout']
        ax.scatter(h.k,100*h.mid_iv,s=15,facecolor='none',edgecolor='#333333',lw=.55,label='Held-out strike')
        ax.set_title(f'{date[:4]}-{date[4:6]}-{date[6:]}',fontsize=9)
        ax.set_xlabel(r'$\log(K/F)$')
        ax.set_ylabel('Implied volatility (%)')
        ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncols=3,frameon=False,fontsize=7.5)
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURES/'public_spx_smiles.pdf',bbox_inches='tight')
    fig.savefig(DATA/'paper_smile_comparison.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
