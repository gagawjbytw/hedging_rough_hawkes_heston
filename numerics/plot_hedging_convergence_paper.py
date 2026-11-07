#!/usr/bin/env python3
"""Plot the manuscript's fixed-cutoff experiment from stored simulation data."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'numerics/output/original_market_stability'


def main():
    data=pd.read_csv(DATA/'convergence.csv')
    d=data[(data.steps==512)&(data.cutoff==20)].sort_values('epsilon_over_T',ascending=False)
    eps=d.epsilon_over_T.to_numpy()
    np.testing.assert_allclose(eps,[1/4,1/8,1/16,1/32,1/64])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(5.7,3.1),layout='constrained')
    for column,label,color,marker in [
        ('strategy_l2_same_cutoff',r'Strategy distance $\widehat d_{\varepsilon,20}^{\,2}$','#146d91','o'),
        ('gain_sup_same_cutoff',r'Uniform gain distance $\widehat g_{\varepsilon,20}^{\,2}$','#bf593c','s')]:
        mean=d[column].to_numpy(); se=d[column+'_se'].to_numpy()
        assert np.all(mean-1.96*se>0)
        ax.plot(eps,mean,color=color,marker=marker,ms=4,lw=1.5,label=label)
        ax.fill_between(eps,mean-1.96*se,mean+1.96*se,color=color,alpha=.16,linewidth=0)
    ax.set_xscale('log',base=2)
    ax.set_yscale('log')
    ax.set_xticks(eps,[r'$1/4$',r'$1/8$',r'$1/16$',r'$1/32$',r'$1/64$'])
    ax.invert_xaxis()
    ax.set_xlabel(r'Kernel shift $\varepsilon/T$')
    ax.set_ylabel('Squared distance')
    ax.legend(loc='upper right',frameon=False,fontsize=8.5)
    ax.grid(alpha=.16,which='both')
    fig.savefig(ROOT/'arxiv/figures/hedging_convergence.pdf',bbox_inches='tight')
    fig.savefig(DATA/'hedging_convergence_paper.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
