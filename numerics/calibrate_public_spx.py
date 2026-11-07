#!/usr/bin/env python3
"""Reproducible SPX-only calibration of the manuscript and its nu=0 control.

Python 3.11+, numpy/scipy/pandas/numba. See numerics/README.md for data
provenance and the interpretation of holdouts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.special import gamma, ndtr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "numerics/data/public_spx"
OUT = ROOT / "numerics/output/public_spx_calibration"
DATES = ["20241219", "20250416", "20250620", "20250702"]
NAMES = ["alpha", "rho", "b", "c", "gamma", "v0", "Lambda"]
# Positive parameters c/gamma/v0 are optimized in logs. The manuscript allows b in R.
LOW = np.array([.501, -.995, -50., np.log(.0001), np.log(.000001), np.log(.0001), 0.])
HIGH = np.array([.999, .995, 20., np.log(40.), np.log(6.), np.log(.6), 3.])


def physical(x):
    return np.array([x[0], x[1], x[2], np.exp(x[3]),
                     np.exp(x[4]), np.exp(x[5]), x[6] if len(x) == 7 else 0.])


def transformed(p, jumps):
    x = np.array([p[0], p[1], p[2], np.log(p[3]),
                  np.log(p[4]), np.log(p[5]), p[6]])
    return x if jumps else x[:6]


def kernel_factors(alpha, order=5, xmax=1e8):
    """Positive log-Laplace quadrature; low-frequency mass included explicitly."""
    if alpha == 1.:
        return np.array([0.]), np.array([1.])
    z, w = leggauss(order)
    edges = np.arange(-8., np.log10(xmax) + .5)
    ss = np.concatenate([(a+b)/2+(b-a)/2*z for a,b in zip(edges[:-1]*np.log(10), edges[1:]*np.log(10))])
    ww = np.tile(w*np.log(10)/2, len(edges)-1)
    norm = 1/(gamma(alpha)*gamma(1-alpha))
    x0 = 1e-8
    nodes = np.r_[x0*(1-alpha)/(2-alpha), np.exp(ss)]
    weights = np.r_[norm*x0**(1-alpha)/(1-alpha), norm*ww*np.exp((1-alpha)*ss)]
    return nodes, weights


@njit(cache=True)
def rmap(u, v, p, jumps):
    _, rho, b, c, _, _, lam = p
    r = .5*(u*u-u)+(b+rho*np.sqrt(c)*u)*v+.5*c*v*v
    if jumps:
        r += 1/(1+lam*u-v)+u*lam/(1+lam)-1-v
    return r


@njit(cache=True)
def drmap(u, v, p, jumps):
    r = p[2]+p[1]*np.sqrt(p[3])*u+p[3]*v
    if jumps:
        r += 1/(1+p[6]*u-v)**2-1
    return r


@njit(cache=True)
def transform_lift(u, times, targets, nodes, weights, p, jumps):
    """Exponential trapezoidal lift, scalar implicit Newton solve per frequency.

    With the consistent g0=v0+gamma*integral(K), log M=v0*int(R)+gamma*int(psi).
    """
    nf, nq = len(nodes), len(u)
    y = np.zeros((nq, nf), dtype=np.complex128)
    psi = np.zeros(nq, dtype=np.complex128)
    old = np.array([rmap(ui, 0j, p, jumps) for ui in u])
    logm = np.zeros(nq, dtype=np.complex128)
    out = np.zeros((len(targets), nq), dtype=np.complex128)
    target = 0
    for n in range(1, len(times)):
        h = times[n]-times[n-1]
        decay = np.exp(-nodes*h)
        a = np.empty(nf)
        beta = np.empty(nf)
        for j in range(nf):
            z = nodes[j]*h
            if abs(z)<1e-5:
                a[j] = h*(1-z/2+z*z/6)
                beta[j] = h*(.5-z/6+z*z/24)
            else:
                a[j] = -np.expm1(-z)/nodes[j]
                beta[j] = (h-a[j])/z
        wb = np.sum(weights*beta)
        wa = np.sum(weights*(a-beta))
        for k in range(nq):
            base = wa*old[k]
            for j in range(nf):
                base += weights[j]*decay[j]*y[k,j]
            v = psi[k]
            converged = False
            for it in range(30):
                f = rmap(u[k],v,p,jumps)
                delta = (v-base-wb*f)/(1-wb*drmap(u[k],v,p,jumps))
                v -= delta
                if abs(delta)<1e-11*(1+abs(v)):
                    converged = True
                    break
            if not converged or not np.isfinite(v.real) or v.real>1e-6:
                return out+np.nan
            new = rmap(u[k],v,p,jumps)
            logm[k] += h*.5*(p[5]*(old[k]+new)+p[4]*(psi[k]+v))
            for j in range(nf):
                y[k,j] = decay[j]*y[k,j]+(a[j]-beta[j])*old[k]+beta[j]*new
            psi[k], old[k] = v,new
        if target<len(targets) and abs(times[n]-targets[target])<1e-12:
            out[target] = np.exp(logm)
            target += 1
    return out


def black_otm(k, t, vol):
    s = vol*np.sqrt(t)
    d1 = -k/s+s/2
    d2 = d1-s
    return np.where(k>=0, ndtr(d1)-np.exp(k)*ndtr(d2),
                    np.exp(k)*ndtr(-d2)-ndtr(-d1))


def implied_vol(price, k, t):
    lo = np.full_like(k, 1e-7)
    hi = np.full_like(k, 5.)
    if np.any(~np.isfinite(price)) or np.any(price<=0) or np.any(price>=np.minimum(1,np.exp(k))):
        raise FloatingPointError("Invalid OTM option price")
    for _ in range(36):
        mid = (lo+hi)/2
        mask = black_otm(k,t,mid)<price
        lo = np.where(mask,mid,lo)
        hi = np.where(mask,hi,mid)
    return (lo+hi)/2


class Pricer:
    def __init__(self, frame, steps=256, cutoff=400., quad_order=10, factor_order=5):
        self.k = frame.k.to_numpy()
        self.t = frame.Texp.to_numpy()
        self.targets = np.unique(self.t)
        self.time_index = np.searchsorted(self.targets,self.t)
        # A quadratic grid resolves the fractional singularity at the origin.
        self.times = np.unique(np.r_[self.targets, self.targets[-1]*np.linspace(0,1,steps+1)**2])
        z,w = leggauss(quad_order)
        # Jump transforms vary rapidly near frequency zero. Small first panels
        # are essential; increasing only the upper cutoff does not cure this.
        edges = np.r_[0.,.5,1.,2.,4.,8.,np.arange(16.,cutoff,16.),cutoff]
        self.freq = np.concatenate([(a+b)/2+(b-a)/2*z for a,b in zip(edges[:-1],edges[1:])])
        self.qw = np.concatenate([(b-a)/2*w for a,b in zip(edges[:-1],edges[1:])])
        self.u = .5+1j*self.freq
        self.fourier = np.exp(-1j*self.k[:,None]*self.freq)*self.qw/(self.freq**2+.25)/np.pi*np.exp(self.k[:,None]/2)
        self.bs_vol = .25
        self.bs_transform = np.exp(.5*(self.u*self.u-self.u)[None,:]*self.targets[:,None]*self.bs_vol**2)
        self.bs_price = black_otm(self.k,self.t,self.bs_vol)
        self.factor_order = factor_order

    def prices(self,p,jumps):
        nodes,weights = kernel_factors(p[0],self.factor_order)
        phi = transform_lift(self.u,self.times,self.targets,nodes,weights,p,jumps)
        delta = (self.bs_transform-phi)[self.time_index]
        return self.bs_price+np.real(np.sum(self.fourier*delta,axis=1))

    def vols(self,p,jumps):
        return implied_vol(self.prices(p,jumps),self.k,self.t)


def select_data(date):
    raw = pd.read_csv(DATA/f"{date}_spx_vol.csv")
    d = raw.copy()
    d['raw_row'] = np.arange(len(d))+2
    d['k'] = np.log(d.Strike/d.Fwd)
    d['mid_iv'] = (d.Bid+d.Ask)/2
    keep = (np.isfinite(d[['Texp','Strike','Bid','Ask','Fwd']]).all(axis=1)
            &d.Texp.between(.02,.20)&d.k.between(-.30,.10)
            &(d.Bid>.01)&(d.Ask>=d.Bid)&(d.Ask<2.)
            &((d.Ask-d.Bid)/d.mid_iv<=.20))
    d = d[keep].copy()
    available = np.array(sorted(t for t,g in d.groupby('Texp') if len(g)>=25))
    fit_times = sorted(set(float(available[np.argmin(abs(available-t))]) for t in [.032,.052,.072,.092,.168]))
    extra = np.array([t for t in available if t not in fit_times])
    held_time = float(extra[np.argmin(abs(extra-.124))])
    selected = []
    for t in sorted(fit_times+[held_time]):
        g = d[d.Texp==t].sort_values('k').drop_duplicates('Strike')
        # Nearest observed strikes to a uniform log-moneyness grid; no interpolation.
        ids = np.unique([int(np.argmin(abs(g.k.to_numpy()-x))) for x in np.linspace(g.k.min(),g.k.max(),41)])
        g = g.iloc[ids].copy()
        g['split'] = ['maturity_holdout' if t==held_time else ('strike_holdout' if i%4==1 else 'train') for i in range(len(g))]
        g['date'] = date
        selected.append(g)
    selected = pd.concat(selected,ignore_index=True)
    audit = dict(raw_rows=len(raw),eligible_rows=len(d),selected_rows=len(selected),
                 fit_times=fit_times,held_time=held_time,split_counts=selected.split.value_counts().to_dict())
    return selected,audit


def metrics(df,vol):
    e = vol-df.mid_iv.to_numpy()
    return dict(n=len(df),rmse_vol_points=float(100*np.sqrt(np.mean(e*e))),
                relative_rmse_pct=float(100*np.sqrt(np.mean((e/df.mid_iv.to_numpy())**2))),
                mae_vol_points=float(100*np.mean(abs(e))),
                inside_bid_ask_pct=float(100*np.mean((vol>=df.Bid.to_numpy())&(vol<=df.Ask.to_numpy()))))


def calibrate(df,jumps,starts,steps,cutoff,max_nfev):
    pricer = Pricer(df,steps=steps,cutoff=cutoff)
    market = df.mid_iv.to_numpy()
    size = 7 if jumps else 6
    logs = []
    def residual(x):
        try:
            return (pricer.vols(physical(x),jumps)-market)/market
        except (FloatingPointError,ValueError):
            return np.full(len(market),100.)
    best = None
    for i,p in enumerate(starts):
        x = np.clip(transformed(p,jumps), LOW[:size]+1e-8,HIGH[:size]-1e-8)
        t0 = time.perf_counter()
        result = least_squares(residual,x,bounds=(LOW[:size],HIGH[:size]),
            diff_step=2e-4,ftol=2e-7,xtol=2e-7,gtol=2e-7,max_nfev=max_nfev,x_scale='jac')
        row = dict(start=i,relative_rmse_pct=float(100*np.sqrt(np.mean(result.fun**2))),
                   nfev=result.nfev,status=int(result.status),message=result.message,
                   optimality=float(result.optimality),seconds=time.perf_counter()-t0,
                   parameters=dict(zip(NAMES,physical(result.x).tolist())),
                   active_mask=result.active_mask.tolist())
        logs.append(row)
        print(('RHH' if jumps else 'RH'),json.dumps(row),flush=True)
        if best is None or result.cost<best.cost:
            best=result
    return physical(best.x),logs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dates',nargs='+',default=DATES)
    ap.add_argument('--starts',type=int,choices=range(1,5),default=3)
    ap.add_argument('--steps',type=int,default=256)
    ap.add_argument('--cutoff',type=float,default=800.)
    ap.add_argument('--max-nfev',type=int,default=160)
    ap.add_argument('--skip-existing',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for date in args.dates:
        saved=OUT/f'{date}_summary.json'
        if args.skip_existing and saved.exists():
            previous=json.loads(saved.read_text())
            if set(previous.get('models',{}))=={'rough_heston','rough_hawkes_heston'}:
                continue
        df,audit=select_data(date)
        df.to_csv(OUT/f'{date}_selected.csv',index=False)
        training=df[df.split=='train']
        atm=training.iloc[np.argmin(abs(training.k.to_numpy()))].mid_iv**2
        starts=[np.array([.60,-.75,-1.8,.3,.07,atm,.25]),
                np.array([.80,-.85,-3.,1.,.15,atm,.10]),
                np.array([.53,-.60,-.5,.1,.03,atm,.50]),
                np.array([.70,-.50,-8.,2.,.3,atm,.05])][:args.starts]
        summary=dict(date=date,data=audit,settings=vars(args),
                     transformed_parameter_bounds=dict(lower=LOW.tolist(),upper=HIGH.tolist()),models={})
        print('DATE',date,json.dumps(audit),flush=True)
        for jumps,name in [(False,'rough_heston'),(True,'rough_hawkes_heston')]:
            p,logs=calibrate(training,jumps,starts,args.steps,args.cutoff,args.max_nfev)
            # Refine the best basin at twice the time resolution.
            p,ref=calibrate(training,jumps,[p],2*args.steps,args.cutoff,100)
            settings=dict(steps=4*args.steps,cutoff=2*args.cutoff,quad_order=12,factor_order=7)
            vol=Pricer(df,**settings).vols(p,jumps)
            coarse=Pricer(df,steps=2*args.steps,cutoff=args.cutoff).vols(p,jumps)
            df[name+'_iv']=vol
            bysplit={split:metrics(df[df.split==split],vol[df.split==split]) for split in df.split.unique()}
            bysplit['all']=metrics(df,vol)
            tail=(df.k<-.1)&(df.split!='train')
            bysplit['left_tail_holdout']=metrics(df[tail],vol[tail])
            summary['models'][name]=dict(parameters=dict(zip(NAMES,p.tolist())),
                parameter_count=7 if jumps else 6,multistart=logs,refinement=ref,
                metrics=bysplit,convergence=dict(max_abs_iv_vol_points=float(100*np.max(abs(vol-coarse))),
                rms_iv_vol_points=float(100*np.sqrt(np.mean((vol-coarse)**2)))),evaluation_settings=settings)
            (OUT/f'{date}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            df.to_csv(OUT/f'{date}_predictions.csv',index=False)
            print('METRICS',date,name,json.dumps(summary['models'][name]),flush=True)
            if not jumps:
                # Same number of starts, including the reoptimized no-jump basin.
                starts[0]=p.copy(); starts[0][6]=.15


if __name__=='__main__':
    main()
