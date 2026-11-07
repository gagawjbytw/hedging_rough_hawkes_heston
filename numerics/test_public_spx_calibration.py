"""Independent numerical checks for the public-data calibration pricer."""
import json
import unittest

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize_scalar
from scipy.special import gamma

from calibrate_public_spx import (OUT, NAMES, Pricer, kernel_factors,
                                 transform_lift, rmap, drmap)


@njit(cache=True)
def direct_fractional_transform(u, t, p, jumps, left, right):
    """Independent O(N^2) linear product integration of the fractional equation."""
    steps = len(left)
    h = t/steps
    out = np.empty(len(u),dtype=np.complex128)
    for k in range(len(u)):
        f = np.empty(steps+1,dtype=np.complex128)
        f[0] = rmap(u[k],0j,p,jumps)
        oldpsi = 0j
        integ = 0j
        for n in range(1,steps+1):
            base = left[n-1]*f[0]
            for j in range(1,n):
                base += (left[n-j-1]+right[n-j])*f[j]
            v = oldpsi
            for it in range(40):
                change = (v-base-right[0]*rmap(u[k],v,p,jumps))/(1-right[0]*drmap(u[k],v,p,jumps))
                v -= change
                if abs(change)<1e-12*(1+abs(v)):
                    break
            f[n] = rmap(u[k],v,p,jumps)
            integ += h/2*(p[5]*(f[n-1]+f[n])+p[4]*(oldpsi+v))
            oldpsi = v
        out[k] = np.exp(integ)
    return out


def direct(u,t,p,jumps,steps=2048):
    alpha=p[0]
    lag=np.arange(1,steps+1,dtype=float)
    a=(lag**alpha-(lag-1)**alpha)/alpha
    right=lag*a-(lag**(alpha+1)-(lag-1)**(alpha+1))/(alpha+1)
    factor=(t/steps)**alpha/gamma(alpha)
    return direct_fractional_transform(u,t,p,jumps,(a-right)*factor,right*factor)


def heston_transform(u,t,p):
    beta=-p[2]-p[1]*np.sqrt(p[3])*u
    d=np.sqrt(beta**2-p[3]*(u*u-u))
    g=(beta-d)/(beta+d)
    e=np.exp(-d*t)
    D=(beta-d)/p[3]*(1-e)/(1-g*e)
    A=p[4]/p[3]*((beta-d)*t-2*np.log((1-g*e)/(1-g)))
    return np.exp(A+p[5]*D)


def lifted(u,t,p,jumps,steps=2048):
    x,w=kernel_factors(p[0],7)
    grid=t*np.linspace(0,1,steps+1)**2
    return transform_lift(u,grid,np.array([t]),x,w,p,jumps)[0]


class PricingTests(unittest.TestCase):
    def test_constant_variance_black_limit(self):
        d=pd.DataFrame({'k':np.tile([-.2,-.1,0,.05],3),'Texp':np.repeat([.032,.092,.168],4)})
        p=np.array([.65,-.7,-1.,0.,.04,.04,0.])
        np.testing.assert_allclose(Pricer(d).vols(p,False),.2,atol=2e-8)

    def test_classical_heston_transform(self):
        u=.5+1j*np.array([0.,1.,5.,20.,80.,200.])
        p=np.array([1.,-.7,-2.,.3,.08,.04,0.])
        np.testing.assert_allclose(lifted(u,.17,p,False),heston_transform(u,.17,p),atol=3e-7,rtol=2e-5)

    def test_fractional_solver_with_and_without_jumps(self):
        u=.5+1j*np.array([0.,1.,5.,20.,80.,200.])
        p=np.array([.55,-.7,-1.8,.3,.049,.04,.276])
        for jumps in [False,True]:
            np.testing.assert_allclose(lifted(u,.09,p,jumps),direct(u,.09,p,jumps),atol=5e-5,rtol=2e-4)

    def test_normalization_and_martingale(self):
        p=np.array([.527,-.731,-1.812,.115,.049,.0079,.276])
        np.testing.assert_allclose(lifted(np.array([0j,1+0j]),.168,p,True),1.,atol=1e-13)

    def test_jump_fourier_quadrature_convergence(self):
        # Large Lambda needs resolved low frequencies as well as a large cutoff.
        d=pd.DataFrame({'k':[-.25,-.1,0.,.06],'Texp':[.032]*4})
        p=np.array([.73,-.76,-16.,2.,.43,.07,1.9])
        a=Pricer(d,steps=512,cutoff=800.,quad_order=10).vols(p,True)
        b=Pricer(d,steps=1024,cutoff=1600.,quad_order=16,factor_order=7).vols(p,True)
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=0)


def validate_fits():
    rows=[]
    for path in sorted(OUT.glob('*_summary.json')):
        summary=json.loads(path.read_text())
        for name,model in summary['models'].items():
            p=np.array([model['parameters'][key] for key in NAMES])
            jumps=name=='rough_hawkes_heston'
            # The manuscript's global criterion requires v < 2*Lambda.
            # Exponential marks permit v < 1+2*Lambda for a finite-moment
            # upper bound from the positive local martingale e^U (not an
            # assertion of equality in the manuscript's moment formula).
            top=1+2*p[6]-1e-6 if jumps else 200.
            global_top=max(2*p[6]-1e-6,1e-10) if jumps else top
            barrier=minimize_scalar(lambda v:float(rmap(2+0j,v+0j,p,jumps).real),bounds=(0,global_top),method='bounded')
            u=.5+1j*np.array([0.,1.,5.,20.,80.,200.])
            t=max(summary['data']['fit_times'])
            error=float(np.max(abs(lifted(u,t,p,jumps)-direct(u,t,p,jumps,4096))))
            # If the global negative barrier fails, a constant supersolution on
            # [0,T] suffices: A_T max(F2(0),F2(v)) < v. F2 is convex there.
            scale=t**p[0]/gamma(p[0]+1)
            f0=float(rmap(2+0j,0j,p,jumps).real)
            finite=minimize_scalar(lambda v:scale*max(f0,float(rmap(2+0j,v+0j,p,jumps).real))-v,
                                   bounds=(0,top),method='bounded')
            # Stronger supersolution h(t)=v*(t/T)^alpha. Convexity of F2 makes
            # G(x)=max(F2(0),F2(x)) nondecreasing for x>=0. The right-endpoint
            # sum below is an upper bound on the fractional integral.
            y=np.linspace(0,1,4097)
            upper_weights=(1-y[:-1])**p[0]-(1-y[1:])**p[0]
            powers=y[1:]**p[0]
            def power_gap(v):
                values=np.maximum(f0,np.real(rmap(2+0j,v*powers+0j,p,jumps)))
                return scale*np.dot(upper_weights,values)-v
            power=minimize_scalar(power_gap,bounds=(0,top),method='bounded')
            row=dict(date=summary['date'],model=name,
                     parameters=model['parameters'],
                     independent_cf_max_abs_error=error,
                     l2_barrier_v=float(barrier.x),F2_at_barrier=float(barrier.fun),
                     l2_global_barrier_passed=bool(barrier.fun<=0),
                     finite_horizon=t,finite_horizon_barrier_v=float(finite.x),
                     finite_horizon_supersolution_margin=float(-finite.fun),
                     power_barrier_v=float(power.x),power_upper_sum_cells=4096,
                     power_supersolution_margin=float(-power.fun),
                     l2_finite_horizon_sufficient_criterion_passed=bool(barrier.fun<=0 or finite.fun<0 or power.fun<0))
            assert error<2e-4,row
            assert barrier.fun<=0 or finite.fun<0 or power.fun<0,row
            rows.append(row)
    (OUT/'independent_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    result=unittest.main(exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    validate_fits()
