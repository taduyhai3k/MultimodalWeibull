"""Pure NumPy survival math; lambda is a rate: S(t)=exp(-lambda*t**k)."""
import numpy as np


def curves(k, rate, weights, grid):
    k=np.asarray(k,dtype=np.float64);rate=np.asarray(rate,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64);t=np.maximum(np.asarray(grid,dtype=np.float64),1e-7)
    if k.ndim!=1 or k.shape!=rate.shape or w.shape!=k.shape:
        raise ValueError('Expected one patient: k, rate, weights [M]')
    if not all(np.isfinite(v).all() for v in (k,rate,w)) or (k<=0).any() or (rate<=0).any() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid Weibull parameters')
    w=w/w.sum()
    logt=np.log(t)[None,:]
    z=np.exp(np.minimum(np.log(rate)[:,None]+k[:,None]*logt,80.))
    survival=np.exp(-z)
    logpdf=np.log(rate)[:,None]+np.log(k)[:,None]+(k[:,None]-1)*logt-z
    pdf=np.exp(np.minimum(logpdf,700))
    return dict(pdf=pdf,survival=survival,weighted_pdf=w[:,None]*pdf,
                mixture_pdf=(w[:,None]*pdf).sum(0),mixture_survival=(w[:,None]*survival).sum(0))


def nll(k,rate,weights,t,c):
    k,rate,w=[np.asarray(v,dtype=np.float64) for v in (k,rate,weights)]
    lt=np.log(np.maximum(np.asarray(t).reshape(-1,1),1e-7))
    z=np.exp(np.minimum(np.log(rate)+k*lt,80.))
    log_s=-z;log_f=np.log(rate)+np.log(k)+(k-1)*lt-z
    w=np.maximum(w,1e-7);w=w/w.sum(1,keepdims=True)
    terms=np.log(w)+np.where(np.asarray(c).reshape(-1,1)==1,log_s,log_f)
    m=terms.max(1)
    return -(m+np.log(np.exp(terms-m[:,None]).sum(1)))


def c_index(times,censors,risks):
    t=np.asarray(times);c=np.asarray(censors);r=np.asarray(risks)
    pairs=0;score=0.
    for i in range(len(t)):
        if c[i]!=0:continue
        eligible=(t>t[i])|((t==t[i])&(c==1))
        delta=r[i]-r[eligible];pairs+=len(delta)
        score+=(delta>1e-8).sum()+.5*(np.abs(delta)<=1e-8).sum()
    return dict(c_index=float(score/pairs) if pairs else None,comparable_pairs=int(pairs))


def risk_rmst(k,rate,weights,horizon):
    grid=np.linspace(0,horizon,200)
    values=[]
    for a,b,w in zip(k,rate,weights):
        s=curves(a,b,w,grid)['mixture_survival'];s[0]=1.
        # numpy 1.x and 2.x
        integrate=getattr(np,'trapezoid',None) or np.trapz
        values.append(-float(integrate(s,grid)))
    return np.asarray(values)
