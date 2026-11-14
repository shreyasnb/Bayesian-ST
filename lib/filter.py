"""One correction per observation with pre-correction prior storage."""
from collections import deque
import numpy as np
from .geometry import center, mode
from .models import increment
from .em import Record, fit_noise


class BinghamFilter:
    def __init__(self,n,k,s,tau2=None,beta=1.,A0=None,window=32,complex_data=False):
        if not 0<k<n or not 0<beta<=1 or s<=0 or window<1:
            raise ValueError('invalid rank, forgetting, variance, or window')
        self.n,self.k,self.s,self.tau2=n,k,float(s),tau2
        self.beta,self.complex_data=beta,complex_data
        self.A=center(np.zeros((n,n),dtype=complex if complex_data else float) if A0 is None else A0)
        self.records=deque(maxlen=window)
        self.t=0
        self.last_fit=None

    def update(self,u,v=None,G=None,fit=False,mean=None,iterations=5,log_normalizer=None):
        u=np.asarray(u)
        if u.shape != (self.n,): raise ValueError('observation dimension')
        G=np.eye(self.n) if G is None else np.asarray(G)
        if not np.allclose(G.conj().T@G,np.eye(self.n),atol=1e-9):
            raise ValueError('prediction requires an orthogonal/unitary G')
        prior=center(self.beta*G@self.A@G.conj().T)
        self.records.append(Record(prior.copy(),u.copy(),None if v is None else np.asarray(v).copy()))
        if fit:
            if len(self.records)<self.records.maxlen or mean is None:
                raise ValueError('noise fitting requires a full window and a mean evaluator')
            self.last_fit=fit_noise(list(self.records),self.k,self.s,self.tau2,mean,iterations,
                                    log_normalizer=log_normalizer,complex_data=self.complex_data)
            self.s,self.tau2=self.last_fit.s,self.last_fit.tau2
        self.A=center(prior+increment(u,self.s,self.tau2,v,self.complex_data))
        self.t+=1
        return self.basis

    @property
    def basis(self): return mode(self.A,self.k)
