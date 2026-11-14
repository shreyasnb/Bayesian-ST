import numpy as np

def isotropized_mean(M,P,tolerance=1e-9):
    n=len(M);rho=float((np.trace(P@M).real-1/n)/(1-1/n))
    if not -tolerance<=rho<=1+tolerance:raise ArithmeticError('Matched isotropization coefficient outside [0,1]')
    rho=float(np.clip(rho,0.,1.));result=rho*P+(1-rho)*np.eye(n)/n
    if abs(np.trace(P@(M-result)).real)>10*tolerance:
        raise ArithmeticError('Matched isotropization changes modal chordal risk')
    return result,rho
