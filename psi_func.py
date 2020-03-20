import numpy as np
from scipy.integrate import fixed_quad
from scipy.optimize import root_scalar, root
from psi_para import *

def g(t): 
    """owner's beliefs"""
    return Q*np.exp(-l*t)/(Q*np.exp(-l*t)+1.-Q)

def S(t,T): 
    """surplus"""
    if t <= T:
        Z = np.exp(-(r+b)*(T-t))
        return ((1.-g(t))/(1.-g(T)))*(C1*Z+(1.-g(T))*C2*(1.-Z))
    else:
        return np.nan

def VL(t,T): 
    """L-value"""
    qo = fixed_quad(lambda s: g(s)*(Y+S(s,T))*np.exp(-r*(s-t)),t,T)
    return VL1*np.exp(-r*(T-t))+l*qo[0]


def q(t,T): 
    """market beliefs"""
    if t <= T:
        return r*(VL(t,T)+c)/(l*Y)
    else:
        return np.nan

def f(t,y,T): 
    """
    Right-hand-side for Lambda-Gamma ODE
    
    Parameters
    ----------
    t : float
        time
    y : ndarray
        y[0] is Lambda, y[1] is Gamma
    T : float
        H-type selling time

    Returns
    -------
    out : ndarray, 
    """
    F = Q*(1.-q(t,T))/(q(t,T)*(1.-Q)-Q*np.exp(-l*t)*(1.-q(t,T)))
    y_prime = np.zeros(y.shape)
    y_prime[0] = l*np.exp(-l*t)*(1.-y[1])
    y_prime[1] = b*F*y[0]-b*(1.-y[1])
    return y_prime

def Gamma(T,t_hat,t=None):
    """solves for Gamma: either at T or over [t_hat,T]"""
    y0 = np.array([1.-np.exp(-l*t_hat),0.]) 
    if t:
        sol = solve_ivp(lambda s,y: f(s,y,T),(tau(T),T),y0,t_eval=t)
        return sol.y[1]
    else:
        sol = solve_ivp(lambda s,y: f(s,y,T),(tau(T),T),y0)
        return sol.y[1][-1]

def findbif(x): 
    """bifurcation"""
    T, t = x
    return np.array([VL(t,T)-(Q*l*Y/r-c),r*VL(t,T)-l*g(t)*(Y+S(t,T))])

@np.vectorize
def that(T): 
    y0 = np.array([1.-np.exp(-l*tau(T)),np.exp(-b*tau(T))]) 
    sol = solve_ivp(lambda s,y: f(s,y,T),(tau(T),T),y0,events=that_evnt)
    return sol.y[1][-1]
