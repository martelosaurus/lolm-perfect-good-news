import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad, solve_ivp
from scipy.optimize import root_scalar, root

def _g(t): 
    """owner's beliefs"""
    return Q*np.exp(-l*t)/(Q*np.exp(-l*t)+1.-Q)

@np.vectorize
def S(t,T): 
    """surplus"""
    if t <= T:
        Z = np.exp(-(r+b)*(T-t))
        return ((1.-_g(t))/(1.-_g(T)))*(C1*Z+(1.-_g(T))*C2*(1.-Z))
    else:
        return np.nan

@np.vectorize
def VL(t,T): 
    """L-value"""
    qo = fixed_quad(lambda s: _g(s)*(Y+S(s,T))*np.exp(-r*(s-t)),t,T)
    return VL1*np.exp(-r*(T-t))+l*qo[0]

@np.vectorize
def _q(t,T): 
    """market beliefs"""
    if t <= T:
        return r*(VL(t,T)+c)/(l*Y)
    else:
        return np.nan

@np.vectorize
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
    out : ndarray
    """
    F = Q*(1.-_q(t,T))/(_q(t,T)*(1.-Q)-Q*np.exp(-l*t)*(1.-_q(t,T)))
    y_prime = np.zeros(y.shape)
    y_prime[0] = l*np.exp(-l*t)*(1.-y[1])
    y_prime[1] = b*F*y[0]-b*(1.-y[1])
    return y_prime

@np.vectorize
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
    """
    Finds the bifurcation point

    Parameters
    ----------
    x : ndarray
        Point (T,t) 

    Notes
    -----
    The bifurcation point is the point (T,t) at which dq/dt = 0 and q = Q
    """
    T, t = x
    return np.array([VL(t,T)-(Q*l*Y/r-c),r*VL(t,T)-l*_g(t)*(Y+S(t,T))])

@np.vectorize
def that(T): 
    """
    Find initial time for each final time

    Parameters
    ----------
    T : float
        Final time
    """
    y0 = np.array([1.-np.exp(-l*tau(T)),np.exp(-b*tau(T))]) 
    sol = solve_ivp(lambda s,y: f(s,y,T),(tau(T),T),y0,events=that_evnt)
    return sol.y[1][-1]

class Model:

    def __init__(self,b=.1,c=.1,l=.5,r=.5,Q=7,Y=1.):

        # parameters
        self.b = .1
        self.c = .1
        self.l = .5
        self.r = .5
        self.Q = .7
        self.Y = 1.

        # title string
        self.titstr = '$\\pi = $' + str(b) + ', '
        self.titstr += '$c = $' + str(c) + ', '
        self.titstr += '$\\lambda = $' + str(l) + ', '
        self.titstr += '$r = $' + str(r) + ', ' 
        self.titstr += '$Q = $' + str(Q) + ', '
        self.titstr += '$Y = $' + str(Y)

        # synthetic parameters
        self.C1 = r*c/(r+b)
        self.C2 = l*Y/(r+b)

        # terminal conditions
        self.VL1 = l*Y/r-c 
        self.VH1 = l*Y/r-r*c/(r+b)

        # optimization parameters
        self.T_max = 10.
        self.opttol = 1.e-6

        # solve for equilibrium T_star
        if True:
            if np.isinf(T1):
                T_opt = root_scalar(G2,x0=1.5*T0,x1=2.*T0).root
            else:
                T_opt = root_scalar(G2,bracket=((1.+opttol)*T0,(1.-opttol)*T1)).root
        else:
            T_opt = T1

        sol = root(findbif,np.array([2.,1.]))
        T0, t0 = sol.x

        sol = root_scalar(lambda T: q(0.,T)-Q,x0=T0,x1=2*T0)
        if np.isnan(sol.root):
            T1 = np.inf
        else:  
            T1 = sol.root

    def plot(Tt_max,fname,N=200):
        """ 
        Plots levels sets of q, Gamma in T-t space:
                Q = q(t_hat,T)
                1 = Gamma(T,t_hat,T)

        Parameters
        ----------
        Tt_max : float
            Maximum T and t_hat to plot
        fname : str
            Filename (without extension) of plot

        Examples
        --------
        >>> from lolm_good_base import Model
        >>> model = Model()
        >>> model.plot(T_max,'demo' + version)
        """

        #plotting 
        npz = np.zeros(N)
        npo = np.ones(N)
        rcol = 'tab:red'
        bcol = 'tab:blue'
                
        # parameters
        xpos1 =  .3*Tt_max
        xpos2 =  .6*Tt_max
        ypos =  .5*Tt_max

        # plotting vectors and matrices
        T_vec = np.linspace(0.,Tt_max,N)
        t_vec = np.linspace(0.,Tt_max,N)
        T_mat, t_mat = np.meshgrid(T_vec,t_vec)

        # upper-lower t_hat
        t_hat_upper = t_hat(T_vec,True) 
        t_hat_lower = t_hat(T_vec,False)

        # plot
        plt.contour(T_mat,t_mat,_q(t_mat,T_mat),colors="black",alpha=.1)
        plt.plot(T_vec,t_vec,'-k')
        plt.plot(T0,t0,'ok')
        plt.plot(T_SC(t_vec),t_vec,color=rcol)

        # q(t,T)-Q 
        plt.plot(T_vec,t_hat_upper,linestyle='--',color=bcol)
        plt.plot(T_vec,t_hat_lower,linestyle='-',color=bcol)
        plt.annotate('$q(T,T)=1$',(xpos1,ypos))
        plt.annotate('$q(t,T)=Q$',(xpos2,ypos))
        this_xticks = [0.,T0,T_opt]
        this_xticklabels = ['$0$','$T_{0}$']
        if not np.isinf(T1): 
            this_xticks += [T1]
            this_xticklabels += ['$T_{1}$']
        plt.xticks(this_xticks,this_xticklabels)
        plt.yticks([0.,t0],['$0$','$t_{0}$'])
        plt.axis([0.,Tt_max,0.,Tt_max])
        plt.xlabel('$T$')
        plt.ylabel('$t$')
        plt.title(titstr)
        plt.grid()
        plt.savefig(fname + 'Tt.pdf')
        plt.show()
