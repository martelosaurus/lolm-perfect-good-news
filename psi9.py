import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad, solve_ivp
from scipy.optimize import root_scalar, root
from psi_para import *
from psi_func import *
from psi_mod1 import *
from psi_eqlm import *
import psi_plot

sol = root(findbif,np.array([2.,1.]))
T0, t0 = sol.x

sol = root_scalar(lambda T: q(0.,T)-Q,x0=T0,x1=2*T0)
if np.isnan(sol.root):
    T1 = np.inf
else:  
    T1 = sol.root

# solve for equilibrium T_star
if True:
    if np.isinf(T1):
        T_opt = root_scalar(G2,x0=1.5*T0,x1=2.*T0).root
    else:
        T_opt = root_scalar(G2,bracket=((1.+opttol)*T0,(1.-opttol)*T1)).root
else:
    T_opt = T1

# plot
psi_plot.plottT(T_max,'demo' + version)
psi_plot.plotqG(T_opt,'demo' + version)
