import numpy as np
import matplotlib.pyplot as plt
from psi_para import *
from psi_func import *

def plotTt(Tt_max,fname):
    """ plots levels sets of q, Gamma in T-t space:
            Q = q(t_hat,T)
            1 = Gamma(T,t_hat,T)
    """

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
    plt.contour(T_mat,t_mat,q(t_mat,T_mat),colors="black",alpha=.1)
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad, solve_ivp
from scipy.optimize import root_scalar, root
from psi_para import *
from psi_func import *
from psi_mod1 import *
from psi_eqlm import *

def plotqG(T,fname):

    # time vectors
    t0_vec = np.linspace(0.,tau(T),N) 
    t1_vec = np.linspace(tau(T),T,N) 
    t2_vec = np.linspace(T,tau(T)+T,N) 

    # belief vectors
    yy1_vec = np.zeros(N)
    qG0_vec = Q*np.ones(N)
    GG1_vec = np.zeros(N)
    qq1_vec = np.zeros(N)
    qG2_vec = np.ones(N)
    GG21_vec = G2(t1_vec,T)

    # t-vec
    t_vec = np.hstack((t0_vec,t1_vec,t2_vec))

    # compute the beliefs
    for j in range(0,N):		
        qq1_vec[j] = q(t1_vec[j],T)
        yy1_vec[j] = g(t1_vec[j])/(1.-g(t1_vec[j])) 

    q_vec = np.hstack((Q*np.ones(N),qq1_vec,qG2_vec))
    G_vec = np.hstack((np.zeros(N),np.ones(N)-GG21_vec*np.exp(b*t1_vec),qG2_vec))

    # compute the 'bid-ask spread'
    ell_vec = 1.  

    # plotting
    fig, ax1 = plt.subplots()

    plt.xticks([0.,tau(T_opt),T_opt],['$0$','$\\tau(T^{*})$','$T^{*}$'])
    plt.yticks([Q,1.],['$Q$','$1$'])
    plt.grid()

    ax1.set_xlabel('time')
    ax1.set_ylabel('market beliefs',color=rcol)
    ax1.plot(t_vec,q_vec,color=rcol)
    ax1.tick_params(axis='y',labelcolor=rcol)

    ax2 = ax1.twinx()  

    plt.yticks([0,1.],['$0$','$1$'])
    plt.grid()
    ax2.set_ylabel('volume',color=bcol)  
    ax2.plot(t_vec,G_vec,color=bcol)
    ax2.tick_params(axis='y',labelcolor=bcol)

    plt.title(titstr)
    plt.savefig(fname + 'qG.pdf')
    plt.show()
