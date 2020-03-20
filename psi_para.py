import numpy as np

#plotting 
N = 200
npz = np.zeros(N)
npo = np.ones(N)
rcol = 'tab:red'
bcol = 'tab:blue'
	
# parameters
b = .1
c = .1
l = .5
r = .5
Q = .7
Y = 1.

# title string
titstr = '$\\pi = $' + str(b) + ', '
titstr += '$c = $' + str(c) + ', '
titstr += '$\\lambda = $' + str(l) + ', '
titstr += '$r = $' + str(r) + ', ' 
titstr += '$Q = $' + str(Q) + ', '
titstr += '$Y = $' + str(Y)

# synthetic parameters
C1 = r*c/(r+b)
C2 = l*Y/(r+b)

# terminal conditions
VL1 = l*Y/r-c 
VH1 = l*Y/r-r*c/(r+b)

# optimization parameters
T_max = 10.
opttol = 1.e-6
