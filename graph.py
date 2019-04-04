import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gmpy2
from gmpy2 import mpz, mpq, mpfr, mpc
import numpy as np
import pandas as pd

gmpy2.get_context().precision = 50

def sigmoid(x, t_1, t_2): 
	z = t_1*x + t_2
	return mpfr(gmpy2.exp(-z) / (1.0 + gmpy2.exp(-z)))


def log_likelihood(x,y, t_1, t_2): 
	sigmoidP = [sigmoid(i, t_1, t_2) for i in x] 
	u = [q*gmpy2.log(p) + (1 - q)*gmpy2.log(1 - p) for p,q in zip(sigmoidP,y)]
	return gmpy2.fsum(u)



fig = plt.figure()
ax = fig.gca(projection='3d')

t_1 = np.arange(15, 15.1, 0.001)
t_2 = np.arange(-.41, -.4, 0.0001)

T_1, T_2 = np.meshgrid(t_1,t_2, indexing = 'ij')

data = pd.read_csv("data.csv") 
x = data['col1'].tolist()
y = data['col2'].tolist()

zs = np.ones((100, 100))
zs = np.array([float(log_likelihood(x,y,i,j)) for i,j in zip( np.ravel(T_1),np.ravel(T_2))])
Z = zs.reshape(T_1.shape)


ax.plot_surface(T_1, T_2, Z, linewidth=0, antialiased=False)

ax.set_xlabel('T_1 Axis')
ax.set_ylabel('T_2 Axis')
ax.set_zlabel('Z logLikelihood')

plt.show()


