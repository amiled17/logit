import gmpy2
from gmpy2 import mpz, mpq, mpfr, mpc
import numpy as np
import pandas as pd


gmpy2.get_context().precision = 1100

def sigmoid(x, t_1, t_2): 
	z = t_1*x + t_2
	return mpfr(gmpy2.exp(-z) / (1.0 + gmpy2.exp(-z)))


def logLikelihood(x,y, t_1, t_2): 
	psigmoid = [sigmoid(i, t_1, t_2) for i in x] 
	u = [q*gmpy2.log(p) + (1 - q)*gmpy2.log(1 - p) for (p,q) in zip(psigmoid,y)]
	return gmpy2.fsum(u)

def gradient(x, y, t_1, t_2):
	psigmoid = [sigmoid(i, t_1, t_2) for i in x]
	u = [q-p for (p,q) in zip(psigmoid,y)]
	v = [t*k for (t,k) in zip(u,x)] 
	return 	np.array([[gmpy2.fsum(v), gmpy2.fsum(u)]])

def hessian(x, y, t_1, t_2):                                                          
	psigmoid = [sigmoid(i, t_1, t_2) for i in x]
	u = [p*(1 - p)*q*q for (p,q) in zip(psigmoid,x)]
	d1 = gmpy2.fsum(u)
                                       
	v = [p*(1 - p)*q for (p,q) in zip(psigmoid,x)]
	d2 = gmpy2.fsum(v)
                
	w = [p*(1 - p) for p in psigmoid]   	
	d3 = gmpy2.fsum(w)    
                  
	H = np.array([[d1, d2],[d2, d3]])                                           
	return H

def invHessian(x, y, t_1, t_2):
	hess = hessian(x, y, t_1, t_2)
	a = mpfr(hess[0][0])
	b = mpfr(hess[0][1])
	c = mpfr(hess[1][0])
	d = mpfr(hess[1][1])
	det = mpfr(a*d - b*c)
	H_inv = np.ones((2, 2), dtype=object)
	q = (mpfr(1.0)/det)
	H_inv[0][0] = gmpy2.mul(q,d)
	H_inv[0][1] = gmpy2.mul(-q,b)
	H_inv[1][0] = gmpy2.mul(-q,c)
	H_inv[1][1] = gmpy2.mul(q,a)
	return H_inv

def newton(x,y, nsteps = 15, tol = 1e-10):

	t_1 = mpfr(15.1)
	t_2 = mpfr(-.4)
	delta = 1 + tol
	l = logLikelihood(x,y, t_1, t_2)
	i = 0

	while abs(delta) > tol and i < nsteps:
		i += 1
		g = gradient(x, y, t_1, t_2)
		H_inv = invHessian(x, y, t_1, t_2)

		dt_1 = H_inv[0][0]*g[0][0] + H_inv[0][1]*g[0][1]
		dt_2 = H_inv[1][0]*g[0][0] + H_inv[1][1]*g[0][1]

		t_1 += dt_1
		t_2 += dt_2
	
		lNew = logLikelihood(x,y, t_1, t_2)
		delta = l - lNew
		l = lNew

	return [np.array([t_1, t_2])]



data = pd.read_csv("data.csv") 
x = data['col1'].tolist()
y = data['col2'].tolist()
un= newton(x,y, 15,  1e-10)
print(un)


