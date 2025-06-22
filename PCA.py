# lets learn PCA
# Principle component analysis 
# some topic without code 
# preserving the Varience 
# Principle Component 
# in python  principle component is find by the standard matrix factorization technique called Singular Value Decomposition (SVD)
# for this we have the svd() function in python  in numpy 

import numpy as np

x_centered=X-X.mean(axis=0)
U,s,vt=np.linalg.svd(x_centered)
c1=vt.T[:,0]
c2=vt.T[:,1]
