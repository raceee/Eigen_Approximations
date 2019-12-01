import power_methods as pm
import numpy as np


A = np.array([[4,-1,1],[-1,3,-2],[1,-2,3]])
eigvalue = 6
eigvec = np.array([1,-1,1])
x = np.array([1,-1,1])


pm.wielandt(A.shape, A, eigvalue, eigvec, x)