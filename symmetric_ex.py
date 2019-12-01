import power_methods as pm
import numpy as np


A = np.array([[4,-1,1],[-1,3,-2],[1,-2,3]])
x = np.array([1,0,0])

print(pm.symmetric_method(A.shape[0], A, x))