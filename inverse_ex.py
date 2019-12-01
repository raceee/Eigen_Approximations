import power_methods as pm
import numpy as np



A = np.array([[-4,14,0],[-5,13,0],[-1,0,2]])
x = np.array([1,1,1])

print(pm.inverse_power_method(A.shape, A, x))