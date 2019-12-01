import numpy as np
from numpy import linalg as LA



def power_method(shape, mat_A, vec_x, tolerance=0.000001, iterations=100):
    # ADVANTAGES
    # Simple, easy to implement.
    # The eigenvector corresponding to the dominant(i.e., maximum) eigenvalue is generated at the same time.
    # The inverse power method solves for the minimal eigenvalue / eigenvector pair.
    #
    # DISADVANTAGES
    # The
    # power
    # method
    # provides
    # only
    # one
    # eigenvalue / eigenvector
    # pair, which
    # corresponds
    # to
    # the “maximum” eigenvalue.
    #
    # Some
    # modifications
    # must
    # be
    # implemented
    # to
    # find
    # other
    # eigenvalues, e.g., shift
    # method, deflation, etc.
    #
    # Also,
    # if the dominant eigenvalue is not sufficiently larger than others, a large number of iterations are required.

    k = 1
    xp = LA.norm(vec_x, np.inf)
    yp = 0
    vec_x = np.divide(vec_x, xp)
    # print(vec_x)
    while k <= iterations:
        y = np.dot(mat_A, vec_x)
        mu = yp
        yp = LA.norm(y, np.inf)
        if yp == 0:
            return "Eigenvector: {} A has the eigenvalue 0, select a new vector x and restart".format(vec_x)
        Err = np.amax(np.subtract(vec_x,np.divide(y,yp)))
        vec_x = np.divide(y,yp)
        if Err < tolerance:
            return mu, vec_x, k
        # print(mu, " ", vec_x)
        k+=1


def aikens_power_method(shape, mat_A, vec_x, tolerance=0.000001, iterations=100):
    k = 1
    mu_0 = 0
    mu_1 = 0
    xp = LA.norm(vec_x, np.inf)
    vec_x = np.divide(vec_x, xp)
    yp = 0
    while k <= iterations:
        y = np.matmul(mat_A, vec_x)
        mu = yp
        mu_hat = mu_0 - np.divide(((mu_1 - mu_0) ** 2), (mu - (2*mu_1) + mu_0))
        yp = LA.norm(y, np.inf)
        print("Eigenvector {} \n mu {} \n mu_hat {} \n \n".format(vec_x, mu, mu_hat))
        if yp == 0:
            return "EigenVector {}   A has the eigenvalue 0, selelct a new vector x and restart".format(vec_x)
        err = LA.norm(np.subtract(vec_x, np.divide(y,yp)), np.inf)
        vec_x = np.divide(y, yp)

        # print("Eigenvector {} \n mu {} \n mu_hat {} \n \n".format(vec_x, mu, mu_hat))
        if err < tolerance and k >= 4:
            return mu_hat, vec_x, k
        k += 1
        mu_0 = mu_1
        mu_1 = mu


def symmetric_method(shape, mat_A, vec_x, tolerance=0.00001, iterations=100):
    k = 1
    #print(vec_x)
    vec_x = np.divide(vec_x,LA.norm(vec_x, 2))
    #print(vec_x)
    while k <= iterations:
        y = np.matmul(mat_A, vec_x) #this y vector is correct
        #print("Y: ",y)
        mu = np.matmul(vec_x,y)
        print("Eigenvector {} \n mu {} \n y {} \n \n".format(vec_x, mu, y))
        if LA.norm(y, 2) == 0:
            return "Eigenvector {}. A has eigenvalue 0, select new vector X and restart".format(vec_x)
        err = LA.norm(np.subtract(vec_x, np.divide(y,LA.norm(y,2))))
        #print(err)
        vec_x = np.divide(y, LA.norm(y, 2))

        if err < tolerance:

            return mu, vec_x, k
        #print(mu, " ", vec_x)
        k += 1


def inverse_power_method(shape, mat_A, vec_x, tolerance=0.0001, iterations=100):
    q = np.divide(np.matmul(vec_x, np.matmul(mat_A,vec_x)), np.matmul(vec_x, vec_x))
    k = 1
    xp = LA.norm(vec_x, np.inf)
    err = 0
    vec_x = np.divide(vec_x, xp)
    yp = 1
    p = 0
    while k <= iterations:
        matty = np.subtract(mat_A, q * np.identity(shape[0], dtype='float'))
        y = LA.solve(matty, vec_x)
        # print("y",y)
        mu = y[p]
        yp = LA.norm(y, np.inf)
        for j in range(len(y)):
            if abs(y[j]) == yp:
                p = j
                break
        err = LA.norm(np.subtract(vec_x, np.divide(y, y[p])), np.inf)
        vec_x = np.divide(y, y[p])
        if err < tolerance:
            mu = np.add(np.divide(1, mu), q)
            return mu, vec_x, k
        k += 1

def wielandt(shape, mat_A, approx_eig, eigvec, vec_x, tolerance=0.000001, iterations=100): #shape is tuple in this function
    vi = np.amax(eigvec)
    i = np.where(eigvec == vi)
    i = i[0][0]
    B = np.empty((shape[0], shape[0]), dtype='float')
    if i != 1:
        for k in range(i):
            for j in range(i):
                B[k][j] = mat_A[k][j] - (eigvec[k] / vi) * mat_A[i][j]

    if i != 1 and i != shape[0]:
        for k in range(i, shape[0] - 1):
            for j in range(i):
                B[k][j] = mat_A[k+1][j] - (eigvec[k+1]/vi) * mat_A[i][j]
                B[j][k] = mat_A[j][k+1] - (eigvec[j]/vi) * mat_A[i][j+1]

    if i != shape[0]:
        for k in range(i, shape[0] - 1):
            for j in range(i, shape[0] - 1):
                B[k][j] = mat_A[k+1][j+1] - eigvec[k+1]/vi * mat_A[i][j+1]
    mu_2, w, k = power_method(shape, B, vec_x)
    w = list(w)
    w.insert(0, w.pop())
    w = np.asarray(w)
    x = (1 / approx_eig) * np.asarray([mat_A[i][m] for m in range(shape[0])])
    temp = (np.matmul(x, w))
    eigvectwo = (mu_2 - approx_eig) * w + (approx_eig * temp * eigvec)

    return mu_2, eigvectwo

