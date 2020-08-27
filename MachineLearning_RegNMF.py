'''
author: Nikolaos Mparoutis

RegNMF (Regulated non matrix factorization)
Is an alternative solution for the generalized PCA and regularized PCA. 
This keeps  the non negative numbers of the input array and has easier interpretable components.
---Returns---
The non negative tables W (d x k) and C (k X N) fully representing the X.
C   :   matrix of components
W   :   is the base
---Inputs---
X   :   non negative table dxN = (500 x 1000). 
        the input array for decomposition
k   :   integer 
        the number of components
lambda_ :   float 
            the regularization parameter
epsilon :   float
            the lowest threshold for the termination
'''

def gradientDescent_C(X, W, C):
    W_transp = np.transpose(W)
    nom = np.dot(W_transp,X)
    denom = np.linalg.multi_dot([W_transp, W, C])
    div = np.divide(nom, denom)
    C = np.multiply(C, div)
    return C


def gradientDescent_W(X, W, C):
    C_transp = np.transpose(C)
    nom = np.dot(X, C_transp)
    denom = np.linalg.multi_dot([W, C, C_transp])
    div = np.divide(nom, denom)
    W = np.multiply(W, div)
    return W


def error_(X, W, C):
    dist = np.linalg.norm(np.subtract(X, np.dot(W, C)))
    X_norm_2 = np.linalg.norm(X)
    error = np.divide(dist, X_norm_2)
    return error


def RegNMF(X, k, lambda_, epsilon):
    D = 500
    N = 1000
    iters = 500
    # n = 0.001 # step
    W = abs(np.random.rand(D, k))     # random initializions on W, C
    C = abs(np.random.rand(k, N))
    for e in epsilon:
        for t in range(1, iters):
            W_prev = W
            C_prev = C
            C = gradientDescent_C(X, W, C)
            W = gradientDescent_W(X, W, C)
            error_t_prev = error_(X, W_prev, C_prev)
            error_t = error_(X, W, C)
            # print("error_t : ", error_t)
            # print("error_t_prev : ", error_t_prev)

            if abs(np.subtract(error_t, error_t_prev)) < e:
                yield W, C, t, e
                break

X = abs(np.random.rand(500, 1000))
k = 10
lambda_ = 0.1
epsilon = [0.01, 0.001, 0.0001]

gen = RegNMF(X, k, lambda_, epsilon)
for W, C, t, e in gen:
    print("W : ", W)
    print("C : ", C)
    print("Run for {} iterations and epsilon {} \n".format(t, e))

