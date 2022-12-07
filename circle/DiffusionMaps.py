# %%
import numpy as np
from numpy.linalg import eig as eig
# import scipy
# import scipy.sparse
# import scipy.sparse.linalg.eigsh as eigsh
# import matplotlib.pyplot as plt

# Parameters
alpha = 1                                   # rotation frequency
dt = 2 * np.pi / alpha / 100 / np.sqrt(2)   # sampling interval 
Train_size = 5000                           # number of training datapoints
Test_size  = 1000                           # number of test datapoints
l = 10                                      # number of basis functions
epsilon = 0.05                              # RBF bandwidth parameter
q = 500                                     # number of forecast timesteps
alphaDM = 1


# Generate training and test data
X = lambda s: [np.cos(s), np.sin(s)] 
Y = lambda s: np.sin(s)

Train_t = list(range(Train_size) * dt) 
Train_omega = alpha * Train_t
Train_x = np.array(X(Train_omega))                  # covariate training data
Train_y = np.array([Y(Train_omega)]).T                # response training data

Test_t = list(range(Test_size) * dt)
Test_omega = alpha * Test_t + 1 /np.sqrt(2);
Test_x = np.array(X(Test_omega))                # covariate test data             
Test_y = np.array([Y(Test_omega)]).T              # response test data


def rbf(u,e):
    return np.exp(-u / e**2)

# Distance matrix function
# Given two clouds of points in nD-dimensional space
# represented by the  arrays x_1 and x_2, respectively of size [nD, nX1] and [nD, nX2]
# y = dist_matrix(x_1, x_2) returns the distance array y of size [nX1, nX2] such that 
# y(i, j) = norm(x_1(:, i) - x_2(:, j))^2
def dist_matrix(x_1,x_2):
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    y = -2 * np.matmul(np.conj(x_1).T, x_2)
    w_1 = np.sum(np.power(x_1, 2), axis = 0)
    y = y + w_1
    w_2 = np.sum(np.power(x_2, 2), axis = 0)
    y = y + w_2
    return y

def Kop(k, x_2, f):
    g = lambda x_1: np.matmul(k(x_1, x_2), f) / f.shape[0]
    return g

# Right Normalization
def rNormalize(k, x_3, alpha):
    r = Kop(k, x_3, np.ones((x_3.shape[1], 1)))
    p = lambda x_1, x_2: np.divide(k(x_1, x_2), (np.power(r(x_2), alpha)).T)
    return p
                        
# Left Normalization
def lNormalize(k, x_3, alpha):
    l = Kop(k, x_3, np.ones((x_3.shape[1], 1)))
    p = lambda x_1, x_2: np.divide(k(x_1, x_2), np.power(l(x_1), alpha))
    return p

# Diffusion maps normalization
def dmNormalize(k, x_3, alpha):
    r = rNormalize(k, x_3, alpha)
    p = lNormalize(r, x_3, alphaDM)
    return p
    
def Uop(f):
    g = np.zeros((f.shape))
    for i in range(0, g.shape[0]-2):
        g[i] = f[i+1]
    return g 
    
def Uorb(f, col):
    g = np.zeros((col, f.shape[0]))
    g[0,:] = f[:,0]
    for i in range(1, col-1):
        g[i,:] = Uop(g[i-1,:])
    return g
    
 
# Build kernel
eta = lambda u: rbf(u, epsilon)                        # shape function 
w = lambda x_1, x_2: eta(dist_matrix(x_1,x_2))         # unormalized kernel function
p = dmNormalize(w, Train_x, alphaDM)                   # diffusion maps kernel
kappa = lambda x_1, x_2: p(x_1,x_2)  

# Build kernel integral operator from L^2 to L^2
# Compute eigenfunctions
G = kappa(Train_x, Train_x) / Train_size
eigenvalues, eigenvectors = eig(G) 
index = eigenvalues.argsort()[::-1][:l]
lamb = eigenvalues[index]
phi = eigenvectors[:,index]
lamb = np.conj(np.diag(lamb)).T
phi = np.sqrt(Train_size) * phi

# Compute expansion coefficients of the response variable in the phi basis
y_Q = Uorb(Train_y, q)
y_Hat = np.matmul(y_Q, phi) / Train_size

# Build kernel integral operator from L^2 to continuous functions
K = lambda f: Kop(kappa, Train_x, f)

# Create continuous extensions of the phi basis vectors 
kPhi = K(phi)
varphi = lambda x: np.divide(kPhi(x), lamb)

# Compute target function
z = lambda x: np.matmul(varphi(x), np.conj(y_Hat).T)
# %%
