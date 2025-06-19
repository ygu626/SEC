"""
Spectral Exterior Calculus (SEC)
Circle S1 Example
Approximations of vector fields on the cirlce
Given pushforward of tangent vectors on the circle
and determinstically sampled Monte Carlo points on the circle
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy.linalg import eig as eig
import multiprocess as mp
from scipy.integrate import quad
from scipy.integrate import solve_ivp


# Parameters
I = 20          # Inner index for eigenfunctions
J = 10           # Outer index for eigenfunctions
K = 5           # Index for gradients of eigenfunctions
n = 20          # Number of approximated tangent vectors
N = 800         # Number of Monte Carlo training data points 

epsilon = 0.1  # RBF bandwidth parameter
tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix
C2 = 1.5         # Component function parameter for vector field v
C3 = 0.5


"""
Training data set
with pushforward of vector fields v on the circle
and Embedding map F with pushforward vF
"""


# Deterministically sampled Monte Carlo training data points
def monte_carlo_points(start_pt = 0, end_pt = 2*np.pi, N = 800):
    u = np.arange(start_pt, end_pt, 2*np.pi/N)
    # subsets = np.arange(0, N+1, N/400)
    # for i in range(0, 400):
    #    start = int(subsets[i])
    #    end = int(subsets[i+1])
    #    u[start:end] = random.uniform(low = (i/400)*b, high = ((i+1)/400)*b, size = end - start)
    # random.shuffle(u)
    
    training_data = np.empty([2, N], dtype = float)
    for j in range(0, N):
            training_data[:, j] = np.array([2*np.cos(u[j]), 2*np.sin(u[j])])
    
    return u, training_data

u, training_data = monte_carlo_points()
plt.scatter(training_data[0,:], training_data[1,:])
plt.show()


# n pushforward of vector field v ("arrows") on the circle
# given points (x, y) specified by angle theat on the circle
THETA_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))
X_func = lambda theta: 2*np.cos(theta)
Y_func = lambda theta: 2*np.sin(theta)
TRAIN_X = np.array(X_func(THETA_LST))
TRAIN_Y = np.array(Y_func(THETA_LST))



TRAIN_V1 = np.empty([n, 4], dtype = float)
TRAIN_V2 = np.empty([n, 4], dtype = float)
TRAIN_V3 = np.empty([n, 4], dtype = float)

for i in range(0, n):
    TRAIN_V1[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i], TRAIN_X[i]])
    TRAIN_V2[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i] - C2*TRAIN_Y[i]*TRAIN_X[i], TRAIN_X[i] + C2*TRAIN_X[i]*TRAIN_X[i]])
    TRAIN_V3[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -np.exp(C3*TRAIN_X[i])*TRAIN_Y[i], np.exp(C3*TRAIN_X[i])*TRAIN_X[i]])

X_11, Y_11, U_11, V_11 = zip(*TRAIN_V1)
X_12, Y_12, U_12, V_12 = zip(*TRAIN_V2)
X_13, Y_13, U_13, V_13 = zip(*TRAIN_V3)


# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta: np.array([2*np.cos(theta), 2*np.sin(theta)])
v1F = lambda theta: np.array([-2*np.sin(theta), 2*np.cos(theta)])
v2F = lambda theta: np.array([-np.sin(theta) - C2*np.sin(theta)*np.cos(theta), np.cos(theta) + C2*(np.cos(theta))**2])
v3F = lambda theta: np.array([-np.exp(C3*np.cos(theta))*np.sin(theta), np.exp(C3*np.cos(theta))*np.cos(theta)])


# Component functions as part of the vector field v
h1 = lambda theta: 1 + C2*np.cos(theta)
h2 = lambda theta: np.exp(C3*np.cos(theta))                  # Jump function



"""
Functions utilized in the following program
"""


# Double and triple products of functions
def double_prod(f, g):
    def fg(x):
        return f(x) * g(x)
    return fg

def triple_prod(f, g, h):
    def fgh(x):
        return f(x) * g(x) * h(x)
    return fgh


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
    y = y + w_1[:, np.newaxis]
    w_2 = np.sum(np.power(x_2, 2), axis = 0)
    y = y + w_2
    return y


"""
Implementation of diffusion maps algorithm
Approximation of eigenvalues and eigenfunctions of the 0-Laplacian
uo to a constant scaling factor
"""


# Diffusion maps algorithm
# Normalization function q that corresponds to diagonal matrix Q
def make_normalization_func(k, x_train):
    def normalized(x):
        y = np.sum(k(x, x_train), axis = 1)
        return y
    return normalized

# Heat kernel function k
k = lambda x_1, x_2: np.exp(-dist_matrix(x_1, x_2)/(epsilon**2))

# Build kernel matrix K
# K = k(training_data, training_data)

# Normalized kernel function k_hat
def make_k_hat(k, q):
    def k_hat(x, y):
        q_x = q(x).reshape(q(x).shape[0], 1)
        q_y = q(y).reshape(1, q(y).shape[0])
        # treat qx as column vector
        k_hat_xy = np.divide(k(x, y), np.matmul(q_x, q_y))
        return k_hat_xy
    return k_hat

# Build normalized kernel matrix K_hat
q = make_normalization_func(k, training_data)
k_hat = make_k_hat(k, q)
K_hat = k_hat(training_data, training_data)
# print(K_hat[:3,:3])

# Normalization function d that corresponds to diagonal matrix D
d = make_normalization_func(k_hat, training_data)
D = d(training_data)

# Markov kernel function p
def make_p(k_hat, d):
    def p_func(x, y):
        d_x = d(x).reshape(1, d(x).shape[0])

        p_xy = np.divide(k_hat(x, y), d_x)
        return p_xy
    return p_func

# Build Markov kernel matrix P
p_func = make_p(k_hat, d)
P = p_func(training_data, training_data)

print(np.trace(P))
print(np.pi/(epsilon))
print(np.sum(P, axis = 1))


# Similarity transformation function s
def make_s(p_func, d):
    def s(x, y):
        d_x = np.power(d(x).reshape(d(x).shape[0], 1), (1/2))
        d_y = np.power(d(y).reshape(1, d(y).shape[0]), (1/2))
        
        s_xy = np.divide(np.multiply(p_func(x, y), d_x), d_y)
        return s_xy
    return s

# Build Similarity matrix S
s = make_s(p_func, d)
S = s(training_data, training_data)
# print(S[:3,:3])


# Solve eigenvalue problem for similarity matrix S
eigenvalues, eigenvectors = eig(S) 
index = eigenvalues.argsort()[::-1][:2*I+1]
Lambs = eigenvalues[index]
Phis = np.real(eigenvectors[:, index])

# Compute approximated 0-Laplacian eigengunctions
lambs = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
            lambs[i] = 4*(-np.log(np.real(Lambs[i]))/(epsilon**2)) 

# %%
print(lambs)
# %%

# Normalize eigenfunctions Phi_j
Phis_normalized = np.empty([N, 2*I+1], dtype = float)
D_sqrt = np.power(D.reshape(1, D.shape[0]), (1/2))
for j in range(0, 2*I+1):
    Phis_normalized[:, j] = np.divide(np.real(Phis[:, j]), D_sqrt)

Phis_normalized = Phis_normalized/Phis_normalized[0, 0]


# Appeoximate eigenvalues and eigenfunctions for the 0-Laplacian
def make_varphi(k, x_train, lambs, phis):
    phi_lamb = phis / lambs
    
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    
    return varphi

# Produce continuous extentions varphi_j for the eigenfunctions Phi_j
Lambs_normalized = np.power(Lambs, 4)
varphi = make_varphi(p_func, training_data, Lambs, Phis_normalized)

print(np.max(Phis_normalized[:,0]))
print(np.min(Phis_normalized[:,0]))



"""
SEC approximation
for pushforward of vector fields on the circle
"""


# Fourier coefficients F_ak pf F w.r.t. difusion maps approximated eigenvectors Phi_j
F_ak = (1/N)*np.matmul(F(u), Phis_normalized)


# Compute c_ijp coefficients
# using Monte Carlo integration
pool = mp.Pool()

def c_func(i, j, p):
    return (1/N)*np.sum(Phis_normalized[:, i]*Phis_normalized[:, j]*Phis_normalized[:, p])

c = pool.starmap(c_func, 
              [(i, j, p) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for p in range(0, 2 * I + 1)])
            
c = np.reshape(np.array(c), (2 * I + 1, 2 * I + 1, 2 * I + 1))
# print(c[:,3,3])


# Compute g_ijp Riemannian metric coefficients
# using Monte Carlo integration
g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
g_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
for p in range(0, 2*I+1):
            for i in range(0, 2*I+1):
                        for j in range(0, 2*I+1):
                                    g_coeff[p,i,j] = (lambs[i] + lambs[j] - lambs[p])/2
         
g = np.multiply(g_coeff, c)
         
# print(g[:,3,3])


# Compute G_ijpq entries for the Gram operator and its dual
# using Monte Carlo integration
G = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G = np.einsum('ipm, mjq -> ijpq', c, g, dtype = float)

G = G[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G = np.reshape(G, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))


# Perform singular value decomposition (SVD) of the Gram operator G
# and plot these singular values
u2, s2, vh = np.linalg.svd(G, full_matrices = True, compute_uv = True, hermitian = False)

sing_lst = np.arange(0, len(s2), 1, dtype = int)
plt.figure(figsize=(24, 6))
plt.scatter(sing_lst, s2, color = 'red')

plt.xticks(np.arange(0, ((2*J+1)*(2*K+1))+0.1, 1))
plt.xlabel('Indices')
plt.yticks(np.arange(0, max(s2)+0.1, 1))
plt.ylabel('Singular Values')
plt.title('Singular Values of the Gram Operator G_ijpq (descending order)')

plt.show()

# print(np.max(s2))
# print(np.min(s2))


# Teuncate singular values of G based based on a small percentage of the largest singular valuecof G
threshold = 0.01      # Threshold value for truncated SVD


# Compute dual Gram operator G* using pseudoinverse based on truncated singular values of G
G_dual = np.linalg.pinv(G, rcond = threshold)
# G_dual_mc = np.linalg.pinv(G_mc_weighted)
# %%



"""
v1: Rational rotation on the circle
"""
# %%


# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product1(Phis, u, N = 800):
    v_an = v1F(u)
    integral = (1/N)*np.sum(Phis*v_an, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func1(m):
    return monte_carlo_product1(Phis_normalized[:, m], u)


b_am1 = pool.map(b_func1, 
                [m for m in range(0, 2 * I + 1)])
b_am1 = np.array(b_am1).T


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km1 = np.einsum('ak, am -> km', F_ak, b_am1, dtype = float)

g1 = g[:, :(2*K+1), :]
eta_qlm1 = np.einsum('lqk, km -> qlm', g1, gamma_km1, dtype = float)

c1 = c[:(2*J+1), :, :]
v_hat_prime1 = np.einsum('qlm, plm -> pq', eta_qlm1, c1, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime1[:, q] = np.exp(-tau*lambs[q])*v_hat_prime1[:, q]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime1 = np.reshape(v_hat_prime1, ((2*J+1)*(2*K+1), 1))


# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat1 = np.matmul(G_dual, v_hat_prime1)
v_hat1 = np.reshape(v_hat1, (2*J+1, 2*K+1))


# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights


# Weighted g_ijp Riemannian metric coefficients
g_weighted1 = np.zeros([2*I+1, 2*K+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted1[:, j, :] = np.exp(-tau*lambs[j])*g1[:, j, :]


h_ajl1 = np.einsum('ak, ljk -> ajl', F_ak, g_weighted1, dtype = float)

# c = c[:(2*J+1), :, :]
d_jlm1 = np.einsum('ij, ilm -> jlm', v_hat1, c1, dtype = float)

p_am1 = np.einsum('ajl, jlm -> am', h_ajl1, d_jlm1, dtype = float)


W_theta_x1 = np.zeros(n, dtype = float)
W_theta_y1 = np.zeros(n, dtype = float)
vector_approx1 = np.empty([n, 4], dtype = float)

def W_x1(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am1[0, :]*varphi_xy)

def W_y1(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am1[1, :]*varphi_xy)

for i in range(0, n):
    W_theta_x1[i] = W_x1(TRAIN_X[i], TRAIN_Y[i])
    W_theta_y1[i] = W_y1(TRAIN_X[i], TRAIN_Y[i])
    vector_approx1[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x1[i], W_theta_y1[i]])

X_2, Y_2, U_2, V_2 = zip(*vector_approx1)
# %%

"""
Plot the pushfoward map F_* of the embedding F applying to v
as a quiver plot in R^2 to capture the bias in SEC approximation
using meshgrid as the training data set
"""

m1 = 12           #Square root of number of points used in quiver plot of F_*

x_train_new1 = np.linspace(-5, 5, m1)
y_train_new1 = np.linspace(-5, 5, m1)

X_TRAIN_NEW1, Y_TRAIN_NEW1 = np.meshgrid(x_train_new1, y_train_new1)

W_theta_x_new1 = np.zeros([m1, m1], dtype = float)
W_theta_y_new1 = np.zeros([m1, m1], dtype = float)


for i in range(0, m1):
    for j in range(0, m1):
        W_theta_x_new1[i, j] = W_x1(X_TRAIN_NEW1[i, j], Y_TRAIN_NEW1[i, j])
        W_theta_y_new1[i, j] = W_y1(X_TRAIN_NEW1[i, j], Y_TRAIN_NEW1[i, j])
        

U_TRAIN_NEW1 = W_theta_x_new1
V_TRAIN_NEW1 = W_theta_y_new1


# %%
"""
Solve ODEs in the SEC approximated system
and compare with the solution in the true system
"""

# %%
"""
True System
"""

# Define derivative function for the true system
def f_true1(t, y):
    dydt = [-2*np.sin(np.arctan2(y[1], y[0])), 2*np.cos(np.arctan2(y[1], y[0]))]
    return dydt

# Define time spans and initial values for the true system
tspan1 = np.linspace(0, 50, num=5000)
yinit1 = [1, 0]

# Solve ODE under the true system
sol_true1 = solve_ivp(lambda t, y: f_true1(t, y),
                     [tspan1[0], tspan1[-1]], yinit1, t_eval=tspan1, rtol=1e-8)
# %%


"""
SEC Approximated System
"""

# Define derivative function for the SEC approximated system
def f_sec1(t, y):
    dydt = [W_x1(y[0], y[1]), W_y1(y[0], y[1])]
    return dydt

# Define time spans and initial values for the SEC approximated system
tspan1 = np.linspace(0, 50, num=5000)
yinit1 = [1, 0]

# Solve ODE under the SEC approximated system
sol_sec1 = solve_ivp(lambda t, y: f_sec1(t, y),
                    [tspan1[0], tspan1[-1]], yinit1, t_eval=tspan1, rtol = 1e-12, atol = 1e-12)

# %%


"""
v2: Arc with fixed point on the circle
"""
# %%

# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product2(Phis, u, N = 800):
    v_an = v2F(u)
    integral = (1/N)*np.sum(Phis*v_an, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func2(m):
    return monte_carlo_product2(Phis_normalized[:, m], u)


b_am2 = pool.map(b_func2, 
                [m for m in range(0, 2 * I + 1)])
b_am2 = np.array(b_am2).T


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km2 = np.einsum('ak, am -> km', F_ak, b_am2, dtype = float)

g2 = g[:, :(2*K+1), :]
eta_qlm2 = np.einsum('lqk, km -> qlm', g2, gamma_km2, dtype = float)

# g2 = g[:(2*K+1), :, :]

# eta_qlm2 = np.einsum('qkl, km -> qlm', g2, gamma_km2, dtype = float)

c2 = c[:(2*J+1), :, :]
v_hat_prime2 = np.einsum('qlm, plm -> pq', eta_qlm2, c2, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime2[:, q] = np.exp(-tau*lambs[q])*v_hat_prime2[:, q]

v_hat_prime2 = np.reshape(v_hat_prime2, ((2*J+1)*(2*K+1), 1))


# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat2 = np.matmul(G_dual, v_hat_prime2)
v_hat2 = np.reshape(v_hat2, (2*J+1, 2*K+1))




# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights


# Weighted g_ijp Riemannian metric coefficients
g_weighted2 = np.zeros([2*I+1, 2*K+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted2[:, j, :] = np.exp(-tau*lambs[j])*g2[:, j, :]


# h_ajl2 = np.einsum('ak, jkl -> ajl', F_ak, g_weighted2, dtype = float)
h_ajl2 = np.einsum('ak, ljk -> ajl', F_ak, g_weighted2, dtype = float)


# c = c[:(2*J+1), :, :]
d_jlm2 = np.einsum('ij, ilm -> jlm', v_hat2, c2, dtype = float)

p_am2 = np.einsum('ajl, jlm -> am', h_ajl2, d_jlm2, dtype = float)


W_theta_x2 = np.zeros(n, dtype = float)
W_theta_y2 = np.zeros(n, dtype = float)
vector_approx2 = np.empty([n, 4], dtype = float)

def W_x2(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am2[0, :]*varphi_xy)

def W_y2(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am2[1, :]*varphi_xy)

for i in range(0, n):
    W_theta_x2[i] = W_x2(TRAIN_X[i], TRAIN_Y[i])
    W_theta_y2[i] = W_y2(TRAIN_X[i], TRAIN_Y[i])
    vector_approx2[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x2[i], W_theta_y2[i]])

X_3, Y_3, U_3, V_3 = zip(*vector_approx2)



"""
Plot the pushfoward map F_* of the embedding F applying to v
as a quiver plot in R^2 to capture the bias in SEC approximation
using meshgrid as the training data set
"""


m2 = 12           #Square root of number of points used in quiver plot of F_*

x_train_new2 = np.linspace(-5, 5, m2)
y_train_new2 = np.linspace(-5, 5, m2)

X_TRAIN_NEW2, Y_TRAIN_NEW2 = np.meshgrid(x_train_new2, y_train_new2)

W_theta_x_new2 = np.zeros([m2, m2], dtype = float)
W_theta_y_new2 = np.zeros([m2, m2], dtype = float)


for i in range(0, m2):
    for j in range(0, m2):
        W_theta_x_new2[i, j] = W_x2(X_TRAIN_NEW2[i, j], Y_TRAIN_NEW2[i, j])
        W_theta_y_new2[i, j] = W_y2(X_TRAIN_NEW2[i, j], Y_TRAIN_NEW2[i, j])
        

U_TRAIN_NEW2 = W_theta_x_new2
V_TRAIN_NEW2 = W_theta_y_new2


# Finding fixed points of the compontent functions of vF
def rootsearch(f, a, b, dx):
    x1 = a
    f1 = f(a)
    x2 = a + dx
    f2 = f(x2)
    
    while f1*f2 > 0.0:
        if x1 >= b:
            return None,None
        
        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f(x2)
        
    return x1,x2


def bisect(f, x1, x2, switch=0, epsilon=1e-8):
    root_lst = []
    f1 = f(x1)
    
    if f1 == 0.0:
        return x1
    f2 = f(x2)
    
    if f2 == 0.0:
        return x2
    
    if f1*f2 > 0.0:
        print('Root is not bracketed')
        return None
    
    l = int(np.ceil(np.log(abs(x2 - x1)/epsilon)/np.log(2.0)))
    
    for i in range(l):
        x3 = 0.5*(x1 + x2)
        f3 = f(x3)
        
        if (switch == 1) and (abs(f3) >abs(f1)) and (abs(f3) > abs(f2)):
            return None
        
        if f3 == 0.0:
            return x3
        
        if f2*f3 < 0.0:
            x1 = x3
            f1 = f3
        
        else:
            x2 =x3
            f2 = f3
            
    root_lst.append((x1 + x2)/2.0)
    # return (x1 + x2)/2.0
    return root_lst
      


def roots(f, a, b, eps = 1e-6):
    # print ('The roots on the interval [%f, %f] are:' % (a,b))
    
    while 1:
        x1,x2 = rootsearch(f, a, b, eps)
        
        if x1 != None:
            a = x2
            root = bisect(f, x1, x2, 1)
            
            if root != None:
                pass
                # root_lst = [round(root,-int(math.log(eps, 10)))]
                # print(round(root,-int(math.log(eps, 10)))
        else:
            # print ('\nDone')
            break
    
    return root


v2F_fixed_points = roots(h1, 0, 2*np.pi)

for i in range(0, len(v2F_fixed_points)):
    v2F_fixed_points.append(np.pi + (np.pi - v2F_fixed_points[i]))

print(v2F_fixed_points)


"""
Solve ODEs in the SEC approximated system
and compare with the solution in the true system
"""



"""
True System
"""

# Define derivative function for the true system
def f_true2(t, y):
    dydt = [-np.sin(np.arctan2(y[1], y[0])) - C2*np.sin(np.arctan2(y[1], y[0]))*np.cos(np.arctan2(y[1], y[0])), np.cos(np.arctan2(y[1], y[0])) + C2*(np.cos(np.arctan2(y[1], y[0]))**2)]
    return dydt

# Define time spans and initial values for the true system
tspan2 = np.linspace(0, 50, num=5000)
yinit2 = [1, 0]

# Solve ODE under the true system
sol_true2 = solve_ivp(lambda t, y: f_true2(t, y),
                     [tspan2[0], tspan2[-1]], yinit2, t_eval=tspan2, rtol=1e-8)


"""
SEC Approximated System
"""


# Define derivative function for the SEC approximated system
def f_sec2(t, y):
    dydt = [W_x2(y[0], y[1]), W_y2(y[0], y[1])]
    return dydt


# Define time spans and initial values for the SEC approximated system
tspan2 = np.linspace(0, 50, num=5000)
yinit2 = [1, 0]

# Solve ODE under the SEC approximated system
sol_sec2 = solve_ivp(f_sec2,
                    [tspan2[0], tspan2[-1]], yinit2, t_eval=tspan2, rtol=1e-12, atol = 1e-12)

# %%


"""
v3: Variable speed rotation on the circle
"""
# %%

# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product3(Phis, u, N = 800):
    v_an = v3F(u)
    integral = (1/N)*np.sum(Phis*v_an, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func3(m):
    return monte_carlo_product3(Phis_normalized[:, m], u)


b_am3 = pool.map(b_func3, 
                [m for m in range(0, 2 * I + 1)])
b_am3 = np.array(b_am3).T


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km3 = np.einsum('ak, am -> km', F_ak, b_am3, dtype = float)

g3 = g[:, :(2*K+1), :]
eta_qlm3 = np.einsum('lqk, km -> qlm', g3, gamma_km3, dtype = float)

c3 = c[:(2*J+1), :, :]
v_hat_prime3 = np.einsum('qlm, plm -> pq', eta_qlm3, c3, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime3[:, q] = np.exp(-tau*lambs[q])*v_hat_prime3[:, q]

v_hat_prime3 = np.reshape(v_hat_prime3, ((2*J+1)*(2*K+1), 1))


# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat3 = np.matmul(G_dual, v_hat_prime3)
v_hat3 = np.reshape(v_hat3, (2*J+1, 2*K+1))


# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights


# Weighted g_ijp Riemannian metric coefficients
g_weighted3 = np.zeros([2*I+1, 2*K+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted3[:, j, :] = np.exp(-tau*lambs[j])*g3[:, j, :]


h_ajl3 = np.einsum('ak, ljk -> ajl', F_ak, g_weighted3, dtype = float)

d_jlm3 = np.einsum('ij, ilm -> jlm', v_hat3, c3, dtype = float)

p_am3 = np.einsum('ajl, jlm -> am', h_ajl3, d_jlm3, dtype = float)


W_theta_x3 = np.zeros(n, dtype = float)
W_theta_y3 = np.zeros(n, dtype = float)
vector_approx3 = np.empty([n, 4], dtype = float)

def W_x3(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am3[0, :]*varphi_xy)

def W_y3(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am3[1, :]*varphi_xy)

for i in range(0, n):
    W_theta_x3[i] = W_x3(TRAIN_X[i], TRAIN_Y[i])
    W_theta_y3[i] = W_y3(TRAIN_X[i], TRAIN_Y[i])
    vector_approx3[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x3[i], W_theta_y3[i]])

X_4, Y_4, U_4, V_4 = zip(*vector_approx3)


"""
Plot the pushfoward map F_* of the embedding F applying to v
as a quiver plot in R^2 to capture the bias in SEC approximation
using meshgrid as the training data set
"""

m3 = 12           #Square root of number of points used in quiver plot of F_*

x_train_new3 = np.linspace(-5, 5, m3)
y_train_new3 = np.linspace(-5, 5, m3)

X_TRAIN_NEW3, Y_TRAIN_NEW3 = np.meshgrid(x_train_new3, y_train_new3)

W_theta_x_new3 = np.zeros([m3, m3], dtype = float)
W_theta_y_new3 = np.zeros([m3, m3], dtype = float)


for i in range(0, m3):
    for j in range(0, m3):
        W_theta_x_new3[i, j] = W_x3(X_TRAIN_NEW3[i, j], Y_TRAIN_NEW3[i, j])
        W_theta_y_new3[i, j] = W_y3(X_TRAIN_NEW3[i, j], Y_TRAIN_NEW3[i, j])
        

U_TRAIN_NEW3 = W_theta_x_new3
V_TRAIN_NEW3 = W_theta_y_new3


"""
Solve ODEs in the SEC approximated system
and compare with the solution in the true system
"""


"""
True System
"""

# Define derivative function for the true system
def f_true3(t, y):
    dydt = [-np.exp(C3*np.cos(np.arctan2(y[1], y[0])))*np.sin(np.arctan2(y[1], y[0])), np.exp(C3*np.cos(np.arctan2(y[1], y[0])))*np.cos(np.arctan2(y[1], y[0]))]
    return dydt

# Define time spans and initial values for the true system
tspan3 = np.linspace(0, 50, num=5000)
yinit3 = [1, 0]

# Solve ODE under the true system
sol_true3 = solve_ivp(lambda t, y: f_true3(t, y),
                     [tspan3[0], tspan3[-1]], yinit3, t_eval=tspan3, rtol=1e-8)



"""
SEC Approximated System
"""

# Define derivative function for the SEC approximated system
def f_sec3(t, y):
    dydt = [W_x3(y[0], y[1]), W_y3(y[0], y[1])]
    return dydt

# Define time spans and initial values for the SEC approximated system
tspan3 = np.linspace(0, 50, num=5000)
yinit3 = [1, 0]

# Solve ODE under the SEC approximated system
sol_sec3 = solve_ivp(lambda t, y: f_sec3(t, y),
                    [tspan3[0], tspan3[-1]], yinit3, t_eval=tspan3, rtol = 1e-12, atol = 1e-12)

# %%



# %%
ana_dir_coords = np.vstack([TRAIN_X, TRAIN_Y, v1F(THETA_LST)])

vec_ana = ana_dir_coords[2:4, :]
vec_sec = vector_approx1[:, 2:4].T

ana_norm = np.sqrt(np.sum(np.power(vec_ana, 2), axis = 0))
sec_norm = np.sqrt(np.sum(np.power(vec_sec, 2), axis = 0))

norm_ratio = ana_norm/sec_norm


rss = np.sum(np.power((vec_ana - vec_sec), 2))

vec_bar = np.mean(vec_ana, axis = 1)
tss = np.sum(np.power(vec_ana, 2))


R_squared = 1 - rss/tss


print(R_squared)
# %%


"""
Tangent vectors plot on tne ckrcle
"""
# %%

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (16, 12), layout="constrained")
# fig.suptitle(r"Tangent Vectors on $S^1$", fontsize = 36)
# fig.tight_layout(pad = 7.0)
t = np.linspace(0, 2*np.pi, 100000)



ax1.set_title(r"(a) $V_1$", fontsize = 32)
ax1.set_xlim([-2,2])
ax1.set_ylim([-2,2])

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-2, -1, 0, 1, 2])
ax1.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax1.set_ylabel("y", rotation = 0,  fontsize = 20)
ax1.set_yticks([-2, -1, 0, 1, 2])
ax1.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax1.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax1.quiver(X_11, Y_11, U_11, V_11, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'red')

ax1.set_aspect('equal', adjustable='box')



ax2.set_title(r"(b) $V_2$", fontsize = 32)
ax2.set_xlim([-3,3])
ax2.set_ylim([-3,3])

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax2.set_ylabel("y", rotation = 0, fontsize = 20)
ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax2.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax2.quiver(X_12, Y_12, U_12, V_12, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'red')

ax2.set_aspect('equal', adjustable='box')



ax3.set_title(r"(c) $V_3$", fontsize = 32)
ax3.set_xlim([-2,2])
ax3.set_ylim([-2,2])

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-2, -1, 0, 1, 2])
ax3.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax3.set_ylabel("y", rotation = 0, fontsize = 20)
ax3.set_yticks([-2, -1, 0, 1, 2])
ax3.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax3.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax3.quiver(X_13, Y_13, U_13, V_13, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'red')

ax3.set_aspect('equal', adjustable='box')



ax4.set_title(r"(d) $V_1^{(L)}$", fontsize = 32)
ax4.set_xlim([-2,2])
ax4.set_ylim([-2,2])

ax4.set_xlabel("x", fontsize = 20)
ax4.set_xticks([-2, -1, 0, 1, 2])
ax4.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax4.set_ylabel("y", rotation = 0, fontsize = 20)
ax4.set_yticks([-2, -1, 0, 1, 2])
ax4.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax4.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax4.quiver(X_2, Y_2, U_2, V_2, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax4.set_aspect('equal', adjustable='box')



ax5.set_title(r"(e) $V_2^{(L)}$", fontsize = 32)
ax5.set_xlim([-3,3])
ax5.set_ylim([-3,3])

ax5.set_xlabel("x", fontsize = 20)
ax5.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax5.set_ylabel("y", rotation = 0, fontsize = 20)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax5.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax5.quiver(X_3, Y_3, U_3, V_3, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax5.set_aspect('equal', adjustable='box')



ax6.set_title(r"(f) $V_3^{(L)}$", fontsize = 32)
ax6.set_xlim([-2,2])
ax6.set_ylim([-2,2])

ax6.set_xlabel("x", fontsize = 20)
ax6.set_xticks([-2, -1, 0, 1, 2])
ax6.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax6.set_ylabel("y", rotation = 0, fontsize = 20)
ax6.set_yticks([-2, -1, 0, 1, 2])
ax6.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax6.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax6.quiver(X_4, Y_4, U_4, V_4, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax6.set_aspect('equal', adjustable='box')



plt.show()
# %%


"""
POSTER
Tangent vectors plot on tne ckrcle
"""
# %%

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (16, 12), layout="constrained")
# fig.suptitle(r"Tangent Vectors on $S^1$ Given by $\vec{V}_{\mathrm{SEC}}:\mathbb{R}^2\to\mathbb{R}^2, \vec{V}_{\mathrm{SEC}} = F_* \tilde{V}_{\mathrm{SEC}}$", fontsize = 36)
# fig.tight_layout(pad = 7.0)
# fig.text(-0.1, 0.75, 'True Vector Field', ha='center', va='center', fontsize = 26)
# fig.text(-0.1, 0.27, 'SEC Approximation', ha='center', va='center', fontsize = 26)

t = np.linspace(0, 2*np.pi, 100000)



ax1.set_title("Rotation \n (a)", fontsize = 32)
ax1.set_xlim([-2,2])
ax1.set_ylim([-2,2])

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-2, -1, 0, 1, 2])
ax1.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax1.set_ylabel("y", rotation = 0,  fontsize = 20)
ax1.set_yticks([-2, -1, 0, 1, 2])
ax1.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax1.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax1.quiver(X_11, Y_11, U_11, V_11, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'red')

ax1.set_aspect('equal', adjustable='box')



ax2.set_title("Arc with Fixed Point \n (c)", fontsize = 32)
ax2.set_xlim([-3,3])
ax2.set_ylim([-3,3])

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax2.set_ylabel("y", rotation = 0, fontsize = 20)
ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax2.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax2.quiver(X_12, Y_12, U_12, V_12, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'red')

ax2.set_aspect('equal', adjustable='box')



ax3.set_title("Variable Speed Rotation \n (e)", fontsize = 32)
ax3.set_xlim([-2,2])
ax3.set_ylim([-2,2])

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-2, -1, 0, 1, 2])
ax3.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax3.set_ylabel("y", rotation = 0, fontsize = 20)
ax3.set_yticks([-2, -1, 0, 1, 2])
ax3.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax3.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax3.quiver(X_13, Y_13, U_13, V_13, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'red')

ax3.set_aspect('equal', adjustable='box')



ax4.set_title("\n (b)", fontsize = 32)
ax4.set_xlim([-2,2])
ax4.set_ylim([-2,2])

ax4.set_xlabel("x \n \n \n", fontsize = 20)
ax4.set_xticks([-2, -1, 0, 1, 2])
ax4.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax4.set_ylabel("y", rotation = 0, fontsize = 20)
ax4.set_yticks([-2, -1, 0, 1, 2])
ax4.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax4.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax4.quiver(X_2, Y_2, U_2, V_2, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax4.set_aspect('equal', adjustable='box')



ax5.set_title("\n (d)", fontsize = 32)
ax5.set_xlim([-3,3])
ax5.set_ylim([-3,3])

ax5.set_xlabel("x \n \n \n", fontsize = 20)
ax5.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax5.set_ylabel("y", rotation = 0, fontsize = 20)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 20)

ax5.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax5.quiver(X_3, Y_3, U_3, V_3, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax5.set_aspect('equal', adjustable='box')



ax6.set_title("\n (f)", fontsize = 32)
ax6.set_xlim([-2,2])
ax6.set_ylim([-2,2])

ax6.set_xlabel("x \n \n \n", fontsize = 20)
ax6.set_xticks([-2, -1, 0, 1, 2])
ax6.set_xticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax6.set_ylabel("y", rotation = 0, fontsize = 20)
ax6.set_yticks([-2, -1, 0, 1, 2])
ax6.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 20)

ax6.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax6.quiver(X_4, Y_4, U_4, V_4, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax6.set_aspect('equal', adjustable='box')


from matplotlib.lines import Line2D

legend_elements = [Line2D([], [], marker = '_', linestyle='None', label='True Dynamical System',
                          color = 'red', markersize = 36),
                   Line2D([], [], marker = '_', linestyle = 'None', label = 'SEC Approximation',
                          color = 'blue', markersize = 36)]

fig.legend(handles = legend_elements, ncol = 2, loc = 'lower center', bbox_to_anchor=(0.5, -0.01), prop={'size': 26})
  

plt.show()
# %%


"""
Pushforward of vector field plot on the circle
"""
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6), layout="constrained")
# fig.suptitle(r"Pushforwards of SEC Approximated Vector Fields in $\mathbb{R}^2$", fontsize = 28)
# fig.tight_layout(pad = 3.0)

ax1.set_title(r"(a) $F_*V_1^{(L)}$", fontsize = 24)
ax1.set_xlim([-5.5,5.5])
ax1.set_ylim([-5.5,5.5])

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax1.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)

ax1.set_ylabel("y", rotation = 0, fontsize = 20)
ax1.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax1.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)


ax1.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax1.quiver(X_TRAIN_NEW1, Y_TRAIN_NEW1, U_TRAIN_NEW1, V_TRAIN_NEW1, angles = 'xy', scale_units = 'xy', scale = 2, linewidth = 2, color = 'blue')

ax1.set_aspect('equal', adjustable='box')



ax2.set_title(r"(b) $F_*V_2^{(L)}$", fontsize = 24)
ax2.set_xlim([-5.5,5.5])
ax2.set_ylim([-5.5,5.5])

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax2.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)

ax2.set_ylabel("y", rotation = 0, fontsize = 20)
ax2.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax2.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)


ax2.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax2.quiver(X_TRAIN_NEW2, Y_TRAIN_NEW2, U_TRAIN_NEW2, V_TRAIN_NEW2, angles = 'xy', scale_units = 'xy', scale = 2, linewidth = 2, color = 'blue')

for i in range(0, len(v2F_fixed_points)):
    x_fixed = np.linspace(-5, 0, 100)
    y_fixed = np.tan(v2F_fixed_points[i])*x_fixed
    
    ax2.plot(x_fixed, y_fixed, color = 'green')

ax2.set_aspect('equal', adjustable='box')



ax3.set_title(r"(c) $F_*V_3^{(L)}$", fontsize = 24)
ax3.set_xlim([-5.5,5.5])
ax3.set_ylim([-5.5,5.5])

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax3.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)

ax3.set_ylabel("y", rotation = 0, fontsize = 20)
ax3.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax3.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)


ax3.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax3.quiver(X_TRAIN_NEW3, Y_TRAIN_NEW3, U_TRAIN_NEW3, V_TRAIN_NEW3, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax3.set_aspect('equal', adjustable='box')



plt.show()
# %%


"""
POSTER
Pushforward of vector field plot on the circle
"""
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6), layout="constrained")
# fig.suptitle(r"Out-of-sample Evaluation $\vec{V}_{\mathrm{SEC}}: \mathbb{R}^2\to\mathbb{R}^2, \vec{V}_{\mathrm{SEC}} = F_* \tilde{V}_{\mathrm{SEC}}$", fontsize = 28)
# fig.tight_layout(pad = 3.0)

ax1.set_title("\n Rotation", fontsize = 24)
ax1.set_xlim([-5.5,5.5])
ax1.set_ylim([-5.5,5.5])

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax1.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)

ax1.set_ylabel("y", rotation = 0, fontsize = 20)
ax1.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax1.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)


ax1.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax1.quiver(X_TRAIN_NEW1, Y_TRAIN_NEW1, U_TRAIN_NEW1, V_TRAIN_NEW1, angles = 'xy', scale_units = 'xy', scale = 2, linewidth = 2, color = 'blue')

ax1.set_aspect('equal', adjustable='box')



ax2.set_title("\n Arc with Fixed Point", fontsize = 24)
ax2.set_xlim([-5.5,5.5])
ax2.set_ylim([-5.5,5.5])

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax2.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)

ax2.set_ylabel("y", rotation = 0, fontsize = 20)
ax2.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax2.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)


ax2.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax2.quiver(X_TRAIN_NEW2, Y_TRAIN_NEW2, U_TRAIN_NEW2, V_TRAIN_NEW2, angles = 'xy', scale_units = 'xy', scale = 2, linewidth = 2, color = 'blue')

for i in range(0, len(v2F_fixed_points)):
    x_fixed = np.linspace(-5, 0, 100)
    y_fixed = np.tan(v2F_fixed_points[i])*x_fixed
    
    ax2.plot(x_fixed, y_fixed, color = 'green')

ax2.set_aspect('equal', adjustable='box')



ax3.set_title("\n Variable Speed Rotation", fontsize = 24)
ax3.set_xlim([-5.5,5.5])
ax3.set_ylim([-5.5,5.5])

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax3.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)

ax3.set_ylabel("y", rotation = 0, fontsize = 20)
ax3.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax3.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize = 20)


ax3.plot(np.cos(t), np.sin(t), linewidth = 0.5, color = 'black')
ax3.quiver(X_TRAIN_NEW3, Y_TRAIN_NEW3, U_TRAIN_NEW3, V_TRAIN_NEW3, angles = 'xy', scale_units = 'xy', scale = 1, linewidth = 0.5, color = 'blue')

ax3.set_aspect('equal', adjustable='box')



plt.show()
# %%


"""
Solutions to ODEs under the true and SEC approximated systems
"""
# %%

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (16, 12), layout="constrained")
# fig.suptitle(r"Solutions to Dynamical Syatems on $S^1$", fontsize = 32)
# fig.tight_layout(pad = 7.0)


ax1.set_title(r"(a) $V_1$", fontsize = 24)
ax1.set_xlim([-1.1,1.1])
ax1.set_ylim([-1.1,1.1])

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax1.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax1.set_ylabel("y", rotation = 0, fontsize = 20)
ax1.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax1.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax1.plot(sol_true1.y.T[:, 0], sol_true1.y.T[:, 1], color = 'red')

ax1.set_aspect('equal', adjustable='box')


ax2.set_title(r"(b) $V_2$", fontsize = 24)
ax2.set_xlim([-1.1,1.1])
ax2.set_ylim([-1.1,1.1])

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax2.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax2.set_ylabel("y", rotation = 0, fontsize = 20)
ax2.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax2.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax2.plot(sol_true2.y.T[:, 0], sol_true2.y.T[:, 1], color = 'red')
#ax2.plot(sol_sec2.y.T[:, 0], sol_sec2.y.T[:, 1], color = 'blue')


ax2.set_aspect('equal', adjustable='box')




ax3.set_title(r"(c) $V_3$", fontsize = 24)
ax3.set_xlim([-1.1,1.1])
ax3.set_ylim([-1.1,1.1])

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax3.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax3.set_ylabel("y", rotation = 0, fontsize = 20)
ax3.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax3.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax3.plot(sol_true3.y.T[:, 0], sol_true3.y.T[:, 1], color = 'red')

ax3.set_aspect('equal', adjustable='box')




ax4.set_title(r"(d) $V_1^{(L)}$", fontsize = 24)
ax4.set_xlim([-1.1,1.1])
ax4.set_ylim([-1.1,1.1])

ax4.set_xlabel("x", fontsize = 20)
ax4.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax4.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax4.set_ylabel("y", rotation = 0, fontsize = 20)
ax4.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax4.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax4.plot(sol_sec1.y.T[:, 0], sol_sec1.y.T[:, 1], color = 'blue')

ax4.set_aspect('equal', adjustable='box')



ax5.set_title(r"(e) $V_2^{(L)}$", fontsize = 24)
ax5.set_xlim([-1.1,1.1])
ax5.set_ylim([-1.1,1.1])

ax5.set_xlabel("x", fontsize = 20)
ax5.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax5.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax5.set_ylabel("y", rotation = 0, fontsize = 20)
ax5.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax5.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax5.plot(sol_sec2.y.T[:, 0], sol_sec2.y.T[:, 1], color = 'blue')

ax5.set_aspect('equal', adjustable='box')




ax6.set_title(r"(f) $V_3^{(L)}$", fontsize = 24)
ax6.set_xlim([-1.1,1.1])
ax6.set_ylim([-1.1,1.1])

ax6.set_xlabel("x", fontsize = 20)
ax6.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax6.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax6.set_ylabel("y", rotation = 0, fontsize = 20)
ax6.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax6.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax6.plot(sol_sec3.y.T[:, 0], sol_sec3.y.T[:, 1], color = 'blue')

ax6.set_aspect('equal', adjustable='box')



plt.show()
# %%


"""
POSTER
Solutions to ODEs under the true and SEC approximated systems
"""
# %%

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (16, 12), layout="constrained")
# fig.suptitle(r"Solutions to Dynamical Syatems on $S^1$", fontsize = 32)
# fig.tight_layout(pad = 7.0)


ax1.set_title("Rotation \n (a)", fontsize = 24)
ax1.set_xlim([-1.1,1.1])
ax1.set_ylim([-1.1,1.1])

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax1.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax1.set_ylabel("y", rotation = 0, fontsize = 20)
ax1.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax1.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax1.plot(sol_true1.y.T[:, 0], sol_true1.y.T[:, 1], color = 'red')

ax1.set_aspect('equal', adjustable='box')


ax2.set_title("Arc wjth Fixed Point \n (c)", fontsize = 24)
ax2.set_xlim([-1.1,1.1])
ax2.set_ylim([-1.1,1.1])

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax2.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax2.set_ylabel("y", rotation = 0, fontsize = 20)
ax2.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax2.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax2.plot(sol_true2.y.T[:, 0], sol_true2.y.T[:, 1], color = 'red')
#ax2.plot(sol_sec2.y.T[:, 0], sol_sec2.y.T[:, 1], color = 'blue')


ax2.set_aspect('equal', adjustable='box')




ax3.set_title("Variable Speed Rotation \n (e)", fontsize = 24)
ax3.set_xlim([-1.1,1.1])
ax3.set_ylim([-1.1,1.1])

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax3.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax3.set_ylabel("y", rotation = 0, fontsize = 20)
ax3.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax3.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax3.plot(sol_true3.y.T[:, 0], sol_true3.y.T[:, 1], color = 'red')

ax3.set_aspect('equal', adjustable='box')




ax4.set_title("\n (b)", fontsize = 24)
ax4.set_xlim([-1.1,1.1])
ax4.set_ylim([-1.1,1.1])

ax4.set_xlabel("x \n \n \n", fontsize = 20)
ax4.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax4.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax4.set_ylabel("y", rotation = 0, fontsize = 20)
ax4.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax4.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax4.plot(sol_sec1.y.T[:, 0], sol_sec1.y.T[:, 1], color = 'blue')

ax4.set_aspect('equal', adjustable='box')



ax5.set_title("\n (d)", fontsize = 24)
ax5.set_xlim([-1.1,1.1])
ax5.set_ylim([-1.1,1.1])

ax5.set_xlabel("x \n \n \n", fontsize = 20)
ax5.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax5.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax5.set_ylabel("y", rotation = 0, fontsize = 20)
ax5.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax5.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax5.plot(sol_sec2.y.T[:, 0], sol_sec2.y.T[:, 1], color = 'blue')

ax5.set_aspect('equal', adjustable='box')




ax6.set_title("\n (f)", fontsize = 24)
ax6.set_xlim([-1.1,1.1])
ax6.set_ylim([-1.1,1.1])

ax6.set_xlabel("x \n \n \n", fontsize = 20)
ax6.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax6.set_xticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax6.set_ylabel("y", rotation = 0, fontsize = 20)
ax6.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax6.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 14)

ax6.plot(sol_sec3.y.T[:, 0], sol_sec3.y.T[:, 1], color = 'blue')

ax6.set_aspect('equal', adjustable='box')


legend_elements = [Line2D([], [], marker = '_', linestyle='None', label='True Dynamical System',
                          color = 'red', markersize = 36),
                   Line2D([], [], marker = '_', linestyle = 'None', label = 'SEC Approximation',
                          color = 'blue', markersize = 36)]

fig.legend(handles = legend_elements, ncol = 2, loc = 'lower center', bbox_to_anchor=(0.5, -0.01), prop={'size': 26})
  


plt.show()
# %%


"""
Time series evolution of ODE solutions under the true and SEC approximated systems
"""
# %%

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex = True, figsize = (24, 18), layout = "constrained")
# fig.suptitle(r"Time Series Evolution of Solutions to Dynamical Syatems on $S^1$", fontsize = 54)
# fig.tight_layout(pad = 10.0)


ax1.set_title("x-coordinates \n (a)", fontsize = 48)
ax1.set_xlim([-0.1,50])
ax1.set_ylim([-1.1, 1.4])

ax1.set_xlabel("t", fontsize = 32)
ax1.set_xticks([0, 10, 20, 30, 40, 50])
ax1.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax1.set_ylabel("x", rotation = 0, fontsize = 32)
ax1.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax1.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)

ax1.plot(sol_true1.t, sol_true1.y.T[:, 0], color='red')
ax1.plot(sol_sec1.t, sol_sec1.y.T[:, 0], color='blue')

ax1.legend([r"$V_1$", r"$V_1^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax2.set_title('y-coordinates \n (b)', fontsize = 48)
ax2.set_xlim([-0.1,50])
ax2.set_ylim([-1.1, 1.4])

ax2.set_xlabel("t", fontsize = 32)
ax2.set_xticks([0, 10, 20, 30, 40, 50])
ax2.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax2.set_ylabel("y", rotation = 0, fontsize = 32)
ax2.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax2.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)

ax2.plot(sol_true1.t, sol_true1.y.T[:, 1], color='red')
ax2.plot(sol_sec1.t, sol_sec1.y.T[:, 1], color='blue')

ax2.legend([r"$V_1$", r"$V_1^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")


ax3.set_title('(c)', fontsize = 48)
ax3.set_xlim([-0.1,50])
ax3.set_ylim([-0.8, 1.1])

ax3.set_xlabel("t", fontsize = 32)
ax3.set_xticks([0, 10, 20, 30, 40, 50])
ax3.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax3.set_ylabel("x", rotation = 0, fontsize = 32)
ax3.set_yticks([-0.75,  -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax3.set_yticklabels([-0.75,  -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 26)

ax3.plot(sol_true2.t, sol_true2.y.T[:, 0], color='red')
ax3.plot(sol_sec2.t, sol_sec2.y.T[:, 0], color='blue')

ax3.legend([r"$V_2$", r"$V_2^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax4.set_title('(d)', fontsize = 48)
ax4.set_xlim([-0.1,50])
ax4.set_ylim([-0.1, 1.1])

ax4.set_xlabel("t", fontsize = 32)
ax4.set_xticks([0, 10, 20, 30, 40, 50])
ax4.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax4.set_ylabel("y", rotation = 0, fontsize = 32)
ax4.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax4.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize = 26)

ax4.plot(sol_true2.t, sol_true2.y.T[:, 1], color='red')
ax4.plot(sol_sec2.t, sol_sec2.y.T[:, 1], color='blue')

ax4.legend([r"$V_2$", r"$V_2^{(L)}$"], loc="upper right", frameon=True, edgecolor = "black")



ax5.set_title('(e)', fontsize = 48)
ax5.set_xlim([-0.1,50])
ax5.set_ylim([-1.1, 1.45])

ax5.set_xlabel("t", fontsize = 32)
ax5.set_xticks([0, 10, 20, 30, 40, 50])
ax5.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax5.set_ylabel("x", rotation = 0, fontsize = 32)
ax5.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax5.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)

ax5.plot(sol_true3.t, sol_true3.y.T[:, 0], color='red')
ax5.plot(sol_sec3.t, sol_sec3.y.T[:, 0], color='blue')

ax5.legend([r"$V_3$", r"$V_3^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax6.set_title('(f)', fontsize = 48)
ax6.set_xlim([-0.1,50])
ax6.set_ylim([-1.1, 1.45])

ax6.set_xlabel("t", fontsize = 32)
ax6.set_xticks([0, 10, 20, 30, 40, 50])
ax6.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax6.set_ylabel("y", rotation = 0, fontsize = 32)
ax6.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax6.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)
ax6.plot(sol_true3.t, sol_true3.y.T[:, 1], color='red')
ax6.plot(sol_sec3.t, sol_sec3.y.T[:, 1], color='blue')

ax6.legend([r"$V_3$", r"$V_3^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")


plt.show()
# %%


"""
POSTER
Time series evolution of ODE solutions under the true and SEC approximated systems
"""
# %%

fig = plt.figure(layout = "constrained", figsize = (24, 18))
# fig.suptitle(r"Dynamical Evolution $\dot{\theta}(t)  = \vec{V}\mid_{\theta(t)}$", fontsize = 54)
# fig.suptitle(r"Dynamical Evolution $\dot{y}(t)  = \vec{V}_\mathrm{SEC}\mid_{y(t)}, y(t) = F(\theta(t))$", fontsize = 54)
# fig.tight_layout(pad = 10.0)


(topfig, midfig, bottomfig) = fig.subfigures(3, 1)
topfig.suptitle('Rotation', fontsize = 48)
                

(ax1, ax2) = topfig.subplots(nrows = 1, ncols = 2)

ax1.set_title("(a)", fontsize = 48)
ax1.set_xlim([-0.1,50])
ax1.set_ylim([-1.1, 1.4])

ax1.set_xlabel("t", fontsize = 32)
ax1.set_xticks([0, 10, 20, 30, 40, 50])
ax1.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax1.set_ylabel("x", rotation = 0, fontsize = 32)
ax1.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax1.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)

ax1.plot(sol_true1.t, sol_true1.y.T[:, 0], color='red')
ax1.plot(sol_sec1.t, sol_sec1.y.T[:, 0], color='blue')

ax1.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax2.set_title('(b)', fontsize = 48)
ax2.set_xlim([-0.1,50])
ax2.set_ylim([-1.1, 1.4])

ax2.set_xlabel("t", fontsize = 32)
ax2.set_xticks([0, 10, 20, 30, 40, 50])
ax2.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax2.set_ylabel("y", rotation = 0, fontsize = 32)
ax2.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax2.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)

ax2.plot(sol_true1.t, sol_true1.y.T[:, 1], color='red')
ax2.plot(sol_sec1.t, sol_sec1.y.T[:, 1], color='blue')

ax2.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")


midfig.suptitle("Arc with Fixed Point", fontsize = 48)
(ax3, ax4) = midfig.subplots(nrows = 1, ncols = 2)


ax3.set_title('(c)', fontsize = 48)
ax3.set_xlim([-0.1,50])
ax3.set_ylim([-0.8, 1.1])

ax3.set_xlabel("t", fontsize = 32)
ax3.set_xticks([0, 10, 20, 30, 40, 50])
ax3.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax3.set_ylabel("x", rotation = 0, fontsize = 32)
ax3.set_yticks([-0.75,  -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
ax3.set_yticklabels([-0.75,  -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize = 26)

ax3.plot(sol_true2.t, sol_true2.y.T[:, 0], color='red')
ax3.plot(sol_sec2.t, sol_sec2.y.T[:, 0], color='blue')

ax3.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax4.set_title('(d)', fontsize = 48)
ax4.set_xlim([-0.1,50])
ax4.set_ylim([-0.1, 1.1])

ax4.set_xlabel("t", fontsize = 32)
ax4.set_xticks([0, 10, 20, 30, 40, 50])
ax4.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax4.set_ylabel("y", rotation = 0, fontsize = 32)
ax4.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax4.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize = 26)

ax4.plot(sol_true2.t, sol_true2.y.T[:, 1], color='red')
ax4.plot(sol_sec2.t, sol_sec2.y.T[:, 1], color='blue')

ax4.legend(["True", "SEC"], loc="upper right", frameon=True, edgecolor = "black")



bottomfig.suptitle("Variable Speed Rotation", fontsize = 48)
(ax5, ax6) = bottomfig.subplots(nrows = 1, ncols = 2)

ax5.set_title('(e)', fontsize = 48)
ax5.set_xlim([-0.1,50])
ax5.set_ylim([-1.1, 1.45])

ax5.set_xlabel("t", fontsize = 32)
ax5.set_xticks([0, 10, 20, 30, 40, 50])
ax5.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax5.set_ylabel("x", rotation = 0, fontsize = 32)
ax5.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax5.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)

ax5.plot(sol_true3.t, sol_true3.y.T[:, 0], color='red')
ax5.plot(sol_sec3.t, sol_sec3.y.T[:, 0], color='blue')

ax5.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax6.set_title('(f)', fontsize = 48)
ax6.set_xlim([-0.1,50])
ax6.set_ylim([-1.1, 1.45])

ax6.set_xlabel("t", fontsize = 32)
ax6.set_xticks([0, 10, 20, 30, 40, 50])
ax6.set_xticklabels([0, 10, 20, 30, 40, 50], fontsize = 26)

ax6.set_ylabel("y", rotation = 0, fontsize = 32)
ax6.set_yticks([-1,  -0.5, 0, 0.5, 1])
ax6.set_yticklabels([-1,  -0.5, 0, 0.5, 1], fontsize = 26)
ax6.plot(sol_true3.t, sol_true3.y.T[:, 1], color='red')
ax6.plot(sol_sec3.t, sol_sec3.y.T[:, 1], color='blue')

ax6.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")


plt.show()
# %%


# %%
ana_dir_coords = np.vstack([TRAIN_X, TRAIN_Y, v2F(THETA_LST)])

vec_ana = ana_dir_coords[2:4, :]
vec_sec = vector_approx2[:, 2:4].T

ana_norm = np.sqrt(np.sum(np.power(vec_ana, 2), axis = 0))
sec_norm = np.sqrt(np.sum(np.power(vec_sec, 2), axis = 0))

norm_ratio = ana_norm/sec_norm


rss = np.sum(np.power((vec_ana - vec_sec), 2))

vec_bar = np.mean(vec_ana, axis = 1)
tss = np.sum(np.power(vec_ana, 2))


R_squared = 1 - rss/tss


print(R_squared)
# %%

