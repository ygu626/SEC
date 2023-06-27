"""
Spectral Exterior Calculus (SEC)
2-torus T2 Example (Roration)
Approximations of vector fields on the 2-torus
usng donut e bedding into R^3
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
I = 10          # Inner index for eigenfunctions
J = 5           # Outer index for eigenfunctions
K = 3           # Index for gradients of eigenfunctions
n = 8          # Number of approximated tangent vectors
N = 800         # Number of Monte Carlo training data points 

epsilon = 0.15  # RBF bandwidth parameter
tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix
a = 1           # Radius of the latitude circle of the torus
b = 1           # Radius of the meridian circle of the torus


"""
Training data set
with pushforward of vector fields v on the torus
and smbedding map F with pushforward F_*v = vF
"""


# Deterministically sampled Monte Carlo training data points
# for two circles *latotude and meridian) of radius a and b
def monte_carlo_points(start_pt = 0, end_pt = 2*np.pi, N = 800):
    u_a = np.zeros(N)
    u_b = np.zeros(N)
    
    subsets = np.arange(0, N+1, N/400)
    for i in range(0, 400):
        start = int(subsets[i])
        end = int(subsets[i+1])
        u_a[start:end] = random.uniform(low = (i/400)*end_pt, high = ((i+1)/400)*end_pt, size = end - start)
        u_b[start:end] = random.uniform(low = (i/400)*end_pt, high = ((i+1)/400)*end_pt, size = end - start)
    
    random.shuffle(u_a)
    random.shuffle(u_b)
    
    training_data_a = np.empty([2, N], dtype = float)
    training_data_b = np.empty([2, N], dtype = float)
    
    for j in range(0, N):
            training_data_a[:, j] = np.array([a*np.cos(u_a[j]), a*np.sin(u_a[j])])
            training_data_b[:, j] = np.array([b*np.cos(u_b[j]), b*np.sin(u_b[j])])
    
    return u_a, u_b, training_data_a, training_data_b

u_a, u_b, training_data_a, training_data_b = monte_carlo_points()

sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(x = training_data_a[0,:], y = training_data_a[1,:], color = 'green')
ax1.set_xlim([-5,5])
ax1.set_ylim([-5,5])
ax1.set_title('Monte Carlo Sampled Latitude Circle with Radius a')

ax2.scatter(x = training_data_b[0,:], y = training_data_b[1,:], color = 'orange')
ax2.set_xlim([-5,5])
ax2.set_ylim([-5,5])
ax2.set_title('Monte Carlo Sampled Meridian Circle with Radius b')

plt.show()
# %%


# %%
# n pushforward of vector field v ("arrows") on the torus
# given points (x, y, z) specified by angle theta on the meridian circle and angle rho on the latitude circle
THETA_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))
RHO_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))

X_func = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
Y_func = lambda theta, rho: (a + b*np.cos(theta))*np.sin(rho)
Z_func = lambda theta: b*np.sin(theta)

X_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.cos(rho)
X_func_drho = lambda theta, rho: -(a + b*np.cos(theta))*np.sin(rho)
Y_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.sin(rho)
Y_func_drho = lambda theta rho: (a + b*np.cos(theta))*np.cos(rho)
Z_func_dtheta = lambda theta: b*np.cos(theta)
Z_func_drho = lambda theta: 0


TRAIN_X = np.array(X_func(THETA_LST, RHO_LST))
TRAIN_Y = np.array(Y_func(THETA_LST, RHO_LST))
TRAIN_Z = np.array([Z_func(THETA_LST)])

TRAIN_X_DERIVATIVE = X_func_dtheta((THETA_LST, RHO_LST)) + X_func_drho((THETA_LST, RHO_LST))
TRAIN_Y_DERIVATIVE = Y_func_dtheta((THETA_LST, RHO_LST)) + Y_func_drho((THETA_LST, RHO_LST))
TRAIN_Z_DERIVATIVE = Z_func_dtheta((THETA_LST)) + Z_func_drho((THETA_LST)


TRAIN_V = np.empty([n, 6], dtype = float)
for i in range(0, n):
    TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], TRAIN_X_DERIVATIVE[i], TRAIN_Y_DERIVATIVE[i], TRAIN_Z_DERIVATIVE[i]])

X_1, Y_1, Z_1 U, V_1, W_1 = zip(*TRAIN_V)

print(U_1)
print(V_1)
print(W_1)



# Embedding map F and its pushforward applied vF to vector field v
F = lambda theta: np.array([np.cos(theta), np.sin(theta)])
vF = lambda theta: np.array([-np.sin(theta), np.cos(theta)])




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

# %%



"""
Implementation of diffusion maps algorithm
Approximation of eigenvalues and eigenfunctions of the 0-Laplacian
uo to a constant scaling factor
"""


# %%
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
    def p(x, y):
        d_x = d(x).reshape(d(x).shape[0], 1)

        p_xy = np.divide(k_hat(x, y), d_x)
        return p_xy
    return p

# Build Markov kernel matrix P
p = make_p(k_hat, d)
P = p(training_data, training_data)
# print(P[:3,:3])

# Similarity transformation function s
def make_s(p, d):
    def s(x, y):
        d_x = d(x)
        d_y = d(y).reshape(d(y).shape[0], 1)
        
        s_xy = np.divide(np.multiply(p(x, y), d_x), d_y)
        return s_xy
    return s

# Build Similarity matrix S
s = make_s(p, d)
S = s(training_data, training_data)
# print(S[:3,:3])


# Solve eigenvalue problem for similarity matrix S
eigenvalues, eigenvectors = eig(S) 
index = eigenvalues.argsort()[::-1][:2*I+1]
Lambs = eigenvalues[index]
Phis = np.real(eigenvectors[:, index])

# Compute approximated 0-Laplacian eigengunctions
lambs_dm = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
            lambs_dm[i] = 4*(-np.log(np.real(Lambs[i]))/(epsilon**2)) 

print(lambs_dm)         
# %%

# Normalize eigenfunctions Phi_j
Phis_normalized = np.empty([N, 2*I+1], dtype = float)
for j in range(0, 2*I+1):
    Phis_normalized[:, j] = np.real(Phis[:, j])*np.sqrt(N)

# Appeoximate eigenvalues and eigenfunctions for the 0-Laplacian
def make_varphi(k, x_train, lambs, phis):
    phi_lamb = phis / lambs
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    return varphi

# Produce continuous extentions varphi_j for the eigenfunctions Phi_j
Lambs_normalized = np.power(Lambs, 4)
varphi = make_varphi(p, training_data, Lambs, Phis_normalized)



"""
Check accuracy of diffusion maps approximation
fir eigenvalues and eigenfunctions of 0-Laplacian
"""

# Check approximations for Laplacian eigenbasis agree with true eigenbasis
# by ploting against linear combinations of true eigenfunctions 

# Get x values of the sine wave
time = u
time2 = u

# Amplitude of the sine wave is sine of a variable like time
amplitude = Phis_normalized[:, 1]
amplitude2 = np.real(varphi(training_data)[:, 1])

# Plot a sine wave using time and amplitude obtained for the sine wave
plt.scatter(time, amplitude, color = 'blue')
plt.scatter(time2, amplitude2, color = 'red')

# Give a title for the sine wave plot
plt.title('Sine wave')

# Give x axis label for the sine wave plot
plt.xlabel('Time')

# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = sin(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.show()
# %%
