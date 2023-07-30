"""
Spectral Exterior Calculus (SEC)
2-torus T2 Example (Roration)
Approximations of vector fields on the 2-torus
usng flat torus embedding into R^4
Given pushforward of tangent vectors on the 2-torus
and determinstically sampled Monte Carlo points on the circle
"""



# %%
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import random
from numpy.linalg import eig as eig
from scipy.sparse.linalg import eigs as eigs
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import multiprocess as mp


# Parameters
I = 20          # Inner index for eigenfunctions
J = 5           # Outer index for eigenfunctions
K = 5           # Index for gradients of eigenfunctions
n = 100          # Number of approximated tangent vectors
N = 100         # Number of Monte Carlo training data points 

epsilon = 0.25  # RBF bandwidth parameter
tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix
a = 4           # Radius of the latitude circle of the torus
b = 1           # Radius of the meridian circle of the torus


"""
Training data set
with pushforward of vector fields v on the torus
and smbedding map F with pushforward F_*v = vF
"""


# Deterministically sampled Monte Carlo training data points
# the latotude and meridian circles with radius a and b
def monte_carlo_points(start_pt = 0, end_pt = 2*np.pi, N = 100):
    u_a = np.zeros(N)
    u_b = np.zeros(N)
    
    subsets = np.arange(0, N+1, (N/50))
    for i in range(0, int(N/2)):
        start = int(subsets[i])
        end = int(subsets[i+1])
        u_a[start:end] = random.uniform(low = (i/(N/2))*end_pt, high = ((i+1)/(N/2))*end_pt, size = end - start)
        u_b[start:end] = random.uniform(low = (i/(N/2))*end_pt, high = ((i+1)/(N/2))*end_pt, size = end - start)
    
    random.shuffle(u_a)
    random.shuffle(u_b)

    training_data_a = np.empty([2, N], dtype = float)
    training_data_b = np.empty([2, N], dtype = float)
    
    for j in range(0, N):
            training_data_a[:, j] = np.array([a*np.cos(u_a[j]), a*np.sin(u_a[j])])
            training_data_b[:, j] = np.array([b*np.cos(u_b[j]), b*np.sin(u_b[j])])
    
    return u_a, u_b, training_data_a, training_data_b

u_a, u_b, training_data_a, training_data_b = monte_carlo_points()

# Create mesh of angles theta and rho for the latitude and meridian cricles
# and transform into grid of points with these two angles
THETA_LST, RHO_LST = np.meshgrid(u_a, u_b)

training_angle = np.vstack([THETA_LST.ravel(), RHO_LST.ravel()])


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
# Functions specifying the coordinates in R4
# using the angles theat and rho for the latitude and meridian circles
X_func = lambda theta: a*np.cos(theta)
Y_func = lambda theta: a*np.sin(theta)
Z_func = lambda rho: b*np.cos(rho)
W_func = lambda rho: b*np.sin(rho)

# N*N training data points corrdinates in the x, y, z and w coordinates
TRAIN_X = X_func(training_angle[0, :])
TRAIN_Y = Y_func(training_angle[0, :])
TRAIN_Z = Z_func(training_angle[1, :])
TRAIN_W = W_func(training_angle[1, :])

# N*N training data points containing all four coordinates of each point
training_data = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, TRAIN_W])


x = a*np.cos(training_angle[0, :])
y = a*np.sin(training_angle[0, :])
z = b*np.cos(training_angle[1, :])
w = b*np.sin(training_angle[1, :])


# X_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.cos(rho)
# X_func_drho = lambda theta, rho: -(a + b*np.cos(theta))*np.sin(rho)
# Y_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.sin(rho)
# Y_func_drho = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
# Z_func_dtheta = lambda theta: b*np.cos(theta)
# Z_func_drho = lambda theta: 0

# TRAIN_X_DERIVATIVE = np.array([x_dtheta + x_drho for x_dtheta, x_drho in zip(list(map(X_func_dtheta, THETA_LST, RHO_LST)), list(map(X_func_drho, THETA_LST, RHO_LST)))])
# TRAIN_Y_DERIVATIVE = np.array([y_dtheta + y_drho for y_dtheta, y_drho in zip(list(map(Y_func_dtheta, THETA_LST, RHO_LST)), list(map(Y_func_drho, THETA_LST, RHO_LST)))])
# TRAIN_Z_DERIVATIVE = np.array(Z_func_dtheta(THETA_LST) + Z_func_drho(THETA_LST))
# %%


# %%
"""
Functions utilized in the following program
"""

# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta, rho: np.array([a*np.cos(theta), a*np.sin(theta), b*np.cos(rho), b*np.sim(rho)])

# The partial detivative vector field v1
v1F = lambda theta, rho: np.array([-a*np.sin(theta), a*np.cos(theta), -b*np.sin(rho), b*np.cos(rho)])

# The Stepanoff flow vector field v2
v2F = lambda theta, rho: np.array([-a*np.sin(theta) - a*np.sin(theta)*np.cos(theta - rho) - a*(1 - alpha)*np.sin(theta)*(1 - np.cos(rho)), 
                                   a*np.cos(theta) + a*np.cos(theta)*np.cos(theta - rho) + a*(1 - alpha)*np.cos(theta)*(1 - np.cos(rho)),
                                   -b*alpha**(np.sin(rho) - np.sin(rho)*np.cos(theta - rho)),
                                   b*alpha*(np.cos(rho) - np.cos(rho)*np.cos(theta - rho))])


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



# %%
"""
Implementation of diffusion maps algorithm
Approximation of eigenvalues and eigenfunctions of the 0-Laplacian
uo to a constant scaling factor
"""


# Diffusion maps algorithm

# Heat kernel function k
k = lambda x_1, x_2: np.exp(-dist_matrix(x_1, x_2)/(epsilon**2))

# Normalization function q corresponding to diagonal matrix Q
def make_normalization_func(k, x_train):
    def normalized(x):
        y = np.sum(k(x, x_train), axis = 1)
        return y
    return normalized


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
# print(K_hat[:2,:2])
# %%


# %%
# Normalization function d that corresponds to diagonal matrix D
d = make_normalization_func(k_hat, training_data)
D = d(training_data)
# %%


# %%
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
# %%


# %%
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
# %%


# %%
# Solve eigenvalue problem for similarity matrix S
eigenvalues, eigenvectors = eigs(S, k = 150) 
index = eigenvalues.argsort()[::-1][:2*I+1]
Lambs = eigenvalues[index]
Phis = np.real(eigenvectors[:, index])

# Compute approximated 0-Laplacian eigengunctions
lambs = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
            lambs[i] = 4*(-np.log(np.real(Lambs[i]))/(epsilon**2))
            # lambs_dm[i] = (1 - np.real(Lambs[i]))/(epsilon**2)   

print(lambs)         



# Normalize eigenfunctions Phi_j
Phis_normalized = np.empty([N*N, 2*I+1], dtype = float)
for j in range(0, 2*I+1):
    Phis_normalized[:, j] = np.real(Phis[:, j])*N

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
# %%



# %%
# Apply the coninuous extensiom varphi to the training data set
varphi_xyzw = varphi(training_data)
# %%



# %%
"""
Check accuracy of diffusion maps approximation
for eigenvalues and eigenfunctions of 0-Laplacian
"""

x_coords = THETA_LST
y_coords = RHO_LST

z_true = np.reshape(Phis_normalized[:, 12], (N, N))
z_dm = np.reshape(np.real(varphi_xyzw[:, 12]), (N, N))


# 3D surface plots of true and diffusion maps approximated 0-Laplacian eigenvectors z_true and z_dm
# against theta and rho with colors corresponding to the values of z_true and z_dm
fig = plt.figure()

ax = fig.gca(projection = Axes3D.name)
ax.plot_wireframe(x_coords, y_coords, z_true)
# ax.plot_wireframe(x_coords, y_coords, z_dm)


cmap = plt.cm.plasma
norm = matplotlib.colors.Normalize(vmin = np.min(z_true), vmax = np.max(z_true))
# norm = matplotlib.colors.Normalize(vmin = np.min(z_dm), vmax = np.max(z_dm))

colors = cmap(norm(z_true))
# colors = cmap(norm(z_dm))

ax.plot_surface(x_coords, y_coords, np.zeros_like(x_coords), cstride = 1, rstride = 1, facecolors = colors, shade = False)

sc = matplotlib.cm.ScalarMappable(cmap = cmap, norm = norm)
sc.set_array([])
plt.colorbar(sc)

plt.title('0-Laplacian eigenvectors against theta and rho')

plt.show()
# %%


# %%
# pcolor plots of true and diffusion maps approximated 0-Laplacian 
# eigenvectors and eigenfunctions z_true and z_dm values

fig = plt.figure(figsize=(8, 16))

plt.subplot(2, 1, 1)
plt.pcolor(z_true, edgecolors = 'k', linewidths = 0, cmap = 'Blues')
plt.title('True 0-Laplacian eigenvector z_true values')

plt.subplot(2, 1, 2)
plt.pcolor(z_dm, edgecolors = 'k', linewidths = 0, cmap = 'summer')
plt.title('Diffusion maps approximated 0-Laplacian eigenfunction z_dm values')

plt.show()
# %%
