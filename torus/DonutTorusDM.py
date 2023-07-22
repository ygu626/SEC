"""
Spectral Exterior Calculus (SEC)
2-torus T2 Example (Roration)
Approximations of vector fields on the 2-torus
usng donut embedding into R^3
Given pushforward of tangent vectors on the circle
and determinstically sampled Monte Carlo points on the circle
"""



# %%
import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd
from numpy import random
from numpy.linalg import eig as eig
import multiprocess as mp
from scipy.integrate import quad
from scipy.integrate import solve_ivp


# Parameters
I = 10          # Inner index for eigenfunctions
J = 5           # Outer index for eigenfunctions
K = 3           # Index for gradients of eigenfunctions
n = 100          # Number of approximated tangent vectors
N = 100         # Number of Monte Carlo training data points 

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

training_data = np.vstack([THETA_LST.ravel(), RHO_LST.ravel()])


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
# Functions specifying the coordinates in R3
# using the angles theat and rho for the latitude and meridian circles
X_func = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
Y_func = lambda theta, rho: (a + b*np.cos(theta))*np.sin(rho)
Z_func = lambda theta: b*np.sin(theta)

# N*N training data point corrdinates in the x, y and z coordinates
TRAIN_X = X_func(training_data[0, :], training_data[1, :])
TRAIN_Y = Y_func(training_data[0, :], training_data[1, :])
TRAIN_Z = Z_func(training_data[0, :])


x = (a + b*np.cos(training_data[0, :]))*np.cos(training_data[1, :])
y = (a + b*np.cos(training_data[0, :]))*np.sin(training_data[1, :])
z = b*np.sin(training_data[0, :])
# %%


# X_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.cos(rho)
# X_func_drho = lambda theta, rho: -(a + b*np.cos(theta))*np.sin(rho)
# Y_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.sin(rho)
# Y_func_drho = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
# Z_func_dtheta = lambda theta: b*np.cos(theta)
# Z_func_drho = lambda theta: 0

# TRAIN_X_DERIVATIVE = np.array([x_dtheta + x_drho for x_dtheta, x_drho in zip(list(map(X_func_dtheta, THETA_LST, RHO_LST)), list(map(X_func_drho, THETA_LST, RHO_LST)))])
# TRAIN_Y_DERIVATIVE = np.array([y_dtheta + y_drho for y_dtheta, y_drho in zip(list(map(Y_func_dtheta, THETA_LST, RHO_LST)), list(map(Y_func_drho, THETA_LST, RHO_LST)))])
# TRAIN_Z_DERIVATIVE = np.array(Z_func_dtheta(THETA_LST) + Z_func_drho(THETA_LST))


# TRAIN_V = np.empty([n, 6], dtype = float)
# for i in range(0, n):
#     TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], TRAIN_X_DERIVATIVE[i], TRAIN_Y_DERIVATIVE[i], TRAIN_Z_DERIVATIVE[i]])


# X_1, Y_1, Z_1, U_1, V_1, W_1 = zip(*TRAIN_V)


# fig = plt.figure()

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.set_zlim(-3,3)
# ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
# ax1.view_init(36, 26)

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_zlim(-3,3)
# ax2.plot_surface(TRAIN_X, TRAIN_Y, TRAIN_Z, rstride=5, cstride=5, color='k', edgecolors='w')
# ax2.view_init(0, 0)
# ax2.set_xticks([])

# plt.show()
# %%
