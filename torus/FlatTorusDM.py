"""
Spectral Exterior Calculus (SEC)
2-torus T2 Example (Roration)
Approximations of vector fields on the 2-torus
using flat embedding into R^4
Given pushforward of tangent vectors on the torus
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
and embedding map F with pushforward F_*v = vF
"""


# Deterministically sampled Monte Carlo training data points
# for two circles (latitude and meridian) of radius a and b
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
# given points (x, y) specified by angle theta on the latitude circle
# and points (z, w)specified by angle rho on the meridian circle
THETA_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))
RHO_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))

X_func = lambda theta: a*np.cos(theta)
Y_func = lambda theta: a*np.sin(theta)
Z_func = lambda rho: b*np.cos(rho)
W_func = lambda rho: b*np.sin(rho)

TRAIN_X = np.array(X_func(THETA_LST))
TRAIN_Y = np.array(Y_func(THETA_LST))
TRAIN_Z = np.array(Z_func(RHO_LST))
TRAIN_W = np.array(W_func(RHO_LST))


TRAIN_V_a = np.empty([n, 4], dtype = float)
TRAIN_V_b = np.empty([n, 4], dtype = float)

for i in range(0, n):
    TRAIN_V_a[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i], TRAIN_X[i]])
    TRAIN_V_b[i, :] = np.array([TRAIN_Z[i], TRAIN_W[i], -TRAIN_W[i], TRAIN_Z[i]])

X_a, Y_a, U_a, V_a = zip(*TRAIN_V_a)
X_b, Y_b, U_b, V_b = zip(*TRAIN_V_b)

print(U_a)
print(V_a)
print(U_b)
print(V_b)



# Embedding map F and its pushforward vF applied to vector field v
F = lambda theta, rho: np.array([a*np.cos(theta), a*np.sin(theta), b*np.cos(rho), b*np.sin(rho)])
vF = lambda theta, rho: np.array([-a*np.sin(theta), a*np.cos(theta), -b*np.sin(rho), b*np.cos(tho)])
# %%



"""
Functions utilized in diffusion maps algorithn and SEC approximation
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
on the latutide circle with radius a and meridian circle with radius b of the torus
uo to constant scaling factors
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
# K = k(training_data_a, training_data_a)

# Normalized kernel function k_hat
def make_k_hat(k, q):
    def k_hat(x, y):
        q_x = q(x).reshape(q(x).shape[0], 1)
        q_y = q(y).reshape(1, q(y).shape[0])
        # treat qx as column vector
        k_hat_xy = np.divide(k(x, y), np.matmul(q_x, q_y))
        return k_hat_xy
    return k_hat

# Build normalized kernel matrices K_hat_a and K_hat_b for tbe latutide and meridian circles
q_a = make_normalization_func(k, training_data_a)
q_b = make_normalization_func(k, training_data_b)

k_hat_a = make_k_hat(k, q_a)
k_hat_b = make_k_hat(k, q_b)

K_hat_a = k_hat_a(training_data_a, training_data_a)
K_hat_b = k_hat_b(training_data_b, training_data_b)
# print(K_hat[:3,:3])


# Normalization function d_a and d_b corresponding to diagonal matrices D_a and D_b for the latitude and meridian circles
d_a = make_normalization_func(k_hat_a, training_data_a)
d_b = make_normalization_func(k_hat_b, training_data_b)

D_a = d_a(training_data_a)
D_b = d_a(training_data_b) 


# Markov kernel function p
def make_p(k_hat, d):
    def p(x, y):
        d_x = d(x).reshape(d(x).shape[0], 1)

        p_xy = np.divide(k_hat(x, y), d_x)
        return p_xy
    return p

# Build Markov kernel matrices P_a and P_b for the latitude and meridian circles
p_a = make_p(k_hat_a, d_a)
p_b = make_p(k_hat_b, d_b)

P_a = p_a(training_data_a, training_data_a)
P_b = p_b(training_data_b, training_data_b)
# print(P[:3,:3])


# Similarity transformation function s
def make_s(p, d):
    def s(x, y):
        d_x = d(x)
        d_y = d(y).reshape(d(y).shape[0], 1)
        
        s_xy = np.divide(np.multiply(p(x, y), d_x), d_y)
        return s_xy
    return s

# Build Similarity matrices S_a and S_b for the latitude and meridian circles
s_a = make_s(p_a, d_a)
s_b = make_s(p_b, d_b)

S_a = s(training_data_a, training_data_a)
S_b = s(training_data_b, training_data_b)
# print(S[:3,:3])


# Solve eigenvalue problem for similarity matrices S_a and S_b
eigenvalues_a, eigenvectors_a = eig(S_a) 
eigenvalues_b, eigenvectors_b = eig(S_b) 

index_a = eigenvalues_a.argsort()[::-1][:2*I+1]
Lambs_a = eigenvalues_a[index_a]
Phis_a = np.real(eigenvectors_a[:, index_a])

index_b = eigenvalues_b.argsort()[::-1][:2*I+1]
Lambs_b = eigenvalues_b[index_b]
Phis_b = np.real(eigenvectors_b[:, index_b])


# Compute approximated 0-Laplacian eigengunctions for the latitude and meridian circles
lambs_a = np.empty(2*I+1, dtype = float)
lambs_b = np.empty(2*I+1, dtype = float)

for i in range(0, 2*I+1):
            lambs_a[i] = (-4*np.log(np.real(Lambs_a[i]))/(epsilon**2))
            lambs_b[i] = (-4*np.log(np.real(Lambs_b[i]))/(epsilon**2))  

print(lambs_a)  
print(lambs_b)        
# %%

# %%
# Normalize eigenfunctions Phi_a_j and Phi_b_j for the latitude and meridian circles
Phis_a_normalized = np.empty([N, 2*I+1], dtype = float)
Phis_b_normalized = np.empty([N, 2*I+1], dtype = float)

for j in range(0, 2*I+1):
    Phis_a_normalized[:, j] = np.real(Phis_a[:, j])*np.sqrt(N)
    Phis_b_normalized[:, j] = np.real(Phis_b[:, j])*np.sqrt(N)


# Appeoximate eigenvalues and eigenfunctions for the 0-Laplacian for the latitude and meridian circles
def make_varphi(k, x_train, lambs, phis):
    phi_lamb = phis / lambs
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    return varphi

# Produce continuous extentions varphi_a_j and varphi_b_j for the eigenfunctions Phi_a_j and Phi_b_j
Lambs_a_normalized = np.power(Lambs_a, 4)
Lambs_b_normalized = np.power(Lambs_b, 4)

varphi_a = make_varphi(p_a, training_data_a, Lambs_a, Phis_a_normalized)
varphi_b = make_varphi(p_b, training_data_b, Lambs_b, Phis_b_normalized)

varphi = lambda training_data_a, training_data_b: np.matmul(varphi_a(training_data_a), varphi_b(training_data_b))
# %%


"""
Check accuracy of diffusion maps approximation
for eigenvalues and eigenfunctions of the 0-Laplacian
on the latitude circle with radius a and meridian circle with radius b of the torus
"""

# Check approximations for Laplacian eigenbasis agree with true eigenbasis
# by ploting against linear combinations of true eigenfunctions 

# Get x values of the sine wave
time_a = u_a * u_b
time2_a = u_a * u_b

# Amplitude of the sine wave is sine of a variable like time
amplitude_a = Phis_a_normalized[:, 1] * Phis_b_normalized[:, 1]
amplitude2_a = np.real(varphi_a(training_data_a)[:, 1]) * np.real(varphi_b(training_data_b)[:, 1])

# print(amplitude_a.shape)


# Plot a sine wave using time and amplitude obtained for the sine wave
plt.scatter(time_a, amplitude_a, color = 'blue')
plt.scatter(time2_a, amplitude2_a, color = 'red')

# Give a title for the sine wave plot
plt.title('Sine multipled by cosine waves')

# Give x axis label for the sine wave plot
plt.xlabel('Time')

# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = sin(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.show()
# %%

# %%
# Get x values of the sine wave
time_b = u_b
time2_b = u_b

# Amplitude of the sine wave is sine of a variable like time
amplitude_b = Phis_b_normalized[:, 1]
amplitude2_b = np.real(varphi_b(training_data_b)[:, 1])
print(amplitude2_b.shape)
# %%

# Plot a sine wave using time and amplitude obtained for the sine wave
plt.scatter(time_b, amplitude_b, color = 'blue')
plt.scatter(time2_b, amplitude2_b, color = 'red')

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
