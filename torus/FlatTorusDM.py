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



# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta, rho: np.array([a*np.cos(theta), a*np.sin(theta), b*np.cos(rho), b*np.sin(rho)])

v1F = lambda theta, rho: np.array([-a*np.sin(theta), a*np.cos(theta), -b*np.sin(rho), b*np.cos(rho)])

# Pushforward of the flat embedding F applied to
# the Stepanoff flow vector field v
v2F = lambda theta, rho: np.array([-a*np.sin(theta) - a*np.sin(theta)*np.cos(theta - rho) - a*(1 - alpha)*np.sin(theta)*(1 - np.cos(rho)), 
                                   a*np.cos(theta) + a*np.cos(theta)*np.cos(theta - rho) + a*(1 - alpha)*np.cos(theta)*(1 - np.cos(rho)),
                                   -b*alpha**(np.sin(rho) - np.sin(rho)*np.cos(theta - rho)),
                                   b*alpha*(np.cos(rho) - np.cos(rho)*np.cos(theta - rho))])
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

S_a = s_a(training_data_a, training_data_a)
S_b = s_b(training_data_b, training_data_b)
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
time_a = u_a * u_a
time2_a = u_a * u_a

# Amplitude of the sine wave is sine of a variable like time
amplitude_a = Phis_a_normalized[:, 1] * Phis_a_normalized[:, 1]
amplitude2_a = np.real(varphi_a(training_data_a)[:, 1]) * np.real(varphi_a(training_data_a)[:, 1])

# print(amplitude_a.shape)


# Plot a sine wave using time and amplitude obtained for the sine wave
plt.scatter(time_a, amplitude_a, color = 'blue')
plt.scatter(time2_a, amplitude2_a, color = 'red')

# Give a title for the sine wave plot
plt.title('Sine square waves')

# Give x axis label for the sine wave plot
plt.xlabel('Time')

# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = sin^2(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.show()
# %%

# %%
# Get x values of the sine wave
time_b = u_b * u_b
time2_b = u_b * u_b

# Amplitude of the sine wave is sine of a variable like time
amplitude_b = Phis_b_normalized[:, 1] * Phis_b_normalized[:, 1]
amplitude2_b = np.real(varphi_b(training_data_b)[:, 1]) * np.real(varphi_b(training_data_b)[:, 1])


# Plot a sine wave using time and amplitude obtained for the sine wave
plt.scatter(time_b, amplitude_b, color = 'blue')
plt.scatter(time2_b, amplitude2_b, color = 'red')

# Give a title for the sine wave plot
plt.title('Cosine square waves')

# Give x axis label for the sine wave plot
plt.xlabel('Time')

# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = cos^2(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.show()
# %%


# %%
# Get x values of the sine wave
time_mixed = u_a * u_b
time2_mixed = u_a * u_b

# Amplitude of the sine wave is sine of a variable like time
amplitude_mixed = Phis_a_normalized[:, 1] * Phis_b_normalized[:, 1]
amplitude2_mixed = np.real(varphi_a(training_data_a)[:, 1]) * np.real(varphi_b(training_data_b)[:, 1])


# Plot a sine wave using time and amplitude obtained for the sine wave
plt.scatter(time_mixed, amplitude_mixed, color = 'blue')
plt.scatter(time2_mixed, amplitude2_mixed, color = 'red')

# Give a title for the sine wave plot
plt.title('Sine times cosine waves')

# Give x axis label for the sine wave plot
plt.xlabel('Time')

# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = sin(time)*cos(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.show()
# %%


"""
SEC approximation
for pushforward of vector fields on the torus
with flat embedding into R4
"""

# %%
# Fourier coefficients F_ak pf F w.r.t. difusion maps eigenvectors Phi_j
F_ak = (1/N)*np.concatenate((np.matmul(F(u_a, u_b)[0:2, :], Phis_a_normalized), np.matmul(F(u_a, u_b)[2:4, :], Phis_b_normalized)), axis = 0)
# %%


# %%
# Compute c_ijp coefficients
# for the latitude and meridian circles
# using Monte Carlo integration
pool = mp.Pool()

def c_a_func(i, j, p):
    return (1/N)*np.sum(Phis_a_normalized[:, i]*Phis_a_normalized[:, j]*Phis_a_normalized[:, p])
def c_b_func(i, j, p):
    return (1/N)*np.sum(Phis_b_normalized[:, i]*Phis_b_normalized[:, j]*Phis_b_normalized[:, p])


c_a = pool.starmap(c_a_func, 
              [(i, j, p) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for p in range(0, 2 * I + 1)])

c_b = pool.starmap(c_b_func, 
              [(i, j, p) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for p in range(0, 2 * I + 1)])
  
    
c_a = np.reshape(np.array(c_a), (2 * I + 1, 2 * I + 1, 2 * I + 1))
c_b = np.reshape(np.array(c_b), (2 * I + 1, 2 * I + 1, 2 * I + 1))
print(c_a[:,3,3])
print(c_b[:,3,3])
# %%

# %%
# Compute g_ijp Riemannian metric coefficients
# for the latitude and meridian circles
# using Monte Carlo integration
g_a = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
g_b = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

g_a_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
g_b_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for p in range(0, 2*I+1):
                                    g_a_coeff[i,j,p] = (lambs_a[i] + lambs_a[j] - lambs_a[p])/2
                                    g_b_coeff[i,j,p] = (lambs_b[i] + lambs_b[j] - lambs_b[p])/2
                                    
g_a = np.multiply(g_a_coeff, c_a)
g_b = np.multiply(g_b_coeff, c_b)
   
# print(g_a[:,3,3])
# print(g_b[:,3,3])
# %%


# %%
# Compute G_ijpq entries for the Gram operator and its dual
# for the latitude and meridian circles
# using Monte Carlo integration
G_a = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G_b = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)

G_a = np.einsum('ipm, jqm -> ijpq', c_a, g_a, dtype = float)
G_b = np.einsum('ipm, jqm -> ijpq', c_b, g_b, dtype = float)

G_a = G_a[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G_b = G_b[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]

G_a = np.reshape(G_a, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))
G_b = np.reshape(G_b, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))


# Perform singular value decompositions (SVD) of the Gram operators G_a and G_b
# and plot these singular values
u2_a, s2_a, vh_a = np.linalg.svd(G_a, full_matrices = True, compute_uv = True, hermitian = False)
u2_b, s2_b, vh_b = np.linalg.svd(G_a, full_matrices = True, compute_uv = True, hermitian = False)

sing_lst_a = np.arange(0, len(s2_a), 1, dtype = int)
sing_lst_b = np.arange(0, len(s2_b), 1, dtype = int)

plt.figure(figsize=(24, 6))
plt.scatter(sing_lst_a, s2_a, color = 'red')

plt.xticks(np.arange(0, ((2*J+1)*(2*K+1))+0.1, 1))
plt.xlabel('Indices')
plt.yticks(np.arange(0, max(s2_a)+0.1, 1))
plt.ylabel('Singular Values')
plt.title('Singular Values of the Gram Operator G_a_ijpq (descending order)')

plt.show()


plt.figure(figsize=(24, 6))
plt.scatter(sing_lst_b, s2_b, color = 'red')

plt.xticks(np.arange(0, ((2*J+1)*(2*K+1))+0.1, 1))
plt.xlabel('Indices')
plt.yticks(np.arange(0, max(s2_b)+0.1, 1))
plt.ylabel('Singular Values')
plt.title('Singular Values of the Gram Operator G_b_ijpq (descending order)')

plt.show()


# Teuncate singular values of G_a and G_b based based on 4% of the largest singular valuecof G
threshold_a = 1/(0.04*np.max(s2_a))      # Threshold value for truncated SVD of G_a
threshold_b = 1/(0.04*np.max(s2_b))      # Threshold value for truncated SVD of G_b



# Compute duall Gram operators G*_a and G*_b using pseudoinverses based on truncated singular values of G_a and G_b
G_dual_a = np.linalg.pinv(G_a, rcond = threshold_a)
G_dual_b = np.linalg.pinv(G_b, rcond = threshold_b)

print(G_dual_a[:2, :2])
print(G_dual_b[:2, :2])
# G_dual_mc = np.linalg.pinv(G_mc_weighted)
# %%



"""
Applying analysis operator T to the pushforwaed F_*v = vF
instead of direcrly to the vector field v
using Monte Carlo integration
to obtain v_hat'_a and v_hat'_b
for the latitude and meridian circles
"""


# (L2) Deterministic Monte Carlo integral of products 
# between eigenfunction phi_a_mn and "arrows" v_a_an, and phi_b_mn and "arrows" v_b_an
def monte_carlo_product_a(Phis_a, u_a, N = 800):
    v_a_an = v1F(u_a, u_b)[:2, :]
    integral_a = (1/N)*np.sum(Phis_a*v_a_an, axis = 1)
    
    return integral_a

def monte_carlo_product_b(Phis_b, u_b, N = 800):
    v_b_an = v1F(u_a, u_b)[2:4, :]
    integral_b = (1/N)*np.sum(Phis_b*v_b_an, axis = 1)
    
    return integral_b



# Compute b_am_a and b_am_b entries using (L2) deterministic Monte Carlo integrals
# for the latitude and meridian cicles
pool = mp.Pool()

def b_func_a(m):
    return monte_carlo_product_a(Phis_a_normalized[:, m], u_a)

def b_func_b(m):
    return monte_carlo_product_b(Phis_b_normalized[:, m], u_b)


b_am_a = pool.map(b_func_a, 
              [m for m in range(0, 2 * I + 1)])
b_am_a = np.array(b_am_a).T

b_am_b = pool.map(b_func_b, 
              [m for m in range(0, 2 * I + 1)])
b_am_b = np.array(b_am_b).T
# %%


# Apply analysis operator T to obtain v_hat_prime_a and v_hat_prime_b
# for the latitude and meridian circles
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km_a = np.einsum('ak, am -> km', F_ak[:2, :], b_am_a, dtype = float)
gamma_km_b = np.einsum('ak, am -> km', F_ak[2:4, :], b_am_b, dtype = float)

g_a = g_a[:(2*K+1), :, :]
g_b = g_b[:(2*K+1), :, :]

eta_qlm_a = np.einsum('qkl, km -> qlm', g_a, gamma_km_a, dtype = float)
eta_qlm_b = np.einsum('qkl, km -> qlm', g_b, gamma_km_b, dtype = float)


c_a = c_a[:(2*J+1), :, :]
c_b = c_b[:(2*J+1), :, :]

v_hat_prime_a = np.einsum('qlm, plm -> pq', eta_qlm_a, c_a, dtype = float)
v_hat_prime_b = np.einsum('qlm, plm -> pq', eta_qlm_b, c_b, dtype = float)


# Weighted v_hat_prime_a and v_hat_prime_b
for q in range(0, 2*K+1):
    v_hat_prime_a[:, q] = np.exp(-tau*lambs_a[q])*v_hat_prime_a[:, q]
    v_hat_prime_b[:, q] = np.exp(-tau*lambs_b[q])*v_hat_prime_b[:, q]

# v_hat_prime_mc_dm = np.reshape(np.array(v_hat_prime_mc_dm), ((2*J+1), (2*K+1)))
v_hat_prime_a = np.reshape(v_hat_prime_a, ((2*J+1)*(2*K+1), 1))
v_hat_prime_b = np.reshape(v_hat_prime_b, ((2*J+1)*(2*K+1), 1))


# print(v_hat_prime_a[:2,:2])
# print(v_hat_prime_b[:2,:2])


# Apply dual Gram operators G+_a amd G+_b to obtain v_hat_a and v_hat_b
# for the latitude and meridian circles
# using pushforward vF and original vector field v
# both with Monte Carlo integration with weights
v_hat_a = np.matmul(G_dual_a, v_hat_prime_a)
v_hat_a = np.reshape(v_hat_a, (2*J+1, 2*K+1))

v_hat_b = np.matmul(G_dual_b, v_hat_prime_b)
v_hat_b = np.reshape(v_hat_b, (2*J+1, 2*K+1))
# %%


# Apply pushforward map F_* of the embedding F to v_hat_a and v_hat_b to obtain approximated vector fields
# using Monte Carlo integration with weights

# Weighted Riemannian metric g_a and g_b
# for the latitude and meridian circles
g_a_weighted = np.zeros([2*K+1, 2*I+1, 2*I+1], dtype = float)
g_b_weighted = np.zeros([2*K+1, 2*I+1, 2*I+1], dtype = float)


for j in range(0, 2*K+1):
    g_a_weighted[j, :, :] = np.exp(-tau*lambs_a[j])*g_a[j, :, :]
    g_b_weighted[j, :, :] = np.exp(-tau*lambs_b[j])*g_b[j, :, :]


h_ajl_a = np.einsum('ak, jkl -> ajl', F_ak[:2, :], g_a_weighted, dtype = float)
h_ajl_b = np.einsum('ak, jkl -> ajl', F_ak[2:4, :], g_b_weighted, dtype = float)

# c_mc = c_mc[:(2*J+1), :, :]
d_jlm_a = np.einsum('ij, ilm -> jlm', v_hat_a, c_a, dtype = float)
d_jlm_b = np.einsum('ij, ilm -> jlm', v_hat_b, c_b, dtype = float)


p_am_a = np.einsum('ajl, jlm -> am', h_ajl_a, d_jlm_a, dtype = float)
p_am_b = np.einsum('ajl, jlm -> am', h_ajl_b, d_jlm_b, dtype = float)


W_theta_x_a = np.zeros(n, dtype = float)
W_theta_y_a = np.zeros(n, dtype = float)
vector_approx_a = np.empty([n, 4], dtype = float)

W_theta_x_b = np.zeros(n, dtype = float)
W_theta_y_b = np.zeros(n, dtype = float)
vector_approx_b = np.empty([n, 4], dtype = float)


def W_x_a(x, y):
    varphi_a_xy = np.real(varphi_a(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am_a[0, :]*varphi_a_xy)

def W_y_a(x, y):
    varphi_a_xy = np.real(varphi_a(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am_a[1, :]*varphi_a_xy)

def W_x_b(x, y):
    varphi_b_xy = np.real(varphi_b(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am_b[0, :]*varphi_b_xy)

def W_y_b(x, y):
    varphi_b_xy = np.real(varphi_b(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am_b[1, :]*varphi_b_xy)


for i in range(0, n):
    W_theta_x_a[i] = W_x_a(TRAIN_X[i], TRAIN_Y[i])
    W_theta_y_a[i] = W_y_a(TRAIN_X[i], TRAIN_Y[i])
    vector_approx_a[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x_a[i], W_theta_y_a[i]])
    
    W_theta_x_b[i] = W_x_b(TRAIN_Z[i], TRAIN_W[i])
    W_theta_y_b[i] = W_y_b(TRAIN_Z[i], TRAIN_W[i])
    vector_approx_b[i, :] = np.array([TRAIN_Z[i], TRAIN_W[i], W_theta_x_b[i], W_theta_y_b[i]])
print(W_theta_x_a)
print(W_theta_y_a)
print(W_theta_x_b)
print(W_theta_y_b)


X_a_approx, Y_a_approx, U_a_approx, V_a_approx = zip(*vector_approx_a)
X_b_approx, Y_b_approx, U_b_approx, V_b_approx = zip(*vector_approx_b)


# Comparison between true pusbforward of vector field and pushforward of SEC approximated vector field
# on thr latitude and meridian circles
plt.figure()
ax = plt.gca()
ax.quiver(X_a, Y_a, U_a, V_a, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'black')
ax.quiver(X_a_approx, Y_a_approx, U_a_approx, V_a_approx, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'red')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_title('Comparisons of True and SEC Approximated Vector Fields on the Latitude Circle with Radius a')

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(a*np.cos(t), a*np.sin(t), linewidth = 2.5, color = 'blue')

plt.draw()
plt.show()


sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sidefig.suptitle('Comparisons of True and SEC Approximated Vector Fields on the Latitude Circle with Radius a')

ax1.scatter(x = THETA_LST, y = -TRAIN_Y, color='black')
ax1.scatter(x = THETA_LST, y = W_theta_x_a, color='red')
ax1.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax1.set_xlabel("Angle Theta")
ax1.set_ylabel("X-coordinates of Vector Fields")
ax1.set_title('X-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

ax2.scatter(x = THETA_LST, y = TRAIN_X, color='black')
ax2.scatter(x = THETA_LST, y = W_theta_y_a, color='red')
ax2.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax2.set_xlabel("Angle Theta")
ax2.set_ylabel("Y-coordinates of Vector Fields on the Latitude Circle with Radius a")
ax2.set_title('Y-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

plt.show()



plt.figure()
ax = plt.gca()
ax.quiver(X_b, Y_b, U_b, V_b, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'black')
ax.quiver(X_b_approx, Y_b_approx, U_b_approx, V_b_approx, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'red')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_title('Comparisons of True and SEC Approximated Vector Fields on the Meridian Circle with Radius a')

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(b*np.cos(t), b*np.sin(t), linewidth = 2.5, color = 'blue')

plt.draw()
plt.show()


sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sidefig.suptitle('Comparisons of True and SEC Approximated Vector Fields on the Meridian Circle with Radius a')

ax1.scatter(x = RHO_LST, y = -TRAIN_W, color='black')
ax1.scatter(x = RHO_LST, y = W_theta_x_b, color='red')
ax1.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax1.set_xlabel("Angle Rho")
ax1.set_ylabel("X-coordinates of Vector Fields")
ax1.set_title('X-coordinates w.r.t. Angle Rho (true = black, SEC = red)')

ax2.scatter(x = RHO_LST, y = TRAIN_Z, color='black')
ax2.scatter(x = RHO_LST, y = W_theta_y_b, color='red')
ax2.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax2.set_xlabel("Angle Rho")
ax2.set_ylabel("Y-coordinates of Vector Fields on the Latitude Circle with Radius a")
ax2.set_title('Y-coordinates w.r.t. Angle Rho (true = black, SEC = red)')

plt.show()
# %%
