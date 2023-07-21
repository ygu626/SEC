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
# the latotude and meridian circles with radius a and b
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
THETA_LST, RHO_LST = np.meshgrid(u_a, u_b)
      
    
X_func = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
Y_func = lambda theta, rho: (a + b*np.cos(theta))*np.sin(rho)
Z_func = lambda theta: b*np.sin(theta)


TRAIN_X = X_func(THETA_LST, RHO_LST)
TRAIN_Y = Y_func(THETA_LST, RHO_LST)
TRAIN_Z = Z_func(THETA_LST)

x = (a + b*np.cos(THETA_LST))*np.cos(RHO_LST)
y = (a + b*np.cos(THETA_LST))*np.sin(RHO_LST)
z = b*np.sin(THETA_LST)
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


fig = plt.figure()

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_zlim(-3,3)
ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
ax1.view_init(36, 26)

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_zlim(-3,3)
ax2.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
ax2.view_init(0, 0)
ax2.set_xticks([])

plt.show()
# %%


# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta, rho: np.array([(a + b*np.cos(theta))*np.cos(rho), (a + b*np.cos(theta))*np.sin(rho), a + b*np.sin(theta)])

v1F = lambda theta, rho: np.array([-b*np.sin(theta)*np.cos(rho) - (a + b*np.cos(theta))*np.sin(rho), -b*np.sin(theta)*np.sin(rho) + (a + b*np.cos(theta))*np.cos(rho), b*np.cos(theta)])
# %%



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
q = make_normalization_func(k, training_data_b)
k_hat = make_k_hat(k, q)
K_hat = k_hat(training_data_a, training_data_b)
# print(K_hat[:2,:2])


# Normalization function d that corresponds to diagonal matrix D
d = make_normalization_func(k_hat, training_data_b)
D = d(training_data_a)


# Markov kernel function p
def make_p(k_hat, d):
    def p(x, y):
        d_x = d(x).reshape(d(x).shape[0], 1)

        p_xy = np.divide(k_hat(x, y), d_x)
        return p_xy
    return p

# Build Markov kernel matrix P
p = make_p(k_hat, d)
P = p(training_data_a, training_data_b)
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
S = s(training_data_a, training_data_b)
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
            # lambs_dm[i] = (1 - np.real(Lambs[i]))/(epsilon**2)   

print(lambs_dm)         
# %%

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
varphi = make_varphi(p, training_data_b, Lambs, Phis_normalized)
# %%


# %%
"""
Check accuracy of diffusion maps approximation
fir eigenvalues and eigenfunctions of 0-Laplacian
"""

# Check approximations for Laplacian eigenbasis agree with true eigenbasis
# by ploting against linear combinations of true eigenfunctions 

# Get x values of the sine wave
time = u_a
time2 = u_b

# Amplitude of the sine wave is sine of a variable like time
amplitude = Phis_normalized[:, 1]
amplitude2 = np.real(varphi(training_data_b)[:, 1])

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



# %%
"""
SEC approximation
for pushforward of vector fields on the 2-torus embedded in R3
"""


# Fourier coefficients F_ak pf F w.r.t. difusion maps approximated eigenvectors Phi_j
F_aok = (1/N)*np.matmul(F(THETA_LST, RHO_LST), Phis_normalized)


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
print(c[:,3,3])
# %%

# %%
# Compute g_ijp Riemannian metric coefficients
# using Monte Carlo integration
g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
g_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for p in range(0, 2*I+1):
                                    g_coeff[i,j,p] = (lambs_dm[i] + lambs_dm[j] - lambs_dm[p])/2

# g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
# for i in range(0, 2*I+1):
#             for j in range(0, 2*I+1):
#                         for p in range(0, 2*I+1):
#                                     g[i,j,p] = (lambs[i] + lambs[j] - lambs[p])*c[i,j,p]/2
         
print(g[:,3,3])
# %%


# %%
# Compute G_ijpq entries for the Gram operator and its dual
# using Monte Carlo integration
G = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G = np.einsum('ipm, jqm -> ijpq', c, g, dtype = float)

G = G[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G = np.reshape(G, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))

print(G[:3:3])
# %%


# %%
# Perform singular value decomposition (SVD) of the Gram operator G
# and plot these singular values
u2, s2, vh = np.linalg.svd(G, full_matrices = True, compute_uv = True, hermitian = False)
# %%

# %%
sing_lst = np.arange(0, len(s2), 1, dtype = int)
plt.figure(figsize=(24, 6))
plt.scatter(sing_lst, s2, color = 'red')

plt.xticks(np.arange(0, ((2*J+1)*(2*K+1))+0.1, 1))
plt.xlabel('Indices')
plt.yticks(np.arange(0, max(s2)+0.1, 1))
plt.ylabel('Singular Values')
plt.title('Singular Values of the Gram Operator G_ijpq (descending order)')

plt.show()
# %%


# %%
# Teuncate singular values of G based based on a small percentage of the largest singular valuecof G
threshold = 1/(0.1*np.max(s2))      # Threshold value for truncated SVD


# Compute duall Gram operator G* using pseudoinverse based on truncated singular values of G
G_dual = np.linalg.pinv(G, rcond = threshold)
# G_dual_mc = np.linalg.pinv(G_mc_weighted)
# %%



# %%
"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""


# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product(Phis, u_a, u_b, N = 800):
    v_an = v1F(THETA_LST, RHO_LST)
    integral = (1/N)*np.sum(Phis*v_an, axis = 1)
    
    return integral



# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func(m):
    return monte_carlo_product(Phis_normalized[:, m], THETA_LST, RHO_LST)


b_aom = pool.map(b_func, 
                [m for m in range(0, 2 * I + 1)])


b_aom = np.reshape(np.array(b_aom), [3, N, 2 *I + 1])


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_kom = np.einsum('aok, aom -> kom', F_aok, b_aom, dtype = float)
# %%

# %%
g = g[:(2*K+1), :, :]


eta_qlom = np.einsum('qkl, kom -> qlom', g, gamma_kom, dtype = float)

c = c[:(2*J+1), :, :]


v_hat_prime = np.einsum('qlom, plm -> pqo', eta_qlom, c, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime[:, q, :] = np.exp(-tau*lambs[q])*v_hat_prime[:, q, :]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime = np.reshape(v_hat_prime, ((2*J+1)*(2*K+1), N))
# print(v_hat_prime[:3,:3])
# %%

# %%
# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat = np.matmul(G_dual, v_hat_prime)
v_hat = np.reshape(v_hat, (2*J+1, 2*K+1, N))

# %%



# %%
# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights

# g = g[:(2*K+1), :, :]

# Weighted g_ijp Riemannian metric coefficients
g_weighted = np.zeros([2*K+1, 2*I+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted[j, :, :] = np.exp(-tau*lambs[j])*g[j, :, :]


h_ajol = np.einsum('aok, jkl -> ajol', F_aok, g_weighted, dtype = float)

# c = c[:(2*J+1), :, :]
d_jlom = np.einsum('ijo, ilm -> jlom', v_hat, c, dtype = float)

p_am = np.einsum('ajol, jlom -> am', h_ajol, d_jlom, dtype = float)
# %%

# %%
W_theta_x = np.zeros(n, dtype = float)
W_theta_y = np.zeros(n, dtype = float)
W_theta_z = np.zeros(n, dtype = float)

vector_approx = np.empty([n, 6], dtype = float)

A = varphi([THETA_LST, RHO_LST])
print(A.shape)
# %%



def W_x(x):
    varphi_x = np.real(varphi(x))
    return np.sum(p_am[0, :]*varphi_x)

def W_y(y):
    varphi_y = np.real(varphi(y))
    return np.sum(p_am[1, :]*varphi_y)

def W_z(z):
    varphi_z = np.real(varphi(z))
    return np.sum(p_am[2, :]*varphi_z)


for i in range(0, n):
    W_theta_x[i] = W_x(TRAIN_X[i])
    W_theta_y[i] = W_y(TRAIN_Y[i])
    W_theta_z[i] = W_z(TRAIN_Z[i])

    vector_approx[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], W_theta_x[i], W_theta_y[i], W_theta_z[i]])

print(W_theta_x)
print(W_theta_y)
print(W_theta_z)
# %%

X_2, Y_2, U_2, V_2 = zip(*vector_approx)


# Comparison between true pusbforward of vector field and pushforward of SEC approximated vector field
plt.figure()
ax = plt.gca()
ax.quiver(X_1, Y_1, U_1, V_1, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'red')
ax.quiver(X_2, Y_2, U_2, V_2, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'blue')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_title('Comparisons of True and SEC Approximated Vector Fields')

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth = 2.5, color = 'black')

plt.draw()
plt.show()


sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sidefig.suptitle('Comparisons of True and SEC Approximated Vector Fields')

ax1.scatter(x = THETA_LST, y = -TRAIN_Y, color='red')
ax1.scatter(x = THETA_LST, y = W_theta_x, color='blue')
ax1.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax1.set_xlabel("Angle Theta")
ax1.set_ylabel("X-coordinates of Vector Fields")
ax1.set_title('X-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

ax2.scatter(x = THETA_LST, y = TRAIN_X, color='red')
ax2.scatter(x = THETA_LST, y = W_theta_y, color='blue')
ax2.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax2.set_xlabel("Angle Theta")
ax2.set_ylabel("Y-coordinates of Vector Fields")
ax2.set_title('Y-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

plt.show()
# %%
