# %%
"""
Spectral Exterior Calculus (SEC)
2-torus T2 Example (Roration)
Approximations of vector fields on the 2-torus
usng donut embedding into R^3
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
from scipy.spatial import distance_matrix
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import multiprocess as mp


# Parameters
I = 100          # Inner index for eigenfunctions
J = 90           # Outer index for eigenfunctions
K = 40           # Index for gradients of eigenfunctions
n = 100          # Number of approximated tangent vectors
N = 100         # Number of Monte Carlo training data points 

# epsilon = 0.31622777    # RBF bandwidth parameter
# epsilon = 0.30902954    # RBF bandwidth parameter
epsilon = 0.125

tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix
a = 0.64618101811           # Radius of the latitude circle of the torus
b = 0.28           # Radius of the meridian circle of the torus
R = 1
r = 1


"""
Training data set
with pushforward of vector fields v on the torus
and smbedding map F with pushforward F_*v = vF
"""


# Deterministically sampled Monte Carlo training data points
# the latotude and meridian circles with radius a and b
def monte_carlo_points(start_pt = 0, end_pt = 2*np.pi, N = 100, a = 1, b = 1):
    u_a = np.arange(start_pt, end_pt, 2*np.pi/N)
    u_b = np.arange(start_pt, end_pt, 2*np.pi/N)
    
    # subsets = np.arange(0, N+1, (N/50))
    # for i in range(0, int(N/2)):
    #    start = int(subsets[i])
    #    end = int(subsets[i+1])
    #    u_a[start:end] = random.uniform(low = (i/(N/2))*end_pt, high = ((i+1)/(N/2))*end_pt, size = end - start)
    #    u_b[start:end] = random.uniform(low = (i/(N/2))*end_pt, high = ((i+1)/(N/2))*end_pt, size = end - start)
    
    # random.shuffle(u_a)
    # random.shuffle(u_b)

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
# Parameterization functions specifying the coordinates in R^3
# corresponding to donut torus embedding
# using the angles theat and rho for the latitude and meridian circles
X_func = lambda theta, rho: (a + b*np.cos(rho))*np.cos(theta)
Y_func = lambda theta, rho: (a + b*np.cos(rho))*np.sin(theta)
Z_func = lambda rho: b*np.sin(rho)

# N*N training data points corrdinates in the x, y and z coordinates
TRAIN_X = X_func(training_angle[0, :], training_angle[1, :])
TRAIN_Y = Y_func(training_angle[0, :], training_angle[1, :])
TRAIN_Z = Z_func(training_angle[1, :])

# N*N training data points containing all three coordinates of each point
training_data = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z])
# %%

# %%
# Parameterization functions specifying the coordinates in R^4
# correspondong to flat torus embedding
# using the angles theat and rho for the latitude and meridian circles

X_func_flat = lambda theta: R*np.cos(theta)
Y_func_flat = lambda theta: R*np.sin(theta)
Z_func_flat = lambda rho: r*np.cos(rho)
W_func_flat = lambda rho: r*np.sin(rho)

# N*N training data points corrdinates in the x, y and z coordinates
TRAIN_X_flat = X_func_flat(training_angle[0, :])
TRAIN_Y_flat = Y_func_flat(training_angle[0, :])
TRAIN_Z_flat = Z_func_flat(training_angle[1, :])
TRAIN_W_flat = W_func_flat(training_angle[1, :])


# N*N training data points containing all four coordinates of each point
training_data_flat = np.vstack([TRAIN_X_flat, TRAIN_Y_flat, TRAIN_Z_flat, TRAIN_W_flat])
# %%




# %%
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)


# ax.plot_surface(x_t_ode, y_t_ode, z_t_ode, antialiased=True, alpha = 0.6, color='orange')

 
ax.scatter3D(training_data[0, :], training_data[1, :], training_data[2, :], color = "green")
plt.title("Solutions to ODE under the true system on the torus")
 
plt.show()
# %%



# x = (a + b*np.cos(training_angle[1, :]))*np.cos(training_angle[0, :])
# y = (a + b*np.cos(training_angle[1, :]))*np.sin(training_angle[0, :])
# z = b*np.sin(training_angle[1, :])



# Partial detivatives of the parameterization functions w.r.t. theta
# X_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.cos(rho)
# Y_func_dtheta = lambda theta, rho: -b*np.sin(theta)*np.sin(rho)
# Z_func_dtheta = lambda theta: b*np.cos(theta)


# Partial detivatives of the parameterization functions w.r.t. rho
# X_func_drho = lambda theta, rho: -(a + b*np.cos(theta))*np.sin(rho)
# Y_func_drho = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
# Z_func_drho = lambda theta: 0


# N*N partial derivative training data point corrdinates w.r.t. theta in the x, y and z coordinates
# TRAIN_X_dtheta = X_func_dtheta(training_angle[0, :], training_angle[1, :])
# TRAIN_Y_dtheta = Y_func_dtheta(training_angle[0, :], training_angle[1, :])
# TRAIN_Z_dtheta = Z_func_dtheta(training_angle[0, :])

# N*N partial derivative training data point corrdinates w.r.t. rho in the x, y and z coordinates
# TRAIN_X_drho = X_func_drho(training_angle[0, :], training_angle[1, :])
# TRAIN_Y_drho = Y_func_drho(training_angle[0, :], training_angle[1, :])
# TRAIN_Z_drho = Z_func_drho(training_angle[0, :])

# N*N linear combinations of partial derivatives of training data point corrdinates in the x, y and z coordinates
# TRAIN_X_DERIVATIVE = TRAIN_X_dtheta + TRAIN_X_drho
# TRAIN_Y_DERIVATIVE = TRAIN_Y_dtheta + TRAIN_Y_drho
# TRAIN_Z_DERIVATIVE = TRAIN_Z_dtheta + TRAIN_Z_drho


# N*N analytic directional coordinates of tangent vectors
# originated from N*N training data points in R^3
# ana_dir_coords = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, TRAIN_X_DERIVATIVE, TRAIN_Y_DERIVATIVE, TRAIN_Z_DERIVATIVE])
# %%



# %%
"""
Functions utilized in the following program
"""

# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta, rho: np.array([(a + b*np.cos(rho))*np.cos(theta), (a + b*np.cos(rho))*np.sin(theta), b*np.sin(rho)])
v1F = lambda theta, rho: np.array([-b*np.sin(rho)*np.cos(theta) - (a + b*np.cos(rho))*np.sin(theta), -b*np.sin(rho)*np.sin(theta) + (a + b*np.cos(rho))*np.cos(theta), b*np.cos(rho)])

# Analytical tangent vector coordinates
ana_dir_coords = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, v1F(training_angle[0, :], training_angle[1, :])])


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
q = make_normalization_func(k, training_data_flat)
k_hat = make_k_hat(k, q)
K_hat = k_hat(training_data_flat, training_data_flat)
# print(K_hat[:2,:2])
# %%



# %%
# Normalization function d that corresponds to diagonal matrix D
d = make_normalization_func(k_hat, training_data_flat)
D = d(training_data_flat)
# %%


# %%
# Markov kernel function p
def make_p(k_hat, d):
    def p(x, y):
        d_x = d(x).reshape(1, d(x).shape[0])

        p_xy = np.divide(k_hat(x, y), d_x)
        return p_xy
    return p

# Build Markov kernel matrix P
p = make_p(k_hat, d)
P = p(training_data_flat, training_data_flat)
# print(P[:3,:3])

print(np.trace(P))
print(np.pi/(4*epsilon**2))
# %%


# %%
# Similarity transformation function s
def make_s(p, d):
    def s(x, y):
        d_x = np.power(d(x).reshape(d(x).shape[0], 1), (1/2))
        d_y = np.power(d(y).reshape(1, d(y).shape[0]), (1/2))
        
        s_xy = np.divide(np.multiply(d_x, p(x, y)), d_y)
        return s_xy
    return s

# Build Similarity matrix S
s = make_s(p, d)
S = s(training_data_flat, training_data_flat)
# print(S[:3,:3])
# %%


# %%
# Solve eigenvalue problem for similarity matrix S
# eigenvalues, eigenvectors = eigs(P, k = 300) 
eigenvalues, eigenvectors = eig(S)
index = eigenvalues.argsort()[::-1][:2*I+1]
Lambs = eigenvalues[index]
Phis = np.real(eigenvectors[:, index])
# %%

# %%
# Compute approximated 0-Laplacian eigengunctions
lambs = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
            # lambs[i] = (4)*(-np.log(np.real(Lambs[i]))/(epsilon**2))
            lambs[i] = 4*(1 - np.real(Lambs[i]))/(epsilon**2)   

print(Lambs)         



# Normalize eigenfunctions Phi_j
Phis_normalized = np.empty([N**2, 2*I+1], dtype = float)
D_sqrt = np.power(D.reshape(1, D.shape[0]), (1/2))
for j in range(0, 2*I+1):
    Phis_normalized[:, j] = np.divide(np.real(Phis[:, j]), D_sqrt)

Phis_normalized = Phis_normalized/Phis_normalized[0, 0]

# %%


# %%
print(np.dot(Phis[:, 4], Phis[:, 2]))
# %%

# %%
print(lambs/(4*np.pi**2))
print(Phis_normalized[:, 0])
print(np.max(Phis_normalized[:, 0]))
print(np.min(Phis_normalized[:, 0]))
# %%



# %%
# Appeoximate eigenvalues and eigenfunctions for the 0-Laplacian
def make_varphi(k, x_train, lambs, phis):
    phi_lamb = np.real(phis / lambs)
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    return varphi

# Produce continuous extentions varphi_j for the eigenfunctions Phi_j
# varphi = make_varphi(p, training_data, Lambs, Phis_normalized)
varphi_flat = make_varphi(p, training_data_flat, Lambs, Phis_normalized)
# %%




# %%
# Apply the coninuous extensiom varphi to the training data set
# varphi_xyzw = varphi(training_data)
varphi_xyzw = varphi_flat(training_data_flat)

# print(varphi_xyz[:,3])
# %%

# %%
"""
Check accuracy of diffusion maps approximation
for eigenvalues and eigenfunctions of 0-Laplacian
"""


z_true = np.reshape(Phis_normalized[:, 1], (N, N))
z_dm = np.reshape(np.real(varphi_xyzw[:, 2]), (N, N))

plt.figure(figsize=(12, 12))
plt.pcolormesh(THETA_LST, RHO_LST, z_dm)

plt.show()
# %%


# %%
# Slice of the heat map
# for specific theta (latitude circle angle) values
y_test = np.reshape(varphi_xyzw[:, 89], (100, 100))

print(np.amax(y_test))
print(np.amin(y_test))


plt.scatter(u_a, y_test[0, :])
plt.show 
# %%



# %%
"""
SEC approximation
for pushforward of vector fields on the 2-torus embedded in R3
"""


# Fourier coefficients F_ak pf F w.r.t. difusion maps approximated eigenvectors Phi_j
F_ak = (1/(N**2))*np.matmul(F(training_angle[0, :], training_angle[1, :]), Phis_normalized)

# print(F_ak[:, 2])



# Compute c_ijp coefficients
# using Monte Carlo integration
pool = mp.Pool()

def c_func(i, j, p):
    return (1/(N**2))*np.sum(Phis_normalized[:, i]*Phis_normalized[:, j]*Phis_normalized[:, p])

c = pool.starmap(c_func, 
              [(i, j, p) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for p in range(0, 2 * I + 1)])
            
c = np.reshape(np.array(c), (2 * I + 1, 2 * I + 1, 2 * I + 1))
# print(c[:2,:2,:2])



# Compute g_ijp Riemannian metric coefficients
# using Monte Carlo integration
g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
# g_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

# for p in range(0, 2*I+1):
#    for i in range(0, 2*I+1):
#        for j in range(0, 2*I+1):
#            g_coeff[p, i,j] = (lambs[i] + lambs[j] - lambs[p])/2
#
# g = np.multiply(g_coeff, c)



for p in range(0, 2*I+1):
    for i in range(0, 2*I+1):
        for j in range(0, 2*I+1):
            g[p,i,j] = (lambs[i] + lambs[j] - lambs[p])*c[i,j,p]/2
         
# print(g[6:8,12:14,:2])
# %%




# %%
# Compute G_ijpq entries for the Gram operator and its dual
# using Monte Carlo integration
# G = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G = np.einsum('mip, mjq -> ijpq', c, g, dtype = float)

G = G[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G = np.reshape(G, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))

print(G[:2,:2])


# Perform singular value decomposition (SVD) of the Gram operator G
# and plot these singular values
s2 = np.linalg.svd(G, full_matrices = True, compute_uv = False, hermitian = False)


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
print(np.max(s2))
print(np.min(s2))
# %%


# %%
# Teuncate singular values of G based based on a small percentage of the largest singular valuecof G
threshold = 0.05    # Threshold value for truncated SVD

# Compute duall Gram operator G* using pseudoinverse based on truncated singular values of G
# G_dual = np.linalg.pinv(G)

G_dual = np.linalg.pinv(G, rcond = threshold)
# G_dual_mc = np.linalg.pinv(G_mc_weighted)



"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""

v1F_flat = lambda theta, rho: np.array([-a*np.sin(theta), a*np.cos(theta), -b*np.sin(rho), b*np.cos(rho)])

# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product(Phis, training_angle, N = 100):
    v_an = v1F(training_angle[0, :], training_angle[1, :])
    integral = (1/(N**2))*np.sum(Phis*v_an, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func(m):
    return monte_carlo_product(Phis_normalized[:, m], training_angle)


b_am = pool.map(b_func, 
                [m for m in range(0, 2 * I + 1)])

b_am = np.array(b_am).T


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km = np.einsum('ak, am -> km', F_ak, b_am, dtype = float)


g = g[:(2*K+1), :, :]

eta_qlm = np.einsum('qkl, km -> qlm', g, gamma_km, dtype = float)


c = c[:(2*J+1), :, :]


v_hat_prime = np.einsum('qlm, plm -> pq', eta_qlm, c, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime[:, q] = np.exp(-tau*lambs[q])*v_hat_prime[:, q]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime = np.reshape(v_hat_prime, ((2*J+1)*(2*K+1)))
# print(v_hat_prime[:10])



# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat = np.matmul(G_dual, v_hat_prime)
v_hat = np.reshape(v_hat, (2*J+1, 2*K+1))


# print(np.amax(v_hat))
# print(np.amin(v_hat))
# %%




# %%
# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights

# g = g[:(2*K+1), :, :]

# Weighted g_ijp Riemannian metric coefficients
g_weighted = np.zeros([2*K+1, 2*I+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted[j, :, :] = np.exp(-tau*lambs[j])*g[j, :, :]


h_ajl = np.einsum('ak, jkl -> ajl', F_ak, g_weighted, dtype = float)


# c = c[:(2*J+1), :, :]
d_jlm = np.einsum('ij, ilm -> jlm', v_hat, c, dtype = float)

p_am = np.einsum('ajl, jlm -> am', h_ajl, d_jlm, dtype = float)


W_theta_x = np.zeros(int(N**2), dtype = float)
W_theta_y = np.zeros(int(N**2), dtype = float)
W_theta_z = np.zeros(int(N**2), dtype = float)


def W_theta(varphi_xyzw):
    varphi_xyzw = np.real(varphi_xyzw)
    
    for i in range(0, int(N**2)):
        W_theta_x[i] = np.sum(p_am[0, :]*varphi_xyzw[i, :])
        W_theta_y[i] = np.sum(p_am[1, :]*varphi_xyzw[i, :])
        W_theta_z[i] = np.sum(p_am[2, :]*varphi_xyzw[i, :])

    return W_theta_x, W_theta_y, W_theta_z

# print(varphi_xyzw[:, 3])

W_x, W_y, W_z = W_theta(varphi_xyzw)

# W_x = W_x / np.sqrt(W_x**2 + W_y**2 + W_z**2)
# W_y = W_y / np.sqrt(W_x**2 + W_y**2 + W_z**2)
# W_z = W_z / np.sqrt(W_x**2 + W_y**2 + W_z**2)

vector_approx = np.empty([int(N**2), 6], dtype = float)
for i in range(0, int(N**2)):
    vector_approx[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], W_x[i], W_y[i], W_z[i]])

print(vector_approx[:, 2])
# %%


# %%
# Compute coefficient of determination R^2
# using the analytic and SEC approximated directional tangent vector coordinates
vec_ana = ana_dir_coords[3:6, :]
vec_sec = vector_approx[:, 3:6].T

ana_norm = np.sqrt(np.sum(np.power(vec_ana, 2), axis = 0))
sec_norm = np.sqrt(np.sum(np.power(vec_sec, 2), axis = 0))

norm_ratio = ana_norm/sec_norm

# print(np.amax(norm_ratio))
# print(np.amin(norm_ratio))
# print(norm_ratio)
print(np.amin(sec_norm))
print(np.amax(sec_norm))

scaling_factor = np.mean(norm_ratio)

rss = np.sum(np.power((vec_ana - vec_sec), 2))

vec_bar = np.mean(vec_ana, axis = 1)
tss = np.sum(np.power(vec_ana, 2))


R_squared = 1 - rss/tss

print(rss)
print(tss)
print(R_squared)
# %%


#%%
def plot_torus(precision, a = 0.64618101811, b = 0.28):
    U_t = np.linspace(0, 2*np.pi, precision)
    V_t = np.linspace(0, 2*np.pi, precision)
    
    U_t, V_t = np.meshgrid(U_t, V_t)
    
    X_t = (a + b*np.cos(U_t))*np.cos(V_t)
    Y_t = (a + b*np.cos(U_t))*np.sin(V_t)
    Z_t = b*np.sin(U_t)
    
    random_num = 200    # for 500 random indices
    random_index = np.random.choice(vector_approx.shape[0], random_num, replace = False)  

    
    return X_t, Y_t, Z_t, random_index

    

x_t, y_t, z_t, rd_idx = plot_torus(100, 0.64618101811, 0.28)
print(rd_idx)


vector_ana_shuffled = (ana_dir_coords.T)[rd_idx]
vector_approx_shuffled = vector_approx[rd_idx]


x2 = vector_approx_shuffled[:, 0]
y2 = vector_approx_shuffled[:, 1]
z2 = vector_approx_shuffled[:, 2]
   
a2 = vector_approx_shuffled[:, 3]
b2 = vector_approx_shuffled[:, 4]
c2 = vector_approx_shuffled[:, 5]

x3 = vector_ana_shuffled[:, 0]
y3 = vector_ana_shuffled[:, 1]
z3 = vector_ana_shuffled[:, 2]
   
a3 = vector_ana_shuffled[:, 3]
b3 = vector_ana_shuffled[:, 4]
c3 = vector_ana_shuffled[:, 5]
    

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.set_title('SEC approximated vector field on a torus embedded in R^3')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

ax.plot_surface(x_t, y_t, z_t, antialiased=True, color='orange')
ax.quiver(x2, y2, z2, a2, b2, c2, length = 1, color = 'blue')


ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.set_title('Analytic vector field on a torus embedded in R^3')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

ax.plot_surface(x_t, y_t, z_t, antialiased=True, color='orange')
ax.quiver(x3, y3, z3, a3, b3, c3, length = 1, color = 'red')

  
plt.show()



ax2 = plt.axes(projection = '3d')

ax2.set_title('Comparisons of analytic and SEC approximated vector fields on a torus embedded in R^3')
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(-1,1)

ax2.plot_surface(x_t, y_t, z_t, antialiased=True, color='orange')

ax2.quiver(x2, y2, z2, a2, b2, c2, length = 1, color = 'blue')
ax2.quiver(x3, y3, z3, a3, b3, c3, length = 1, color = 'red')

plt.show()


# %%



# ODE solver applied to the SEC approximated vector fields
# with initial condition specified
# and the true system


# %%
"""
True system given by analytical solutions
"""

# Define time spans and initial values for the true system
tspan = np.linspace(0, 20, num=2000)
theta_0 = 0
rho_0 = 0

# Anakytical solutions to dydt = v|y = (1, 1) where v = (ddtheta, ddrho) and y = (theta, rho)
theta_t = lambda t: theta_0 + t
rho_t = lambda t: rho_0 + t

sol_true_theta = theta_t(tspan)
sol_true_rho = rho_t(tspan)

# sol_true_a = np.empty([2, 1000], dtype = float)
# sol_true_b = np.empty([2, 1000], dtype = float)


# for j in range(0, 1000):    
#    sol_true_a[:, j] = np.array([np.cos(sol_true_theta[j]), np.sin(sol_true_theta[j])])
#    sol_true_b[:, j] = np.array([np.cos(sol_true_rho[j]), np.sin(sol_true_rho[j])])


# THETA_SOL_TRUE, RHO_SOL_TRUE = np.meshgrid(sol_true_theta, sol_true_rho)

# sol_true_angle = np.vstack([THETA_SOL_TRUE.ravel(), RHO_SOL_TRUE.ravel()])


SOL_TRUE_X = X_func(sol_true_theta, sol_true_rho)
SOL_TRUE_Y = Y_func(sol_true_theta, sol_true_rho)
SOL_TRUE_Z = Z_func(sol_true_theta)

sol_true_xyz_coords = np.vstack([SOL_TRUE_X, SOL_TRUE_Y, SOL_TRUE_Z])


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
x_t_ode, y_t_ode, z_t_ode, rd_idx2 = plot_torus(100, 0.64618101811, 0.28)

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)


# ax.plot_surface(x_t_ode, y_t_ode, z_t_ode, antialiased=True, alpha = 0.6, color='orange')

 
ax.scatter3D(SOL_TRUE_X, SOL_TRUE_Y, SOL_TRUE_Z, color = "green")
plt.title("Solutions to ODE under the true system on the torus")
 
plt.show()
# %%



# Plot solutions to the true system
sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(x = sol_true_a[0,:], y = sol_true_a[1,:], color = 'green')
ax1.set_xlim([-3,3])
ax1.set_ylim([-3,3])
ax1.set_title('Solution angles theta to the ODE under the true system')

ax2.scatter(x = sol_true_b[0,:], y = sol_true_b[1,:], color = 'orange')
ax2.set_xlim([-3,3])
ax2.set_ylim([-3,3])
ax2.set_title('Solution angles rho to the ODE under the true system')

plt.show()


sidefig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(24, 16))
sidefig.suptitle('Solution coordinates to the ODE under the true system')

ax1.plot(tspan, SOL_TRUE_X, color='red')
ax1.set_title('x--coordinates prediction w.r.t. time t')

ax2.plot(tspan, SOL_TRUE_Y, color='blue')
ax2.set_title('y-coordinates prediction w.r.t. time t')

ax3.plot(tspan, SOL_TRUE_Z, color='brown')
ax3.set_title('z-coordinates prediction w.r.t. time t')

plt.show()
# %%



# %%
"""
SEC Approximated System
"""

def W_theta_ode(y):
    varphi_xyz = np.real(varphi_flat(np.reshape(np.array(y), (4, 1))))
    W_x = np.sum(p_am[0, :]*varphi_xyz)
    W_y = np.sum(p_am[1, :]*varphi_xyz)
    W_z = np.sum(p_am[2, :]*varphi_xyz)
    
    return W_x, W_y, W_z



# Define derivative function for the SEC approximated system
def f_sec(t, y):
    W_x, W_y, W_z = W_theta_ode(y)
    dydt = [W_x, W_y, W_z]
    # dydt = [0,0,0]
    return dydt

# Define time spans and initial values for the SEC approximated system
tspan = np.linspace(0, 20, num=2000)
yinit = [a, 0, b, 0]


# Solve ODE under the SEC approximated system
sol_sec = solve_ivp(lambda t, y: f_sec(t, y),
                    [tspan[0], tspan[-1]], yinit, t_eval=tspan, rtol = 1e-5)

# print(sol_sec.shape)
# %%

# %%
SOL_SEC_X = sol_sec.y[0, :]
SOL_SEC_Y = sol_sec.y[1, :]
SOL_SEC_Z = sol_sec.y[2, :]


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
x_t_ode, y_t_ode, z_t_ode, rd_idx2 = plot_torus(100, 0.64618101811, 0.28)

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

# ax.plot_surface(x_t_ode, y_t_ode, z_t_ode, antialiased=True, alpha = 0.6, color='orange')

 
ax.scatter3D(SOL_SEC_X, SOL_SEC_Y, SOL_SEC_Z, color = "green")
plt.title("Solutions to ODE under the SEC approximated system on the torus")
 
plt.show()

# %%



# Plot solutions to the SEC approximated system
plt.figure(figsize=(8, 8))
plt.plot(sol_sec.y.T[:, 0], sol_sec.y.T[:, 1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solutions to ODE under the SEC approximated system')
plt.show()


sidefig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(24, 16))
sidefig.suptitle('Solutions to ODE under the SEC approximated system')

ax1.plot(sol_sec.t, sol_sec.y.T, color='blue')
ax1.set_title('x- & y-coordinates prediction w.r.t. time t')

ax2.plot(sol_sec.t, sol_sec.y.T[:, 0], color='blue')
ax2.set_title('x-coordinates prediction w.r.t. time t')

ax3.plot(sol_sec.t, sol_sec.y.T[:, 1], color='blue')
ax3.set_title('y-coordinates prediction w.r.t. time t')

plt.show()


sidefig, (ax1, ax2) = plt.subplots(2, figsize=(48, 12))
sidefig.suptitle('Comparisons for solutions to ODE under the true and SEC approximated systems')

ax1.plot(sol_true.t, sol_true.y.T[:, 0], color='red')
ax1.plot(sol_sec.t, sol_sec.y.T[:, 0], color='blue')
ax1.set_title('x-coordinates prediction w.r.t. time t (true = red, SEC = blue)')

ax2.plot(sol_true.t, sol_true.y.T[:, 1], color='red')
ax2.plot(sol_sec.t, sol_sec.y.T[:, 1], color='blue')
ax2.set_title('y-coordinates prediction w.r.t. time t (true = red, SEC = blue)')

plt.show()
# %%
