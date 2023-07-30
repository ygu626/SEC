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
from matplotlib import cm
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



# %%
"""
SEC approximation
for pushforward of vector fields on the 2-torus embedded in R3
"""


# Fourier coefficients F_ak pf F w.r.t. difusion maps approximated eigenvectors Phi_j
# using pushforward into the square and R3
F1 = lambda theta, rho: np.array([(1/2)*theta**2, (1/2)*rho**2])
F2 = lambda theta, rho: np.array([(a + b*np.cos(theta))*np.cos(rho), (a + b*np.cos(theta))*np.sin(rho), a + b*np.sin(theta)])

v1F1 = lambda theta, rho: np.array([theta, rho])
v1F2 = lambda theta, rho: np.array([-b*np.sin(theta)*np.cos(rho) - (a + b*np.cos(theta))*np.sin(rho), -b*np.sin(theta)*np.sin(rho) + (a + b*np.cos(theta))*np.cos(rho), b*np.cos(theta)])


F1_ak = (1/(N**2))*np.matmul(F1(training_angle[0, :], training_angle[1, :]), Phis_normalized)
F2_ak = (1/(N**2))*np.matmul(F2(training_angle[0, :], training_angle[1, :]), Phis_normalized)
# %%


# %%
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
g_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for p in range(0, 2*I+1):
                                    g_coeff[i,j,p] = (lambs[i] + lambs[j] - lambs[p])/2

g = np.multiply(g_coeff, c)


# g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
# for i in range(0, 2*I+1):
#             for j in range(0, 2*I+1):
#                         for p in range(0, 2*I+1):
#                                     g[i,j,p] = (lambs[i] + lambs[j] - lambs[p])*c[i,j,p]/2
         
# print(g[:,:2,:2])
# %%



# %%
# Compute G_ijpq entries for the Gram operator and its dual
# using Monte Carlo integration
G = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G = np.einsum('ipm, jqm -> ijpq', c, g, dtype = float)

G = G[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G = np.reshape(G, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))

# print(G[:2,:2])


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
# %%



# %%
# Teuncate singular values of G based based on a small percentage of the largest singular valuecof G
threshold = 0.01/(np.max(s2))      # Threshold value for truncated SVD

# Compute duall Gram operator G* using pseudoinverse based on truncated singular values of G
# G_dual = np.linalg.pinv(G)

G_dual = np.linalg.pinv(G, rcond = threshold)
# G_dual_mc = np.linalg.pinv(G_mc_weighted)


"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""


# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product(Phis, training_angle, N = 100):
    v_an = v1F2(training_angle[0, :], training_angle[1, :])
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
gamma_km = np.einsum('ak, am -> km', F2_ak, b_am, dtype = float)


g = g[:(2*K+1), :, :]

eta_qlm = np.einsum('qkl, km -> qlm', g, gamma_km, dtype = float)


c = c[:(2*J+1), :, :]


v_hat_prime = np.einsum('qlm, plm -> pq', eta_qlm, c, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime[:, q] = np.exp(-tau*lambs[q])*v_hat_prime[:, q]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime = np.reshape(v_hat_prime, ((2*J+1)*(2*K+1)))
# print(v_hat_prime[:3,:3])


# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat = np.matmul(G_dual, v_hat_prime)
v_hat = np.reshape(v_hat, (2*J+1, 2*K+1))
# %%



# %%
# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights

# g = g[:(2*K+1), :, :]

# Weighted g_ijp Riemannian metric coefficients
g_weighted = np.zeros([2*K+1, 2*I+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted[j, :, :] = np.exp(-tau*lambs[j])*g[j, :, :]


h_ajl = np.einsum('ak, jkl -> ajl', F2_ak, g_weighted, dtype = float)


# c = c[:(2*J+1), :, :]
d_jlm = np.einsum('ij, ilm -> jlm', v_hat, c, dtype = float)

p_am = np.einsum('ajl, jlm -> am', h_ajl, d_jlm, dtype = float)
# %%


# %%
W_theta_x1 = np.zeros(int(N**2), dtype = float)
W_theta_y1 = np.zeros(int(N**2), dtype = float)

W_theta_x2 = np.zeros(int(N**2), dtype = float)
W_theta_y2 = np.zeros(int(N**2), dtype = float)
W_theta_z2 = np.zeros(int(N**2), dtype = float)

vector_approx1 = np.empty([int(N**2), 6], dtype = float)
vector_approx2 = np.empty([int(N**2), 7], dtype = float)


def W_theta1(varphi_xyzw):
    varphi_xyzw = np.real(varphi_xyzw)
    
    for i in range(0, int(N**2)):
        W_theta_x1[i] = np.sum(p_am[0, :]*varphi_xyzw[i, :])
        W_theta_y1[i] = np.sum(p_am[1, :]*varphi_xyzw[i, :])

    return W_theta_x1, W_theta_y1


def W_theta2(varphi_xyzw):
    varphi_xyzw = np.real(varphi_xyzw)
    
    for i in range(0, int(N**2)):
        W_theta_x2[i] = np.sum(p_am[0, :]*varphi_xyzw[i, :])
        W_theta_y2[i] = np.sum(p_am[1, :]*varphi_xyzw[i, :])
        W_theta_z2[i] = np.sum(p_am[2, :]*varphi_xyzw[i, :])

    return W_theta_x2, W_theta_y2, W_theta_z2


# W_x1, W_y1 = W_theta1(varphi_xyzw)
W_x2, W_y2, W_z2 = W_theta2(varphi_xyzw)


for i in range(0, int(N**2)):
    # vector_approx1[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], TRAIN_W[i], W_x1[i], W_y1[i]])
    vector_approx2[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], TRAIN_W[i], W_x2[i], W_y2[i], W_z2[i]])

print(vector_approx1[:3, :])
# %%


# %%
# pcolor plot for the 2D pushforward
random_num = 1000    # for 500 random indices
random_index = np.random.choice(vector_approx1.shape[0], random_num, replace = False)  

vector_approx_shuffled = vector_approx1[random_index]


a =  vector_approx_shuffled[:, 4]
b =  vector_approx_shuffled[:, 5]
a, b = np.meshgrid(a, b)

vector_grid = np.vstack([a.ravel(), b.ravel()])


fig = plt.figure(figsize=(12, 12))

plt.pcolor(vector_grid, edgecolors = 'k', linewidths = 0, cmap = 'summer')
plt.title('SEC Approximated Vector Field with 2D Pushforward')

plt.show()
# %%




# %%
# Plot for the 3D pushforward

def plot_torus(precision, a = 4, b = 1):
    U_t = np.linspace(0, 2*np.pi, precision)
    V_t = np.linspace(0, 2*np.pi, precision)
    
    U_t, V_t = np.meshgrid(U_t, V_t)
    
    X_t = (a + b*np.cos(U_t))*np.cos(V_t)
    Y_t = (a + b*np.cos(U_t))*np.sin(V_t)
    Z_t = b*np.sin(U_t)
    
    random_num = 200    # for 500 random indices
    random_index = np.random.choice(vector_approx2.shape[0], random_num, replace = False)  

    
    return X_t, Y_t, Z_t, random_index

    
x_t, y_t, z_t, rd_idx = plot_torus(100, 4, 1)

# ax = plt.axes(projection = '3d')

# ax.set_xlim(-5,5)
# ax.set_ylim(-5,5)
# ax.set_zlim(-5,5)

# ax.plot_surface(x_t, y_t, z_t, antialiased = True, color='orange')


vector_approx_shuffled = vector_approx2[rd_idx]


x2 = vector_approx_shuffled[:, 0]
y2 = vector_approx_shuffled[:, 1]
z2 = vector_approx_shuffled[:, 2]
w2 = vector_approx_shuffled[:, 3]
   
   
a2 = vector_approx2[:, 4]
b2 = vector_approx2[:, 5]
c2 = vector_approx2[:, 6]


# ax.quiver(x2, y2, z2, a2, b2, c2, length = 3, color = 'blue')


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
ax.scatter3D(a2, b2, c2, color = "blue")
plt.title("3D Scatter Plot of SEC Approximated Vector Fidls Coordinates")
 
plt.show()
# %%
