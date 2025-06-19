"""
Spectral Exterior Calculus (SEC)
2-torus T2 Example (Roration)
Approximations of vector fields on the 2-torus
usng donut embedding into R^3
Given pushforward of tangent vectors on the 2-torus
and determinstically sampled Monte Carlo points on the circle
"""


# %%
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
I = 140          # Inner index for eigenfunctions
J = 60           # Outer index for eigenfunctions
K = 10           # Index for gradients of eigenfunctions
n = 100          # Number of approximated tangent vectors
N = 150         # Number of Monte Carlo training data points 

# epsilon = 0.116    # RBF bandwidth parameter
# epsilon = 0.095498
epsilon = 0.0966
# epsilon = 0.09015711376058703

tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix
a = 5/3           # Radius of the latitude circle of the torus
b = 3/5           # Radius of the meridian circle of the torus
ALPHA = np.sqrt(20)



"""
Training data set
with pushforward of vector fields v on the torus
and smbedding map F with pushforward F_*v = vF
"""


# Deterministically sampled Monte Carlo training data points
# the latotude and meridian circles with radius a and b
def monte_carlo_points(start_pt = 0, end_pt = 2*np.pi, N = 150, a = 5/3, b = 3/5):
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



fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(-3,3)


# ax.plot_surface(x_t_ode, y_t_ode, z_t_ode, antialiased=True, alpha = 0.6, color='orange')

 
ax.scatter3D(training_data[0, :], training_data[1, :], training_data[2, :], color = "green")
plt.title("Solutions to ODE under the true system on the torus")
 
plt.show()



"""
Functions utilized in the following program
"""

# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta, rho: np.array([(a + b*np.cos(rho))*np.cos(theta), (a + b*np.cos(rho))*np.sin(theta), b*np.sin(rho)])
v1F = lambda theta, rho: np.array([-b*np.sin(rho)*np.cos(theta) - (a + b*np.cos(rho))*np.sin(theta), -b*np.sin(rho)*np.sin(theta) + (a + b*np.cos(rho))*np.cos(theta), b*np.cos(rho)])
v2F = lambda theta, rho: np.array([-b*ALPHA*np.sin(rho)*np.cos(theta) - (a + b*np.cos(rho))*np.sin(theta), -b*ALPHA*np.sin(rho)*np.sin(theta) + (a + b*np.cos(rho))*np.cos(theta), b*ALPHA*np.cos(rho)])

v32 = lambda theta, rho: ALPHA*(1 - np.cos(theta - rho))
v31 = lambda theta, rho: v32(theta, rho) + (1 - ALPHA)*(1 - np.cos(rho))
v3F = lambda theta, rho: np.array([-v32(theta, rho)*b*np.sin(rho)*np.cos(theta) - v31(theta, rho)*(a + b*np.cos(rho))*np.sin(theta), 
                                   -v32(theta, rho)*b*np.sin(rho)*np.sin(theta) + v31(theta, rho)*(a + b*np.cos(rho))*np.cos(theta), 
                                   v32(theta, rho)*b*np.cos(rho)])



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


# Normalization function q corresponding to diagonal matrix Q
def make_normalization_func(k, x_train):
    def normalized(x):
        y = np.sum(k(x, x_train), axis = 1)
        return y
    return normalized

# Heat kernel function k
k = lambda x_1, x_2: np.exp(-dist_matrix(x_1, x_2)/(epsilon**2))


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
p_func = make_p(k_hat, d)
P = p_func(training_data, training_data)
# print(P[:3,:3])

print(np.trace(P))
print(np.pi/(4*epsilon**2))
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
# eigenvalues, eigenvectors = eigs(P, k = 300) 
eigenvalues, eigenvectors = eig(S)
index = eigenvalues.argsort()[::-1][:2*I+1]
Lambs = eigenvalues[index]
Phis = np.real(eigenvectors[:, index])


# Compute approximated 0-Laplacian eigengunctions
lambs = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
            # lambs[i] = (4)*(-np.log(np.real(Lambs[i]))/(epsilon**2))
            lambs[i] = 4*(1 - np.real(Lambs[i]))/(epsilon**2)   

print(Lambs) 
       


# Normalize eigenfunctions Phi_j
Phis_normalized = np.empty([N**2, 2*I+1], dtype = float)
D_sqrt = np.power(D, (1/2))
for j in range(0, 2*I+1):
    Phis_normalized[:, j] = np.divide(np.real(Phis[:, j]), D_sqrt)

Phis_normalized = Phis_normalized/Phis_normalized[0, 0]
Phis_normalized = Phis_normalized/(2*np.pi)



print(Phis_normalized[:, 0])
print(np.max(Phis_normalized[:, 0]))
print(np.min(Phis_normalized[:, 0]))



D_bar = np.sum(D)
w = np.empty(N**2, dtype = float)
for i in range(0, N**2):
    w[i] = (4*np.pi**2)*D[i]/D_bar



print(np.dot(Phis_normalized[:, 128], Phis_normalized[:, 128]*w))



# Appeoximate eigenvalues and eigenfunctions for the 0-Laplacian
def make_varphi(k, x_train, lambs, phis):
    phi_lamb = np.real(phis / lambs)
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    return varphi

# Produce continuous extentions varphi_j for the eigenfunctions Phi_j
varphi = make_varphi(p_func, training_data, Lambs, Phis_normalized)
# varphi_flat = make_varphi(p, training_data_flat, Lambs, Phis_normalized)


# Apply the coninuous extensiom varphi to the training data set
varphi_xyzw = varphi(training_data)
# varphi_xyzw = varphi_flat(training_data_flat)

# print(varphi_xyz[:,3])
# %%



# Slice of the heat map
# for specific theta (latitude circle angle) values
y_true = np.reshape(Phis_normalized[:, 157], (150, 150))
y_test = np.reshape(varphi_xyzw[:, 157], (150, 150))

print(np.amax(y_test - y_true))
print(np.amin(y_test - y_true))


plt.scatter(u_a, y_test[1, :], color = 'blue')
plt.scatter(u_a, y_true[1, :], color = 'red')

plt.show 



"""
SEC approximation
for pushforward of vector fields on the 2-torus embedded in R3
"""

# %%
Phis_new = np.empty([N**2, 2*I+1], dtype = float)
for i in range(0, 2*I+1):
    Phis_new[:, i] = np.multiply(Phis_normalized[:, i], w)

# Fourier coefficients F_ak pf F w.r.t. difusion maps approximated eigenvectors Phi_j
F_ak = np.matmul(F(training_angle[0, :], training_angle[1, :]), Phis_new)

# print(F_ak[:, 2])



# Compute c_ijp coefficients
# using Monte Carlo integration
pool = mp.Pool()

def c_func(i, j, p):
    return np.sum(Phis_normalized[:, i]*Phis_normalized[:, j]*Phis_normalized[:, p]*w)

c = pool.starmap(c_func, 
              [(i, j, p) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for p in range(0, 2 * I + 1)])
            
c = np.reshape(np.array(c), (2 * I + 1, 2 * I + 1, 2 * I + 1))


print(c[4, 4, 0])


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
G = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
# G = np.einsum('mip, mjq -> ijpq', c, g, dtype = float, order = 'F', optimize = True)
# G = np.einsum('ipm, mjq -> ijpq', c, g, dtype = float)

for i in range(2*I+1):
    for j in range(2*I+1):
        G[i, j, :, :] = np.einsum('mp,  mq -> pq', c[:, i, :], g[:, j, :], dtype = float, order = 'F', optimize = True)

G = G[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G = np.reshape(G, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))
# %%

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
# Teuncate singular values of G based based on a small percentage of the largest singular valuecof G
threshold = 0.05    # Threshold value for truncated SVD

# Compute duall Gram operator G* using pseudoinverse based on truncated singular values of G
# G_dual = np.linalg.pinv(G)

G_dual = np.linalg.pinv(G, rcond = threshold)
# %%





"""
v1: Rational rotation on the torus
"""
# %%
"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""

# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product1(Phis, training_angle, N = 150):
    v_an = v1F(training_angle[0, :], training_angle[1, :])
    integral = np.sum(Phis*v_an*w, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func1(m):
    return monte_carlo_product1(Phis_normalized[:, m], training_angle)


b_am1 = pool.map(b_func1, 
                [m for m in range(0, 2 * I + 1)])

b_am1 = np.array(b_am1).T


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km1 = np.einsum('ak, am -> km', F_ak, b_am1, dtype = float)


# g = g[:(2*K+1), :, :]
g1 = g[:, :(2*K+1), :]

# eta_qlm = np.einsum('kql, km -> qlm', g, gamma_km, dtype = float)
eta_qlm1 = np.einsum('lqk, km -> qlm', g1, gamma_km1, dtype = float)


c1 = c[:(2*J+1), :, :]


v_hat_prime1 = np.einsum('qlm, plm -> pq', eta_qlm1, c1, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime1[:, q] = np.exp(-tau*lambs[q])*v_hat_prime1[:, q]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime1 = np.reshape(v_hat_prime1, ((2*J+1)*(2*K+1)))



# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat1 = np.matmul(G_dual, v_hat_prime1)
v_hat1 = np.reshape(v_hat1, (2*J+1, 2*K+1))



# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights

g_weighted1 = np.zeros([2*I+1, 2*K+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted1[:, j, :] = np.exp(-tau*lambs[j])*g1[:, j, :]


h_ajl1 = np.einsum('ak, ljk -> ajl', F_ak, g_weighted1, dtype = float)

d_jlm1 = np.einsum('ij, ilm -> jlm', v_hat1, c1, dtype = float)

p_am1 = np.einsum('ajl, jlm -> am', h_ajl1, d_jlm1, dtype = float)


W_theta_x1 = np.zeros(int(N**2), dtype = float)
W_theta_y1 = np.zeros(int(N**2), dtype = float)
W_theta_z1 = np.zeros(int(N**2), dtype = float)

def W_theta1(varphi_xyzw):
    varphi_xyzw = np.real(varphi_xyzw)
    
    for i in range(0, int(N**2)):
        W_theta_x1[i] = np.sum(p_am1[0, :]*varphi_xyzw[i, :])
        W_theta_y1[i] = np.sum(p_am1[1, :]*varphi_xyzw[i, :])
        W_theta_z1[i] = np.sum(p_am1[2, :]*varphi_xyzw[i, :])

    return W_theta_x1, W_theta_y1, W_theta_z1

# print(varphi_xyzw[:, 3])

W_x1, W_y1, W_z1 = W_theta1(varphi_xyzw)


vector_approx1 = np.empty([int(N**2), 6], dtype = float)
for i in range(0, int(N**2)):
    vector_approx1[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], W_x1[i], W_y1[i], W_z1[i]])


# Analytical tangent vector coordinates
ana_dir_coords1 = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, v1F(training_angle[0, :], training_angle[1, :])])


# Compute coefficient of determination R^2
# using the analytic and SEC approximated directional tangent vector coordinates
vec_ana1 = ana_dir_coords1[3:6, :] - ana_dir_coords1[:3, :]
vec_sec1 = vector_approx1[:, 3:6].T - vector_approx1[:, :3].T

ana_norm1 = np.sqrt(np.sum(np.power(vec_ana1, 2), axis = 0))
sec_norm1 = np.sqrt(np.sum(np.power(vec_sec1, 2), axis = 0))

norm_ratio1 = ana_norm1/sec_norm1


print(norm_ratio1)
print(np.amin(norm_ratio1))
print(np.amax(norm_ratio1))


rss1 = np.sum(np.power((vec_ana1 - vec_sec1), 2))

vec_bar1 = np.mean(vec_ana1, axis = 1)
tss1 = np.sum(np.power(vec_ana1, 2))

R_squared1 = 1 - rss1/tss1

print(rss1)
print(tss1)
print(R_squared1)



"""
Forward evolution/prediction of
true and SEC approximated dynamical systems
"""

# ODE solver applied to the SEC approximated vector fields
# with initial condition specified
# and the true system


"""
True system given by analytical solutions
"""

# Define time spans and initial values for the true system
tspan1 = np.linspace(0, 20, num=2000)

theta_01 = 0
rho_01 = 0

# Anakytical solutions to dydt = v|y = (1, 1) where v = (ddtheta, ddrho) and y = (theta, rho)
theta_t1 = lambda t: theta_01 + t
rho_t1 = lambda t: rho_01 + t


sol_true_theta1 = theta_t1(tspan1)

sol_true_rho1 = rho_t1(tspan1)


SOL_TRUE_X1 = X_func(sol_true_theta1, sol_true_rho1)
SOL_TRUE_Y1 = Y_func(sol_true_theta1, sol_true_rho1)
SOL_TRUE_Z1 = Z_func(sol_true_rho1)



"""
SEC Approximated System
"""
# Define derivative function for the SEC approximated system

def f_sec1(t, y):
    varphi_xyz = np.real(varphi(np.reshape(np.array(y), (3, 1))))
    W_x = np.sum(p_am1[0, :]*varphi_xyz)
    W_y = np.sum(p_am1[1, :]*varphi_xyz)
    W_z = np.sum(p_am1[2, :]*varphi_xyz)
    
    dydt = [W_x, W_y, W_z]
    return dydt


# Define initial value for the SEC approximated system
yinit1 = [a+b, 0, 0]

# Solve ODE under the SEC approximated system
sol_sec1 = solve_ivp(f_sec1, [tspan1[0], tspan1[-1]], yinit1, t_eval=tspan1, atol = 1e-8, rtol = 1e-8)


SOL_SEC_X1 = sol_sec1.y[0, :]
SOL_SEC_Y1 = sol_sec1.y[1, :]
SOL_SEC_Z1 = sol_sec1.y[2, :]

# %%




"""
v2: Irrational rotation on the torus
"""
# %%
"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""

# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product2(Phis, training_angle, N = 150):
    v_an = v2F(training_angle[0, :], training_angle[1, :])
    integral = np.sum(Phis*v_an*w, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func2(m):
    return monte_carlo_product2(Phis_normalized[:, m], training_angle)


b_am2 = pool.map(b_func2, 
                [m for m in range(0, 2 * I + 1)])

b_am2 = np.array(b_am2).T


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km2 = np.einsum('ak, am -> km', F_ak, b_am2, dtype = float)


# g = g[:(2*K+1), :, :]
g2 = g[:, :(2*K+1), :]

# eta_qlm = np.einsum('kql, km -> qlm', g, gamma_km, dtype = float)
eta_qlm2 = np.einsum('lqk, km -> qlm', g2, gamma_km2, dtype = float)


c2 = c[:(2*J+1), :, :]


v_hat_prime2 = np.einsum('qlm, plm -> pq', eta_qlm2, c2, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime2[:, q] = np.exp(-tau*lambs[q])*v_hat_prime2[:, q]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime2 = np.reshape(v_hat_prime2, ((2*J+1)*(2*K+1)))


# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat2 = np.matmul(G_dual, v_hat_prime2)
v_hat2 = np.reshape(v_hat2, (2*J+1, 2*K+1))


# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights

g_weighted2 = np.zeros([2*I+1, 2*K+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted2[:, j, :] = np.exp(-tau*lambs[j])*g2[:, j, :]


h_ajl2 = np.einsum('ak, ljk -> ajl', F_ak, g_weighted2, dtype = float)

d_jlm2 = np.einsum('ij, ilm -> jlm', v_hat2, c2, dtype = float)

p_am2 = np.einsum('ajl, jlm -> am', h_ajl2, d_jlm2, dtype = float)


W_theta_x2 = np.zeros(int(N**2), dtype = float)
W_theta_y2 = np.zeros(int(N**2), dtype = float)
W_theta_z2 = np.zeros(int(N**2), dtype = float)

def W_theta2(varphi_xyzw):
    varphi_xyzw = np.real(varphi_xyzw)
    
    for i in range(0, int(N**2)):
        W_theta_x2[i] = np.sum(p_am2[0, :]*varphi_xyzw[i, :])
        W_theta_y2[i] = np.sum(p_am2[1, :]*varphi_xyzw[i, :])
        W_theta_z2[i] = np.sum(p_am2[2, :]*varphi_xyzw[i, :])

    return W_theta_x2, W_theta_y2, W_theta_z2

# print(varphi_xyzw[:, 3])

W_x2, W_y2, W_z2 = W_theta2(varphi_xyzw)


vector_approx2 = np.empty([int(N**2), 6], dtype = float)
for i in range(0, int(N**2)):
    vector_approx2[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], W_x2[i], W_y2[i], W_z2[i]])


# Analytical tangent vector coordinates
ana_dir_coords2 = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, v2F(training_angle[0, :], training_angle[1, :])])


# Compute coefficient of determination R^2
# using the analytic and SEC approximated directional tangent vector coordinates
vec_ana2 = ana_dir_coords2[3:6, :] - ana_dir_coords2[:3, :]
vec_sec2 = vector_approx2[:, 3:6].T - vector_approx2[:, :3].T

ana_norm2 = np.sqrt(np.sum(np.power(vec_ana2, 2), axis = 0))
sec_norm2 = np.sqrt(np.sum(np.power(vec_sec2, 2), axis = 0))

norm_ratio2 = ana_norm2/sec_norm2


print(norm_ratio2)
print(np.amin(norm_ratio2))
print(np.amax(norm_ratio2))


rss2 = np.sum(np.power((vec_ana2 - vec_sec2), 2))

vec_bar2 = np.mean(vec_ana2, axis = 1)
tss2 = np.sum(np.power(vec_ana2, 2))

R_squared2 = 1 - rss2/tss2

print(rss2)
print(tss2)
print(R_squared2)



"""
Forward evolution/prediction of
true and SEC approximated dynamical systems
"""

# ODE solver applied to the SEC approximated vector fields
# with initial condition specified
# and the true system


"""
True system given by analytical solutions
"""

# Define time spans and initial values for the true system
tspan2 = np.linspace(0, 20, num=2000)

theta_02 = 0
rho_02 = 0

# Anakytical solutions to dydt = v|y = (1, 1) where v = (ddtheta, ddrho) and y = (theta, rho)
theta_t2 = lambda t: theta_01 + t
rho_t2 = lambda t: rho_01 + ALPHA*t


sol_true_theta2 = theta_t2(tspan2)

sol_true_rho2 = rho_t2(tspan2)


SOL_TRUE_X2 = X_func(sol_true_theta2, sol_true_rho2)
SOL_TRUE_Y2 = Y_func(sol_true_theta2, sol_true_rho2)
SOL_TRUE_Z2 = Z_func(sol_true_rho2)



"""
SEC Approximated System
"""
# Define derivative function for the SEC approximated system

def f_sec2(t, y):
    varphi_xyz = np.real(varphi(np.reshape(np.array(y), (3, 1))))
    W_x = np.sum(p_am2[0, :]*varphi_xyz)
    W_y = np.sum(p_am2[1, :]*varphi_xyz)
    W_z = np.sum(p_am2[2, :]*varphi_xyz)
    
    dydt = [W_x, W_y, W_z]
    return dydt


# Define initial value for the SEC approximated system
yinit2 = [a+b, 0, 0]

# Solve ODE under the SEC approximated system
sol_sec2 = solve_ivp(f_sec2, [tspan2[0], tspan2[-1]], yinit2, t_eval=tspan2, atol = 1e-8, rtol = 1e-8)


SOL_SEC_X2 = sol_sec2.y[0, :]
SOL_SEC_Y2 = sol_sec2.y[1, :]
SOL_SEC_Z2 = sol_sec2.y[2, :]

# %%




"""
v3: Stepanoff flow on the torus
"""
# %%
"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""

# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product3(Phis, training_angle, N = 150):
    v_an = v3F(training_angle[0, :], training_angle[1, :])
    integral = np.sum(Phis*v_an*w, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func3(m):
    return monte_carlo_product3(Phis_normalized[:, m], training_angle)


b_am3 = pool.map(b_func3, 
                [m for m in range(0, 2 * I + 1)])

b_am3 = np.array(b_am3).T


# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km3 = np.einsum('ak, am -> km', F_ak, b_am3, dtype = float)


# g = g[:(2*K+1), :, :]
g3 = g[:, :(2*K+1), :]

# eta_qlm = np.einsum('kql, km -> qlm', g, gamma_km, dtype = float)
eta_qlm3 = np.einsum('lqk, km -> qlm', g3, gamma_km3, dtype = float)


c3 = c[:(2*J+1), :, :]


v_hat_prime3 = np.einsum('qlm, plm -> pq', eta_qlm3, c3, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime3[:, q] = np.exp(-tau*lambs[q])*v_hat_prime3[:, q]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime3 = np.reshape(v_hat_prime3, ((2*J+1)*(2*K+1)))


# Apply dual Gram operator G* to obtain v_hat 
# using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat3 = np.matmul(G_dual, v_hat_prime3)
v_hat3 = np.reshape(v_hat3, (2*J+1, 2*K+1))


# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# using Monte Carlo integration with weights

g_weighted3 = np.zeros([2*I+1, 2*K+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_weighted3[:, j, :] = np.exp(-tau*lambs[j])*g3[:, j, :]


h_ajl3 = np.einsum('ak, ljk -> ajl', F_ak, g_weighted3, dtype = float)

d_jlm3 = np.einsum('ij, ilm -> jlm', v_hat3, c3, dtype = float)

p_am3 = np.einsum('ajl, jlm -> am', h_ajl3, d_jlm3, dtype = float)


W_theta_x3 = np.zeros(int(N**2), dtype = float)
W_theta_y3 = np.zeros(int(N**2), dtype = float)
W_theta_z3 = np.zeros(int(N**2), dtype = float)

def W_theta3(varphi_xyzw):
    varphi_xyzw = np.real(varphi_xyzw)
    
    for i in range(0, int(N**2)):
        W_theta_x3[i] = np.sum(p_am3[0, :]*varphi_xyzw[i, :])
        W_theta_y3[i] = np.sum(p_am3[1, :]*varphi_xyzw[i, :])
        W_theta_z3[i] = np.sum(p_am3[2, :]*varphi_xyzw[i, :])

    return W_theta_x3, W_theta_y3, W_theta_z3

# print(varphi_xyzw[:, 3])

W_x3, W_y3, W_z3 = W_theta3(varphi_xyzw)


vector_approx3 = np.empty([int(N**2), 6], dtype = float)
for i in range(0, int(N**2)):
    vector_approx3[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], W_x3[i], W_y3[i], W_z3[i]])


# Analytical tangent vector coordinates
ana_dir_coords3 = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z, v3F(training_angle[0, :], training_angle[1, :])])


# Compute coefficient of determination R^2
# using the analytic and SEC approximated directional tangent vector coordinates
vec_ana3 = ana_dir_coords3[3:6, :] - ana_dir_coords3[:3, :]
vec_sec3 = vector_approx3[:, 3:6].T - vector_approx3[:, :3].T

ana_norm3 = np.sqrt(np.sum(np.power(vec_ana3, 2), axis = 0))
sec_norm3 = np.sqrt(np.sum(np.power(vec_sec3, 2), axis = 0))

norm_ratio3 = ana_norm3/sec_norm3


print(norm_ratio3)
print(np.amin(norm_ratio3))
print(np.amax(norm_ratio3))


rss3 = np.sum(np.power((vec_ana3 - vec_sec3), 2))

vec_bar3 = np.mean(vec_ana3, axis = 1)
tss3 = np.sum(np.power(vec_ana3, 2))

R_squared3 = 1 - rss3/tss3

print(rss3)
print(tss3)
print(R_squared3)



"""
Forward evolution/prediction of
true and SEC approximated dynamical systems
"""

# ODE solver applied to the SEC approximated vector fields
# with initial condition specified
# and the true system



"""
True system given by analytical solutions
"""

# Define time spans and initial values for the true system
tspan3 = np.linspace(0, 10, num=2000)

# Define derivative function for the true system under the Stepanoff flow
def f_true3(t, y):
    theta, rho = y
    
    dydt = [v31(theta, rho), v32(theta, rho)]
    return dydt


# Define time spans and initial values for the SEC approximated system
theta_031 = np.pi+0.3
rho_031 = np.pi+0.5
yinit31 = [theta_031, rho_031]


# Solve ODE under the SEC approximated system
sol_true3 = solve_ivp(f_true3, [tspan3[0], tspan3[-1]], yinit31, t_eval = tspan3, atol = 1e-8, rtol = 1e-8)

SOL_TRUE_X3 = X_func(sol_true3.y[0, :], sol_true3.y[1, :])
SOL_TRUE_Y3 = Y_func(sol_true3.y[0, :], sol_true3.y[1, :])
SOL_TRUE_Z3 = Z_func(sol_true3.y[1, :])



"""
SEC approximated system
"""

# Define derivative function for the SEC approximated system
def f_sec3(t, y):
    varphi_xyz = np.real(varphi(np.reshape(np.array(y), (3, 1))))
    W_x = np.sum(p_am3[0, :]*varphi_xyz)
    W_y = np.sum(p_am3[1, :]*varphi_xyz)
    W_z = np.sum(p_am3[2, :]*varphi_xyz)
    
    dydt = [W_x, W_y, W_z]
    return dydt


theta_032 = np.pi+0.3
rho_032 = np.pi+0.5
yinit32 = F(theta_032, rho_032)



# Solve ODE under the SEC approximated system
sol_sec3 = solve_ivp(f_sec3, [tspan3[0], tspan3[-1]], yinit32, t_eval=tspan3, atol = 1e-8, rtol = 1e-8)


SOL_SEC_X3 = sol_sec3.y[0, :]
SOL_SEC_Y3 = sol_sec3.y[1, :]
SOL_SEC_Z3 = sol_sec3.y[2, :]

# %%



# %%
def plot_torus(precision, a = 5/3, b = 3/5):
    U_t = np.linspace(0, 2*np.pi, precision)
    V_t = np.linspace(0, 2*np.pi, precision)
    
    U_t, V_t = np.meshgrid(U_t, V_t)
    
    X_t = (a + b*np.cos(U_t))*np.cos(V_t)
    Y_t = (a + b*np.cos(U_t))*np.sin(V_t)
    Z_t = b*np.sin(U_t)
    
    random_num = 100    # for 500 random indices
    random_index = np.random.choice(vector_approx1.shape[0], random_num, replace = False)  

    
    return X_t, Y_t, Z_t, random_index

    

x_t, y_t, z_t, rd_idx = plot_torus(500, 5/3, 3/5)
# %%





"""
Tangent vector plots on the torus
"""
# %%
vector_ana_shuffled1 = (ana_dir_coords1.T)[rd_idx]
vector_ana_shuffled2 = (ana_dir_coords2.T)[rd_idx]
vector_ana_shuffled3 = (ana_dir_coords3.T)[rd_idx]

vector_approx_shuffled1 = vector_approx1[rd_idx]
vector_approx_shuffled2 = vector_approx2[rd_idx]
vector_approx_shuffled3 = vector_approx3[rd_idx]


# Tangent vectors given by v1
x21 = vector_approx_shuffled1[:, 0]
y21 = vector_approx_shuffled1[:, 1]
z21 = vector_approx_shuffled1[:, 2]
   
a21 = vector_approx_shuffled1[:, 3]
b21 = vector_approx_shuffled1[:, 4]
c21 = vector_approx_shuffled1[:, 5]

x31 = vector_ana_shuffled1[:, 0]
y31 = vector_ana_shuffled1[:, 1]
z31 = vector_ana_shuffled1[:, 2]
   
a31 = vector_ana_shuffled1[:, 3]
b31 = vector_ana_shuffled1[:, 4]
c31 = vector_ana_shuffled1[:, 5]
    
       
# Tangent vectors given by v2
x22 = vector_approx_shuffled2[:, 0]
y22 = vector_approx_shuffled2[:, 1]
z22 = vector_approx_shuffled2[:, 2]
   
a22 = vector_approx_shuffled2[:, 3]
b22 = vector_approx_shuffled2[:, 4]
c22 = vector_approx_shuffled2[:, 5]

x32 = vector_ana_shuffled2[:, 0]
y32 = vector_ana_shuffled2[:, 1]
z32 = vector_ana_shuffled2[:, 2]
   
a32 = vector_ana_shuffled2[:, 3]
b32 = vector_ana_shuffled2[:, 4]
c32 = vector_ana_shuffled2[:, 5]
  
  
# Tangent vectors given by v3
x23 = vector_approx_shuffled3[:, 0]
y23 = vector_approx_shuffled3[:, 1]
z23 = vector_approx_shuffled3[:, 2]
   
a23 = vector_approx_shuffled3[:, 3]
b23 = vector_approx_shuffled3[:, 4]
c23 = vector_approx_shuffled3[:, 5]

x33 = vector_ana_shuffled3[:, 0]
y33 = vector_ana_shuffled3[:, 1]
z33 = vector_ana_shuffled3[:, 2]
   
a33 = vector_ana_shuffled3[:, 3]
b33 = vector_ana_shuffled3[:, 4]
c33 = vector_ana_shuffled3[:, 5]
# %% 


# %%
fig = plt.figure(figsize = (16, 12), layout = "constrained")
# fig.suptitle(r"Tangent Vectors on $T^2$", fontsize = 36)


ax1 = fig.add_subplot(2, 3, 1, projection='3d')

ax1.set_title(r"(a) $V_1$", fontsize = 26)
ax1.set_xlim(-3,3)
ax1.set_ylim(-3,3)
ax1.set_zlim(-3,3)

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_ylabel("y", fontsize = 20)
ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_zlabel("z", fontsize = 20)
ax1.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax1.quiver(x31, y31, z31, a31, b31, c31, color = 'red')


ax2 = fig.add_subplot(2, 3, 2, projection='3d')

ax2.set_title(r"(b) $V_2$", fontsize = 26)
ax2.set_xlim(-3,3)
ax2.set_ylim(-3,3)
ax2.set_zlim(-3,3)

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_ylabel("y", fontsize = 20)
ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_zlabel("z", fontsize = 20)
ax2.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax2.quiver(x32, y32, z32, a32, b32, c32, length = 0.5, color = 'red')


ax3 = fig.add_subplot(2, 3, 3, projection='3d')

ax3.set_title(r"(c) $V_3$", fontsize = 32)
ax3.set_xlim(-3,3)
ax3.set_ylim(-3,3)
ax3.set_zlim(-3,3)

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_ylabel("y", fontsize = 20)
ax3.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_zlabel("z", fontsize = 20)
ax3.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax3.quiver(x33, y33, z33, a33, b33, c33, length = 0.2, color = 'red')


ax4 = fig.add_subplot(2, 3, 4, projection='3d')

ax4.set_title(r"(d) $V_1^{(L)}$", fontsize = 26)
ax4.set_xlim(-3,3)
ax4.set_ylim(-3,3)
ax4.set_zlim(-3,3)

ax4.set_xlabel("x", fontsize = 20)
ax4.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_ylabel("y", fontsize = 20)
ax4.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_zlabel("z", fontsize = 20)
ax4.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax4.quiver(x21, y21, z21, a21, b21, c21, color = 'blue')


ax5 = fig.add_subplot(2, 3, 5, projection='3d')

ax5.set_title(r"(e) $V_2^{(L)}$", fontsize = 26)
ax5.set_xlim(-3,3)
ax5.set_ylim(-3,3)
ax5.set_zlim(-3,3)

ax5.set_xlabel("x", fontsize = 20)
ax5.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_ylabel("y", fontsize = 20)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_zlabel("z", fontsize = 20)
ax5.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax5.quiver(x22, y22, z22, a22, b22, c22, length = 0.5, color = 'blue')


ax6 = fig.add_subplot(2, 3, 6, projection='3d')

ax6.set_title(r"(f) $V_3^{(L)}$", fontsize = 26)
ax6.set_xlim(-3,3)
ax6.set_ylim(-3,3)
ax6.set_zlim(-3,3)

ax6.set_xlabel("x", fontsize = 20)
ax6.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_ylabel("y", fontsize = 20)
ax6.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_zlabel("z", fontsize = 20)
ax6.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax6.quiver(x23, y23, z23, a23, b23, c23, length = 0.2, color = 'blue')

  
plt.show()
# %%


"""
POSTER
Tangent vector plots on the torus
"""
# %%
fig = plt.figure(figsize = (18, 10), layout = "constrained")
# fig.suptitle(r"Tangent Vectors on $\mathbb{T}^2$ Given by $\vec{V}_{\mathrm{SEC}}:\mathbb{R}^3\to\mathbb{R}^3, \vec{V}_{\mathrm{SEC}} = F_* \tilde{V}_{\mathrm{SEC}}$", fontsize = 36)
# fig.text(0.04, 0.75, 'True Vector Field', ha='center', va='center', fontsize = 26)
# fig.text(0.04, 0.32, 'SEC Approximation', ha='center', va='center', fontsize = 26)


ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title("Rational Rotation \n (a)", fontsize = 26)

ax1.set_xlim(-3,3)
ax1.set_ylim(-3,3)
ax1.set_zlim(-3,3)

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_ylabel("y", fontsize = 20)
ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_zlabel("z", fontsize = 20)
ax1.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
q1 = ax1.quiver(x31, y31, z31, a31, b31, c31, color = 'red', label = 'True Vector Field')


ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.set_title("Irrational Rotation \n (c)", fontsize = 26)

ax2.set_xlim(-3,3)
ax2.set_ylim(-3,3)
ax2.set_zlim(-3,3)

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_ylabel("y", fontsize = 20)
ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_zlabel("z", fontsize = 20)
ax2.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax2.quiver(x32, y32, z32, a32, b32, c32, length = 0.5, color = 'red')


ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.set_title("Stepanoff Flow \n (e)", fontsize = 32)

ax3.set_xlim(-3,3)
ax3.set_ylim(-3,3)
ax3.set_zlim(-3,3)

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_ylabel("y", fontsize = 20)
ax3.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_zlabel("z", fontsize = 20)
ax3.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax3.quiver(x33, y33, z33, a33, b33, c33, length = 0.2, color = 'red')


ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.set_title("\n (b)", fontsize = 26)

ax4.set_xlim(-3,3)
ax4.set_ylim(-3,3)
ax4.set_zlim(-3,3)

ax4.set_xlabel("x", fontsize = 20)
ax4.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_ylabel("y", fontsize = 20)
ax4.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_zlabel("z", fontsize = 20)
ax4.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
q2 = ax4.quiver(x21, y21, z21, a21, b21, c21, color = 'blue', label = 'SEC Approximation')


ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.set_title("\n (d)", fontsize = 26)

ax5.set_xlim(-3,3)
ax5.set_ylim(-3,3)
ax5.set_zlim(-3,3)

ax5.set_xlabel("x", fontsize = 20)
ax5.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_ylabel("y", fontsize = 20)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_zlabel("z", fontsize = 20)
ax5.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax5.quiver(x22, y22, z22, a22, b22, c22, length = 0.5, color = 'blue')


ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.set_title("\n (f)", fontsize = 26)

ax6.set_xlim(-3,3)
ax6.set_ylim(-3,3)
ax6.set_zlim(-3,3)

ax6.set_xlabel("x", fontsize = 20)
ax6.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_ylabel("y", fontsize = 20)
ax6.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_zlabel("z", fontsize = 20)
ax6.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax6.quiver(x23, y23, z23, a23, b23, c23, length = 0.2, color = 'blue')


fig.legend(handles = [q1, q2], loc = 'lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), prop={'size': 26})
  
plt.show()
# %%


"""
Solutions to ODEs on the torus
"""
# %%
fig = plt.figure(figsize = (16, 12))
# fig.suptitle('Solutions of Dynamical Systems', fontsize = 36)


ax1 = fig.add_subplot(2, 3, 1, projection='3d')

ax1.set_title(r"(a) $V_1$", fontsize = 26)
ax1.set_xlim(-3,3)
ax1.set_ylim(-3,3)
ax1.set_zlim(-3,3)

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_ylabel("y", fontsize = 20)
ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_zlabel("z", fontsize = 20)
ax1.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax1.scatter3D(SOL_TRUE_X1, SOL_TRUE_Y1, SOL_TRUE_Z1, s = 1, color = "red")


ax2 = fig.add_subplot(2, 3, 2, projection='3d')

ax2.set_title(r"(b) $V_2$", fontsize = 26)
ax2.set_xlim(-3,3)
ax2.set_ylim(-3,3)
ax2.set_zlim(-3,3)

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_ylabel("y", fontsize = 20)
ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_zlabel("z", fontsize = 20)
ax2.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax2.scatter3D(SOL_TRUE_X2, SOL_TRUE_Y2, SOL_TRUE_Z2, s = 1, color = "red")



ax3 = fig.add_subplot(2, 3, 3, projection='3d')

ax3.set_title(r"(c) $V_3$", fontsize = 26)
ax3.set_xlim(-3,3)
ax3.set_ylim(-3,3)
ax3.set_zlim(-3,3)

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_ylabel("y", fontsize = 20)
ax3.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_zlabel("z", fontsize = 20)
ax3.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax3.scatter3D(SOL_TRUE_X3, SOL_TRUE_Y3, SOL_TRUE_Z3, s = 0.5, color = "red")



ax4 = fig.add_subplot(2, 3, 4, projection='3d')

ax4.set_title(r"(d) $V_1^{(L)}$", fontsize = 26)
ax4.set_xlim(-3,3)
ax4.set_ylim(-3,3)
ax4.set_zlim(-3,3)

ax4.set_xlabel("x", fontsize = 20)
ax4.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_ylabel("y", fontsize = 20)
ax4.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_zlabel("z", fontsize = 20)
ax4.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax4.scatter3D(SOL_SEC_X1, SOL_SEC_Y1, SOL_SEC_Z1, s = 1, color = "blue")



ax5 = fig.add_subplot(2, 3, 5, projection='3d')

ax5.set_title(r"(e) $V_2^{(L)}$", fontsize = 26)
ax5.set_xlim(-3,3)
ax5.set_ylim(-3,3)
ax5.set_zlim(-3,3)

ax5.set_xlabel("x", fontsize = 20)
ax5.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_ylabel("y", fontsize = 20)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_zlabel("z", fontsize = 20)
ax5.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax5.scatter3D(SOL_SEC_X2, SOL_SEC_Y2, SOL_SEC_Z2, s = 1, color = "blue")


ax6 = fig.add_subplot(2, 3, 6, projection='3d')

ax6.set_title(r"(f) $V_3^{(L)}$", fontsize = 26)
ax6.set_xlim(-3,3)
ax6.set_ylim(-3,3)
ax6.set_zlim(-3,3)

ax6.set_xlabel("x", fontsize = 20)
ax6.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_ylabel("y", fontsize = 20)
ax6.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_zlabel("z", fontsize = 20)
ax6.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax6.scatter3D(SOL_SEC_X3, SOL_SEC_Y3, SOL_SEC_Z3, s = 0.05, color = "blue")

  
plt.show()
# %%


"""
POSTER
Solutions to ODEs on the torus
"""
# %%
fig = plt.figure(figsize = (18, 10))
# fig.suptitle(r'Dynamical Evolution $(\dot{\theta}_1(t), \dot{\theta}_2(t)) = V\mid_{(\theta_1(t), \theta_2(t))}$', fontsize = 36)
# fig.text(0.04, 0.75, 'True Dynamical System', ha='center', va='center', fontsize = 26)
# fig.text(0.04, 0.32, 'SEC Approximation', ha='center', va='center', fontsize = 26)

ax1 = fig.add_subplot(2, 3, 1, projection='3d')

ax1.set_title("\n \n Rational Rotation \n (a)", fontsize = 26)
ax1.set_xlim(-3,3)
ax1.set_ylim(-3,3)
ax1.set_zlim(-3,3)

ax1.set_xlabel("x", fontsize = 20)
ax1.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_ylabel("y", fontsize = 20)
ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.set_zlabel("z", fontsize = 20)
ax1.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax1.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax1.scatter3D(SOL_TRUE_X1, SOL_TRUE_Y1, SOL_TRUE_Z1, s = 1, color = "red", label = 'True Dynamical System')


ax2 = fig.add_subplot(2, 3, 2, projection='3d')

ax2.set_title("\n\n Irrational Rotation \n (c)", fontsize = 26)
ax2.set_xlim(-3,3)
ax2.set_ylim(-3,3)
ax2.set_zlim(-3,3)

ax2.set_xlabel("x", fontsize = 20)
ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_ylabel("y", fontsize = 20)
ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.set_zlabel("z", fontsize = 20)
ax2.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax2.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax2.scatter3D(SOL_TRUE_X2, SOL_TRUE_Y2, SOL_TRUE_Z2, s = 1, color = "red")



ax3 = fig.add_subplot(2, 3, 3, projection='3d')

ax3.set_title("\n\n Stepanoff Flow \n (e)", fontsize = 26)
ax3.set_xlim(-3,3)
ax3.set_ylim(-3,3)
ax3.set_zlim(-3,3)

ax3.set_xlabel("x", fontsize = 20)
ax3.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_ylabel("y", fontsize = 20)
ax3.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.set_zlabel("z", fontsize = 20)
ax3.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax3.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax3.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax3.scatter3D(SOL_TRUE_X3, SOL_TRUE_Y3, SOL_TRUE_Z3, s = 0.5, color = "red")



ax4 = fig.add_subplot(2, 3, 4, projection='3d')

ax4.set_title("\n (b)", fontsize = 26)
ax4.set_xlim(-3,3)
ax4.set_ylim(-3,3)
ax4.set_zlim(-3,3)

ax4.set_xlabel("x", fontsize = 20)
ax4.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_ylabel("y", fontsize = 20)
ax4.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.set_zlabel("z", fontsize = 20)
ax4.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax4.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax4.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax4.scatter3D(SOL_SEC_X1, SOL_SEC_Y1, SOL_SEC_Z1, s = 1, color = "blue", label = 'SEC Approximation')



ax5 = fig.add_subplot(2, 3, 5, projection='3d')

ax5.set_title("\n (d)", fontsize = 26)
ax5.set_xlim(-3,3)
ax5.set_ylim(-3,3)
ax5.set_zlim(-3,3)

ax5.set_xlabel("x", fontsize = 20)
ax5.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_ylabel("y", fontsize = 20)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.set_zlabel("z", fontsize = 20)
ax5.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax5.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color='aliceblue')
ax5.scatter3D(SOL_SEC_X2, SOL_SEC_Y2, SOL_SEC_Z2, s = 1, color = "blue")


ax6 = fig.add_subplot(2, 3, 6, projection='3d')

ax6.set_title("\n (f)", fontsize = 26)
ax6.set_xlim(-3,3)
ax6.set_ylim(-3,3)
ax6.set_zlim(-3,3)

ax6.set_xlabel("x", fontsize = 20)
ax6.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_ylabel("y", fontsize = 20)
ax6.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.set_zlabel("z", fontsize = 20)
ax6.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax6.set_zticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 18)

ax6.plot_surface(x_t, y_t, z_t, antialiased=True, alpha = 0.5, color = 'aliceblue')
ax6.scatter3D(SOL_SEC_X3, SOL_SEC_Y3, SOL_SEC_Z3, s = 0.05, color = "blue")


from matplotlib.lines import Line2D

legend_elements = [Line2D([], [], marker = '_', linestyle='None', label='True Dynamical System',
                          color = 'red', markersize = 36),
                   Line2D([], [], marker = '_', linestyle = 'None', label = 'SEC Approximation',
                          color = 'blue', markersize = 36)]

fig.legend(handles = legend_elements, ncol = 2, loc = 'lower center', bbox_to_anchor=(0.5, -0.01), prop={'size': 26})
  
plt.show()
# %%



"""
Time series evolutions of ODEs on the torus
"""
# %%
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (30, 18), layout = "constrained")
# fig.suptitle('Time Series Evolution of Dynamical Syatem Solutions', fontsize = 60)
# fig.tight_layout(pad = 10.0)


ax1.set_title('x-coordinates \n (a)', fontsize = 48)
ax1.set_xlim([0,20])
ax1.set_ylim([-1.6, 3.1])

ax1.set_xlabel("t", fontsize = 32)
ax1.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax1.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax1.set_ylabel("x", rotation = 0, fontsize = 32)
ax1.set_yticks([-1.5, -1, -0.5, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
ax1.set_yticklabels([-1.5, -1, -0.5, -0.5, 0, 0.5, 1, 1.5, 2, 2.5], fontsize = 26)

ax1.plot(sol_sec1.t, SOL_TRUE_X1, color='red')
ax1.plot(sol_sec1.t, SOL_SEC_X1, color='blue')

ax1.legend([r"$V_1$", r"$V_1^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax2.set_title('y-coordinates \n (b)', fontsize = 48)
ax2.set_xlim([0,20])
ax2.set_ylim([-2.1, 2.5])

ax2.set_xlabel("t", fontsize = 32)
ax2.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax2.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax2.set_ylabel("y", rotation = 0, fontsize = 32)
ax2.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
ax2.set_yticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], fontsize = 26)

ax2.plot(sol_sec1.t, SOL_TRUE_Y1, color='red')
ax2.plot(sol_sec1.t, SOL_SEC_Y1, color='blue')

ax2.legend([r"$V_1$", r"$V_1^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax3.set_title('z-coordinates \n (c)', fontsize = 48)
ax3.set_xlim([0,20])
ax3.set_ylim([-0.85, 1.15])

ax3.set_xlabel("t", fontsize = 32)
ax3.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax3.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax3.set_ylabel("z", rotation = 0, fontsize = 32)
ax3.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax3.set_yticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize = 26)

ax3.plot(sol_sec1.t, SOL_TRUE_Z1, color='red')
ax3.plot(sol_sec1.t, SOL_SEC_Z1, color='blue')

ax3.legend([r"$V_1$", r"$V_1^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax4.set_title('(d)', fontsize = 48)
ax4.set_xlim([0,20])
ax4.set_ylim([-2.6, 3.1])

ax4.set_xlabel("t", fontsize = 32)
ax4.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax4.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax4.set_ylabel("x", rotation = 0, fontsize = 32)
ax4.set_yticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
ax4.set_yticklabels([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5], fontsize = 26)

ax4.plot(sol_sec2.t, SOL_TRUE_X2, color='red')
ax4.plot(sol_sec2.t, sol_sec2.y.T[:, 0], color='blue')

ax4.legend([r"$V_2$", r"$V_2^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax5.set_title('(e)', fontsize = 48)
ax5.set_xlim([0,20])
ax5.set_ylim([-3.1, 3.2])

ax5.set_xlabel("t", fontsize = 32)
ax5.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax5.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax5.set_ylabel("y", rotation = 0, fontsize = 32)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 26)

ax5.plot(sol_sec2.t, SOL_TRUE_Y2, color='red')
ax5.plot(sol_sec2.t, sol_sec2.y.T[:, 1], color='blue')

ax5.legend([r"$V_2$", r"$V_2^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax6.set_title('(f)', fontsize = 48)
ax6.set_xlim([0,20])
ax6.set_ylim([-0.85, 0.95])

ax6.set_xlabel("t", fontsize = 32)
ax6.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax6.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax6.set_ylabel("z", rotation = 0, fontsize = 32)
ax6.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax6.set_yticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize = 26)

ax6.plot(sol_sec2.t, SOL_TRUE_Z2, color='red')
ax6.plot(sol_sec2.t, sol_sec2.y.T[:, 2], color='blue')

ax6.legend([r"$V_2$", r"$V_2^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax7.set_title('(g)', fontsize = 48)
ax7.set_xlim([0,10])
ax7.set_ylim([-2.1, 2.1])

ax7.set_xlabel("t", fontsize = 32)
ax7.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax7.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize = 26)

ax7.set_ylabel("x", rotation = 0, fontsize = 32)
ax7.set_yticks([-2, -1, 0, 1, 2])
ax7.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 26)

ax7.plot(sol_sec3.t, SOL_TRUE_X3, color='red')
ax7.plot(sol_sec3.t, sol_sec3.y.T[:, 0], color='blue')

ax7.legend([r"$V_3$", r"$V_3^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax8.set_title('(h)', fontsize = 48)
ax8.set_xlim([0,10])
ax8.set_ylim([-3.1, 3.3])

ax8.set_xlabel("t", fontsize = 32)
ax8.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax8.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize = 30)

ax8.set_ylabel("y", rotation = 0, fontsize = 32)
ax8.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax8.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 30)

ax8.plot(sol_sec3.t, SOL_TRUE_Y3, color='red')
ax8.plot(sol_sec3.t, sol_sec3.y.T[:, 1], color='blue')

ax8.legend([r"$V_3$", r"$V_3^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



ax9.set_title('(i)', fontsize = 48)
ax9.set_xlim([0,10])
ax9.set_ylim([-0.85, 0.95])

ax9.set_xlabel("t", fontsize = 32)
ax9.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax9.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize = 26)

ax9.set_ylabel("z", rotation = 0, fontsize = 32)
ax9.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax9.set_yticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize = 26)

ax9.plot(sol_sec3.t, SOL_TRUE_Z3, color='red')
ax9.plot(sol_sec3.t, sol_sec3.y.T[:, 2], color='blue')

ax9.legend([r"$V_3$", r"$V_3^{(L)}$"], loc="upper right", frameon = True, edgecolor = "black")



plt.show()
# %%


"""
POSTER
Time series evolutions of ODEs on the torus
"""
# %%
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (30, 18), layout = "constrained")
# fig.suptitle(r'Dynamical Evolution $\dot{y}(t) = \vec{V}_{\mathrm{SEC}}\mid_{y(t)}, y(t) = F(\theta_1(t), \theta_2(t))$', fontsize = 60)
# fig.suptitle(r'Dynamical Evolution $(\dot{\theta}_1(t), \dot{\theta}_2(t)) = V\mid_{(\theta_1(t), \theta_2(t))}$', fontsize = 60)
# fig.tight_layout(pad = 10.0)


(topfig, midfig, bottomfig) = fig.subfigures(3, 1)
topfig.suptitle('Rational Rotation', fontsize = 48)
               
(ax1, ax2, ax3) = topfig.subplots(nrows = 1, ncols = 3)


ax1.set_title('(a)', fontsize = 64)
ax1.set_xlim([0,20])
ax1.set_ylim([-1.6, 3.1])

ax1.set_xlabel("t", fontsize = 32)
ax1.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax1.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax1.set_ylabel("x", rotation = 0, fontsize = 32)
ax1.set_yticks([-1.5, -1, -0.5, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
ax1.set_yticklabels([-1.5, -1, -0.5, -0.5, 0, 0.5, 1, 1.5, 2, 2.5], fontsize = 26)

ax1.plot(sol_sec1.t, SOL_TRUE_X1, color='red')
ax1.plot(sol_sec1.t, SOL_SEC_X1, color='blue')

ax1.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax2.set_title('(b)', fontsize = 48)
ax2.set_xlim([0,20])
ax2.set_ylim([-2.1, 2.5])

ax2.set_xlabel("t", fontsize = 32)
ax2.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax2.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax2.set_ylabel("y", rotation = 0, fontsize = 32)
ax2.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
ax2.set_yticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], fontsize = 26)

ax2.plot(sol_sec1.t, SOL_TRUE_Y1, color='red')
ax2.plot(sol_sec1.t, SOL_SEC_Y1, color='blue')

ax2.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax3.set_title('(c)', fontsize = 48)
ax3.set_xlim([0,20])
ax3.set_ylim([-0.85, 1.15])

ax3.set_xlabel("t", fontsize = 32)
ax3.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax3.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax3.set_ylabel("z", rotation = 0, fontsize = 32)
ax3.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax3.set_yticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize = 26)

ax3.plot(sol_sec1.t, SOL_TRUE_Z1, color='red')
ax3.plot(sol_sec1.t, SOL_SEC_Z1, color='blue')

ax3.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")

               
midfig.suptitle('Irrational Rotation', fontsize = 64)

(ax4, ax5, ax6) = midfig.subplots(nrows = 1, ncols = 3)


ax4.set_title('(d)', fontsize = 48)
ax4.set_xlim([0,20])
ax4.set_ylim([-2.6, 3.1])

ax4.set_xlabel("t", fontsize = 32)
ax4.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax4.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax4.set_ylabel("x", rotation = 0, fontsize = 32)
ax4.set_yticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
ax4.set_yticklabels([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5], fontsize = 26)

ax4.plot(sol_sec2.t, SOL_TRUE_X2, color='red')
ax4.plot(sol_sec2.t, sol_sec2.y.T[:, 0], color='blue')

ax4.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax5.set_title('(e)', fontsize = 48)
ax5.set_xlim([0,20])
ax5.set_ylim([-3.1, 3.2])

ax5.set_xlabel("t", fontsize = 32)
ax5.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax5.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax5.set_ylabel("y", rotation = 0, fontsize = 32)
ax5.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax5.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 26)

ax5.plot(sol_sec2.t, SOL_TRUE_Y2, color='red')
ax5.plot(sol_sec2.t, sol_sec2.y.T[:, 1], color='blue')

ax5.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax6.set_title('(f)', fontsize = 48)
ax6.set_xlim([0,20])
ax6.set_ylim([-0.85, 0.95])

ax6.set_xlabel("t", fontsize = 32)
ax6.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
ax6.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize = 26)

ax6.set_ylabel("z", rotation = 0, fontsize = 32)
ax6.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax6.set_yticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize = 26)

ax6.plot(sol_sec2.t, SOL_TRUE_Z2, color='red')
ax6.plot(sol_sec2.t, sol_sec2.y.T[:, 2], color='blue')

ax6.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")


bottomfig.suptitle('Stepanoff Flow', fontsize = 64)

(ax7, ax8, ax9) = bottomfig.subplots(nrows = 1, ncols = 3)


ax7.set_title('(g)', fontsize = 48)
ax7.set_xlim([0,10])
ax7.set_ylim([-2.1, 2.1])

ax7.set_xlabel("t", fontsize = 32)
ax7.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax7.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize = 26)

ax7.set_ylabel("x", rotation = 0, fontsize = 32)
ax7.set_yticks([-2, -1, 0, 1, 2])
ax7.set_yticklabels([-2, -1, 0, 1, 2], fontsize = 26)

ax7.plot(sol_sec3.t, SOL_TRUE_X3, color='red')
ax7.plot(sol_sec3.t, sol_sec3.y.T[:, 0], color='blue')

ax7.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax8.set_title('(h)', fontsize = 48)
ax8.set_xlim([0,10])
ax8.set_ylim([-3.1, 3.3])

ax8.set_xlabel("t", fontsize = 32)
ax8.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax8.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize = 30)

ax8.set_ylabel("y", rotation = 0, fontsize = 32)
ax8.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax8.set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize = 30)

ax8.plot(sol_sec3.t, SOL_TRUE_Y3, color='red')
ax8.plot(sol_sec3.t, sol_sec3.y.T[:, 1], color='blue')

ax8.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



ax9.set_title('(i)', fontsize = 48)
ax9.set_xlim([0,10])
ax9.set_ylim([-0.85, 0.95])

ax9.set_xlabel("t", fontsize = 32)
ax9.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax9.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize = 26)

ax9.set_ylabel("z", rotation = 0, fontsize = 32)
ax9.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax9.set_yticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize = 26)

ax9.plot(sol_sec3.t, SOL_TRUE_Z3, color='red')
ax9.plot(sol_sec3.t, sol_sec3.y.T[:, 2], color='blue')

ax9.legend(["True", "SEC"], loc="upper right", frameon = True, edgecolor = "black")



plt.show()
# %%


"""
Component-wise plots of vector field w.r.t. (theta, rho)
"""
# %%
u_a_new = np.empty(25, dtype = float)
u_b_new = np.empty(25, dtype = float)

for i in range(0, 100):
    if (((i+1) % 4 == 0) and (i != 0)):
        u_a_new[int((i+1)/4-1)] = u_a[i]
        u_b_new[int((i+1)/4-1)] = u_b[i]

THETA_LST_NEW, RHO_LST_NEW = np.meshgrid(u_a_new, u_b_new)
theta_lst_periodic = THETA_LST_NEW/np.pi
rho_lst_periodic = RHO_LST_NEW/np.pi

v11lst, v12lst = np.meshgrid(np.linspace(1/np.pi, 1/np.pi, 25), np.linspace(1/np.pi, 1/np.pi, 25))
v21lst, v22lst = np.meshgrid(np.linspace(1/np.pi, 1/np.pi, 25), np.linspace(1/np.pi, ALPHA/np.pi, 25))
v31lst = v31(THETA_LST_NEW, RHO_LST_NEW)/np.pi
v32lst = v32(THETA_LST_NEW, RHO_LST_NEW)/np.pi


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6), layout="constrained")
# fig.suptitle(r"True Vector Field $V(\theta_1,\theta_2)$ on $\pi$-periodic Plot", fontsize = 28)
# fig.tight_layout(pad = 3.0)

ax1.set_title("\n Rational Rotation", fontsize = 24)
ax1.set_xlim([0,2])
ax1.set_ylim([0,2])

ax1.set_xlabel(r"$\theta_1/\pi$", fontsize = 20)
ax1.set_xticks([0, 0.5, 1, 1.5, 2])
ax1.set_xticklabels([0, 0.5, 1, 1.5, 2], fontsize = 20)

ax1.set_ylabel(r"$\theta_2/\pi$", rotation = 0, fontsize = 20)
ax1.set_yticks([0, 0.5, 1, 1.5, 2])
ax1.set_yticklabels([0, 0.5, 1, 1.5, 2], fontsize = 20)


ax1.quiver(theta_lst_periodic, rho_lst_periodic, v11lst, v12lst, angles = 'xy', scale_units = 'xy', scale = 10, linewidth = 2, color = 'blue')

ax1.set_aspect('equal', adjustable='box')



ax2.set_title("\n Irrational Rotation", fontsize = 24)
ax2.set_xlim([0,2])
ax2.set_ylim([0,2])

ax2.set_xlabel(r"$\theta_1/\pi$", fontsize = 20)
ax2.set_xticks([0, 0.5, 1, 1.5, 2])
ax2.set_xticklabels([0, 0.5, 1, 1.5, 2], fontsize = 20)

ax2.set_ylabel(r"$\theta_2/\pi$", rotation = 0, fontsize = 20)
ax2.set_yticks([0, 0.5, 1, 1.5, 2])
ax2.set_yticklabels([0, 0.5, 1, 1.5, 2], fontsize = 20)

ax2.quiver(theta_lst_periodic, rho_lst_periodic, v21lst, v22lst, angles = 'xy', scale_units = 'xy', scale = 10, linewidth = 2, color = 'blue')

ax2.set_aspect('equal', adjustable='box')



ax3.set_title("\n Stepanoff Flow", fontsize = 24)
ax3.set_xlim([0,2])
ax3.set_ylim([0,2])

ax3.set_xlabel(r"$\theta_1/\pi$", fontsize = 20)
ax3.set_xticks([0, 0.5, 1, 1.5, 2])
ax3.set_xticklabels([0, 0.5, 1, 1.5, 2], fontsize = 20)

ax3.set_ylabel(r"$\theta_2/\pi$", rotation = 0, fontsize = 20)
ax3.set_yticks([0, 0.5, 1, 1.5, 2])
ax3.set_yticklabels([0, 0.5, 1, 1.5, 2], fontsize = 20)


ax3.quiver(theta_lst_periodic, rho_lst_periodic, v31lst, v32lst, angles = 'xy', scale_units = 'xy', scale = 10, linewidth = 2, color = 'blue')

ax3.set_aspect('equal', adjustable='box')



plt.show()
# %%






