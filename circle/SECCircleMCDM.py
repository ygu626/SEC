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
J = 5           # Outer index for eigenfunctions
K = 5           # Index for gradients of eigenfunctions
n = 8          # Number of approximated tangent vectors
N = 800         # Number of Monte Carlo training data points 

epsilon = 0.25  # RBF bandwidth parameter
tau = 0         # Weight parameter for Laplacian eigenvalues
alpha = 1       # Weight parameter for Markov kernel matrix



"""
Training data set
with pushforward of vector fields v on the circle
and Embedding map F with pushforward vF
"""


# Deterministically sampled Monte Carlo training data points
def monte_carlo_points(a = 0, b = 2*np.pi, N = 800):
    u = np.zeros(N)
    subsets = np.arange(0, N+1, N/400)
    for i in range(0, 400):
        start = int(subsets[i])
        end = int(subsets[i+1])
        u[start:end] = random.uniform(low = (i/400)*b, high = ((i+1)/400)*b, size = end - start)
    random.shuffle(u)
    
    training_data = np.empty([2, N], dtype = float)
    for j in range(0, N):
            training_data[:, j] = np.array([np.cos(u[j]), np.sin(u[j])])
    
    return u, training_data

u, training_data = monte_carlo_points()
plt.scatter(training_data[0,:], training_data[1,:])
plt.show


# n pushforward of vector field v ("arrows") on the circle
# given points (x, y) specified by angle theat on the circle
THETA_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))
X_func = lambda theta: np.cos(theta)
Y_func = lambda theta: np.sin(theta)
TRAIN_X = np.array(X_func(THETA_LST))
TRAIN_Y = np.array(Y_func(THETA_LST))


TRAIN_V = np.empty([n, 4], dtype = float)
for i in range(0, n):
    TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i], TRAIN_X[i]])

X_1, Y_1, U_1, V_1 = zip(*TRAIN_V)

print(U_1)
print(V_1)



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



"""
SEC approximation
for pushforward of vector fields on the circle
"""


# %%
# Fourier coefficients F_ak pf F w.r.t. difusion maps eigenvectors Phi_j
F_ak_dm = (1/N)*np.matmul(F(u),Phis_normalized)


# Compute c_ijp coefficients
# Using Monte Carlo integration
pool = mp.Pool()

def c_func_mc_dm(i, j, p):
    return (1/N)*np.sum(Phis_normalized[:, i]*Phis_normalized[:, j]*Phis_normalized[:, p])

c_mc_dm = pool.starmap(c_func_mc_dm, 
              [(i, j, p) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for p in range(0, 2 * I + 1)])
            
c_mc_dm = np.reshape(np.array(c_mc_dm), (2 * I + 1, 2 * I + 1, 2 * I + 1))
# print(c_mc_dm[:,3,3])


# Compute g_ijp Riemannian metric coefficients
# Using Monte Carlo integration
g_mc_dm = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
g_mc_dm_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for p in range(0, 2*I+1):
                                    g_mc_dm_coeff[i,j,p] = (lambs_dm[i] + lambs_dm[j] - lambs_dm[p])/2
                                    
g_mc_dm = np.multiply(g_mc_dm_coeff, c_mc_dm)

# g_mc_dm = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
# for i in range(0, 2*I+1):
#             for j in range(0, 2*I+1):
#                         for p in range(0, 2*I+1):
#                                     g_mc_dm[i,j,p] = (lambs_dm[i] + lambs_dm[j] - lambs_dm[p])*c[i,j,p]/2
         
# print(g_mc_dm[:,3,3])


# Compute G_ijpq entries for the Gram operator and its dual
# Using Monte Carlo integration
G_mc_dm = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G_mc_dm = np.einsum('ipm, jqm -> ijpq', c_mc_dm, g_mc_dm, dtype = float)

G_mc_dm = G_mc_dm[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G_mc_dm = np.reshape(G_mc_dm, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))


# Perform singular value decomposition (SVD) of the Gram operator G
# and plot these singular values
u2_dm, s2_dm, vh = np.linalg.svd(G_mc_dm, full_matrices = True, compute_uv = True, hermitian = False)

sing_lst = np.arange(0, len(s2_dm), 1, dtype = int)
plt.figure(figsize=(24, 6))
plt.scatter(sing_lst, s2_dm, color = 'red')

plt.xticks(np.arange(0, ((2*J+1)*(2*K+1))+0.1, 1))
plt.xlabel('Indices')
plt.yticks(np.arange(0, max(s2_dm)+0.1, 1))
plt.ylabel('Singular Values')
plt.title('Singular Values of the Gram Operator G_ijpq (descending order)')

plt.show()

# print(s2_dm)


# Teuncate singular values of G based based on 1% of the largest singular valuecof G
threshold = 1/(0.04*np.max(s2_dm))      # Threshold value for truncated SVD


# Compute duall Gram operator G* using pseudoinverse based on truncated singular values of G
G_dual_mc_dm = np.linalg.pinv(G_mc_dm, rcond = threshold)
# G_dual_mc = np.linalg.pinv(G_mc_weighted)

# %%


"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""


# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product_dm(Phis, u, N = 800):
    v_an = vF(u)
    integral = (1/N)*np.sum(Phis*v_an, axis = 1)
    
    return integral


# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func_mc_dm(m):
    return monte_carlo_product_dm(Phis_normalized[:, m], u)

b_am_mc_dm = pool.map(b_func_mc_dm, 
              [m for m in range(0, 2 * I + 1)])
b_am_mc_dm = np.array(b_am_mc_dm).T


# Apply analysis operator T to obtain v_hat_prime
# Using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km_mc_dm = np.einsum('ak, am -> km', F_ak_dm, b_am_mc_dm, dtype = float)

g_mc_dm = g_mc_dm[:(2*K+1), :, :]
eta_qlm_mc_dm = np.einsum('qkl, km -> qlm', g_mc_dm, gamma_km_mc_dm, dtype = float)

c_mc_dm = c_mc_dm[:(2*J+1), :, :]
v_hat_prime_mc_dm = np.einsum('qlm, plm -> pq', eta_qlm_mc_dm, c_mc_dm, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime_mc_dm[:, q] = np.exp(-tau*lambs_dm[q])*v_hat_prime_mc_dm[:, q]

# v_hat_prime_mc_dm = np.reshape(np.array(v_hat_prime_mc_dm), ((2*J+1), (2*K+1)))
v_hat_prime_mc_dm = np.reshape(v_hat_prime_mc_dm, ((2*J+1)*(2*K+1), 1))
# print(v_hat_prime_mc_dm[:3,:3])


# Apply dual Gram operator G^+ to obtain v_hat 
# Using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat_mc_dm = np.matmul(G_dual_mc_dm, v_hat_prime_mc_dm)
v_hat_mc_dm = np.reshape(v_hat_mc_dm, (2*J+1, 2*K+1))


# Apply pushforward map F_* of the embedding F to v_hat to obtain approximated vector fields
# Using Monte Carlo integration with weights
# g_mc = g_mc[:(2*K+1), :, :]

g_mc_dm_weighted = np.zeros([2*K+1, 2*I+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_mc_dm_weighted[j, :, :] = np.exp(-tau*lambs_dm[j])*g_mc_dm[j, :, :]

h_ajl_mc_dm = np.einsum('ak, jkl -> ajl', F_ak_dm, g_mc_dm_weighted, dtype = float)

# c_mc = c_mc[:(2*J+1), :, :]
d_jlm_mc_dm = np.einsum('ij, ilm -> jlm', v_hat_mc_dm, c_mc_dm, dtype = float)

p_am_mc_dm = np.einsum('ajl, jlm -> am', h_ajl_mc_dm, d_jlm_mc_dm, dtype = float)


W_theta_x_mc_dm = np.zeros(n, dtype = float)
W_theta_y_mc_dm = np.zeros(n, dtype = float)
vector_approx_mc_dm = np.empty([n, 4], dtype = float)

def W_x_mc_dm(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am_mc_dm[0, :]*varphi_xy)

def W_y_mc_dm(x, y):
    varphi_xy = np.real(varphi(np.reshape(np.array([x, y]), (2, 1))))
    return np.sum(p_am_mc_dm[1, :]*varphi_xy)

for i in range(0, n):
    W_theta_x_mc_dm[i] = W_x_mc_dm(TRAIN_X[i], TRAIN_Y[i])
    W_theta_y_mc_dm[i] = W_y_mc_dm(TRAIN_X[i], TRAIN_Y[i])
    vector_approx_mc_dm[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x_mc_dm[i], W_theta_y_mc_dm[i]])
print(W_theta_x_mc_dm)
print(W_theta_y_mc_dm)

X_4, Y_4, U_4, V_4 = zip(*vector_approx_mc_dm)


# Comparison between true pusbforward of vector field and pushforward of SEC approximated vector field
plt.figure()
ax = plt.gca()
ax.quiver(X_1, Y_1, U_1, V_1, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'black')
ax.quiver(X_4, Y_4, U_4, V_4, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'red')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_title('Comparisons of True and SEC Approximated Vector Fields')

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth = 2.5, color = 'blue')

plt.draw()
plt.show()


sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sidefig.suptitle('Comparisons of True and SEC Approximated Vector Fields')

ax1.scatter(x = THETA_LST, y = -TRAIN_Y, color='black')
ax1.scatter(x = THETA_LST, y = W_theta_x_mc_dm, color='red')
ax1.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax1.set_xlabel("Angle Theta")
ax1.set_ylabel("X-coordinates of Vector Fields")
ax1.set_title('X-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

ax2.scatter(x = THETA_LST, y = TRAIN_X, color='black')
ax2.scatter(x = THETA_LST, y = W_theta_y_mc_dm, color='red')
ax2.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax2.set_xlabel("Angle Theta")
ax2.set_ylabel("Y-coordinates of Vector Fields")
ax2.set_title('Y-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

plt.show()
# %%



"""
Plot the pushfoward map F_* of the embedding F
as a quiver plot in R62 to capture tbe bias in SEC approximation
usibg meshgrid as the training data set
"""

# %%
m = 20           # Square root of number of points used in quiver plot of F_*

x_train_new = np.linspace(-1.5,1.5, m)
y_train_new = np.linspace(-1.5, 1.5, m)

X_TRAIN_NEW, Y_TRAIN_NEW = np.meshgrid(x_train_new, y_train_new)

# print(X_TRAIN_NEW.shape)
# print(Y_TRAIN_NEW.shape)

W_theta_x_new = np.zeros([m, m], dtype = float)
W_theta_y_new = np.zeros([m, m], dtype = float)

for i in range(0, m):
    for j in range(0, m):
        W_theta_x_new[i, j] = W_x_mc_dm(X_TRAIN_NEW[i, j], Y_TRAIN_NEW[i, j])
        W_theta_y_new[i, j] = W_y_mc_dm(X_TRAIN_NEW[i, j], Y_TRAIN_NEW[i, j])
        # vector_approx_mc_dm[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x_mc_dm[i], W_theta_y_mc_dm[i]])
        
U_TRAIN_NEW = W_theta_x_new
V_TRAIN_NEW = W_theta_y_new


plt.figure()
ax = plt.gca()
plt.quiver(X_TRAIN_NEW, Y_TRAIN_NEW, U_TRAIN_NEW, V_TRAIN_NEW)

ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_title('Quiver Plot of the SEC Approximated function F: R2-->R2')

plt.show()
# %%



"""
Solve ODEs in the SEC approximated system
and compare with the solution in the true system
"""


# ODE solver applied to the SEC approximated vector fields
# with initial condition specified
# and the true system


"""
True System
"""
# %%
# Define derivative function for the true system
def f_true(t, y):
    # dydt = [-np.sin(np.angle(y[0]+(1j)*y[1])), np.cos(np.angle(y[0]+(1j)*y[1]))]
    dydt = [-np.sin(np.arctan2(y[1], y[0])), np.cos(np.arctan2(y[1], y[0]))]
    return dydt

# Define time spans and initial values for the true system
tspan = np.linspace(0, 10, num=1000)
yinit = [1, 0]

# Solve ODE under the true system
sol_true = solve_ivp(lambda t, y: f_true(t, y),
                     [tspan[0], tspan[-1]], yinit, t_eval=tspan, rtol=1e-5)


# Plot solutions to the true system
plt.figure(figsize=(8, 8))
plt.plot(sol_true.y.T[:, 0], sol_true.y.T[:, 1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solutions to ODE under the true system')
plt.show()


sidefig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(24, 16))
sidefig.suptitle('Solutions to ODE under the true system')

ax1.plot(sol_true.t, sol_true.y.T)
ax1.set_title('x- & y-coordinates prediction w.r.t. time t')

ax2.plot(sol_true.t, sol_true.y.T[:, 0], color='black')
ax2.set_title('x-coordinates prediction w.r.t. time t')

ax3.plot(sol_true.t, sol_true.y.T[:, 1], color='red')
ax3.set_title('y-coordinates prediction w.r.t. time t')

plt.show()

# %%


"""
SEC Approximated System
"""

# %%
# Define derivative function for the SEC approximated system
def f_sec_mc(t, y):
    dydt = [W_x_mc_dm(y[0], y[1]), W_y_mc_dm(y[0], y[1])]
    return dydt

# Define time spans and initial values for the SEC approximated system
tspan = np.linspace(0, 10, num=1000)
yinit = [0.9, 0]

# Solve ODE under the SEC approximated system
sol_sec_mc = solve_ivp(lambda t, y: f_sec_mc(t, y),
                    [tspan[0], tspan[-1]], yinit, t_eval=tspan, rtol=1e-5)


# Plot solutions to the SEC approximated system
plt.figure(figsize=(8, 8))
plt.plot(sol_sec_mc.y.T[:, 0], sol_sec_mc.y.T[:, 1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solutions to ODE under the SEC approximated system')
plt.show()


sidefig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(24, 16))
sidefig.suptitle('Solutions to ODE under the SEC approximated system')

ax1.plot(sol_sec_mc.t, sol_sec_mc.y.T)
ax1.set_title('x- & y-coordinates prediction w.r.t. time t')

ax2.plot(sol_sec_mc.t, sol_sec_mc.y.T[:, 0], color='black')
ax2.set_title('x-coordinates prediction w.r.t. time t')

ax3.plot(sol_sec_mc.t, sol_sec_mc.y.T[:, 1], color='red')
ax3.set_title('y-coordinates prediction w.r.t. time t')

plt.show()


sidefig, (ax1, ax2) = plt.subplots(2, figsize=(48, 12))
sidefig.suptitle('Comparisons for solutions to ODE under the true and SEC approximated systems')

ax1.plot(sol_true.t, sol_true.y.T[:, 0], color='red')
ax1.plot(sol_sec_mc.t, sol_sec_mc.y.T[:, 0], color='blue')
ax1.set_title('x-coordinates prediction w.r.t. time t (true = red, SEC = blue)')

ax2.plot(sol_true.t, sol_true.y.T[:, 1], color='red')
ax2.plot(sol_sec_mc.t, sol_sec_mc.y.T[:, 1], color='blue')
ax2.set_title('y-coordinates prediction w.r.t. time t (true = red, SEC = blue)')

plt.show()
# %%
