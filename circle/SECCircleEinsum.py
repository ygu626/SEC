import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from multiprocessing import set_start_method
from itertools import product
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from numpy import random


# number of non-constant eigenform pairs
L = 10

I = 25
J = 10
K = 3

# Number of data points
n = 20


# Double and triple products of functions
def double_prod(f, g):
    def fg(x):
        return f(x) * g(x)

    return fg


def triple_prod(f, g, h):
    def fgh(x):
        return f(x) * g(x) * h(x)

    return fgh


# (L2) integration using scipy quad function
def quad_l2_integral(f, a, b):
    return (1 / (2 * np.pi)) * quad(f, a, b, limit=100)[0]


# (L2) Monte Carlo integration
def monte_carlo_l2_integral(f, a=0, b=2 * np.pi, N=1000):
    u = np.zeros(N)
    for i in range(len(u)):
        u[i] = random.uniform(a, b)

    integral = 0.0
    for i in u:
        integral += (1 / N) * f(i)

    return integral


# Eigenvalues lambda_i
lamb = np.empty(2 * I + 1, dtype=float)
for i in range(0, 2 * I + 1):
    if i == 0:
        lamb[i] = 0
    elif (i % 2) == 0 and i != 0:
        lamb[i] = i ** 2 / 4
    else:
        lamb[i] = (i + 1) ** 2 / 4


# Eigenfunctions phi_i(theta)
def phi_basis(i):
    if i == 0:
        def phi(x):
            return 1
    elif (i % 2) == 1:
        def phi(x):
            return np.sqrt(2) * np.sin((i + 1) * x / 2)
    else:
        def phi(x):
            return np.sqrt(2) * np.cos(i * x / 2)
    return phi


phis = [phi_basis(i) for i in range(2 * I + 1)]


# Derivatives of eigenfunctions dphi_i(theta)
def dphi_basis(i):
    if i == 0:
        def dphi(x):
            return 0
    elif (i % 2) == 1:
        def dphi(x):
            return np.sqrt(2) * ((i + 1) / 2) * np.cos((i + 1) * x / 2)
    else:
        def dphi(x):
            return -np.sqrt(2) * (i / 2) * np.sin(i * x / 2)
    return dphi


dphis = [dphi_basis(i) for i in range(2 * I + 1)]


# Using multiprocessing to reduce runtimes of nested for loops
# Apply analysis operator T to obtain v_hat_prime
# Using quad integration
def v_hat_prime_func(i,j):
    return quad_l2_integral(double_prod(phis[i], dphis[j]), 0, 2 * np.pi)
if __name__ == '__main__':
    set_start_method('fork')
    with mp.Pool() as pool:
        v_hat_prime = pool.starmap(v_hat_prime_func,
                               [(i, j) for i in range(0, 2 * J + 1)
                                       for j in range(0, 2 * K + 1)])


# Compute c_ijk coefficients
# Using Quad integration
def c_func(i, j, k):
    return quad_l2_integral(triple_prod(phis[i], phis[j], phis[k]), 0, 2 * np.pi)

if __name__ == '__main__':
    with mp.Pool() as pool:
        c = pool.starmap(c_func,
                               [(i, j, k) for i in range(0, 2 * I + 1)
                                          for j in range(0, 2 * I + 1)
                                          for k in range(0, 2 * I + 1)])
        c = np.reshape(c, (2 * I + 1, 2 * I + 1, 2 * I +1))

# print(np.isnan(c).any())
# print(np.isinf(c).any())
# %%

# Compute g_kij Riemannian metric coefficients
# Using quad integration
g = np.empty([2 * I + 1, 2 * I + 1, 2 * I + 1], dtype=float)
for i in range(0, 2 * I + 1):
    for j in range(0, 2 * I + 1):
        for k in range(0, 2 * I + 1):
            g[i, j, k] = (lamb[i] + lamb[j] - lamb[k]) * c[i, j, k] / 2

# print(g[4,2,2])
# print(np.isnan(g).any())
# print(np.isinf(g).any())


c_new = np.empty([2 * I + 1, 2 * I + 1, 2 * I + 1], dtype=float)
for j in range(0, 2 * I + 1):
    for l in range(0, 2 * I + 1):
        for m in range(0, 2 * I + 1):
            c_new[j, l, m] = (1 / 2) * (lamb[j] + lamb[l] - lamb[m]) * c[j, l, m]

# G_new = np.einsum('ikm, jlm -> ijkl', c, c_new, dtype = float)
# G_new = G_new[:2*J+1, :2*K+1, :2*J+1, :2*K+1]
# G_new = np.reshape(G_new, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))


# Compute G_ijkl entries for the Gram operator and its dual
# Using quad integration
G = np.zeros([2 * I + 1, 2 * I + 1, 2 * I + 1, 2 * I + 1], dtype=float)
G = np.einsum('ikm, jlm->ijkl', c, g, dtype=float)

G = G[:2 * J + 1, :2 * K + 1, :2 * J + 1, :2 * K + 1]
G = np.reshape(G, ((2 * J + 1) * (2 * K + 1), (2 * J + 1) * (2 * K + 1)))

G_dual = np.linalg.pinv(G)
# G_dual = np.reshape(G_dual, (21, 7, 21, 7))

# print(G[2,:])
# print(np.isnan(G).any())
# print(np.isinf(G).any())
# print(G_dual[0,0,0,:])
# %%

# Apply dual Gram operator G^+ to obtain v_hat
# Using quad integration
v_hat = np.matmul(G_dual, v_hat_prime)
v_hat = np.reshape(v_hat, (2 * J + 1, 2 * K + 1))

# v_hat = np.einsum('ijkl, kl->ij', G_dual, v_hat_prime, dtype = float)
# print(v_hat[1,:])
# %%


# Data points and corresponding vector field on the unit circle
THETA_LST = list(np.arange(0, 2 * np.pi, np.pi / (n / 2)))
X_func = lambda theta: np.cos(theta)
Y_func = lambda theta: np.sin(theta)
TRAIN_X = np.array(X_func(THETA_LST))
TRAIN_Y = np.array(Y_func(THETA_LST))

TRAIN_V = np.empty([n, 4], dtype=float)
for i in range(0, n):
    TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i], TRAIN_X[i]])

X_1, Y_1, U_1, V_1 = zip(*TRAIN_V)

# Apply oushforward map to v_hat to obtain approximated vector fields
# Using quad integration
F_k = np.zeros([2, 2 * I + 1], dtype=float)
F_k[1, 1] = 1 / np.sqrt(2)
F_k[0, 2] = 1 / np.sqrt(2)

g = g[:(2 * K + 1), :, :]
h_ajl = np.einsum('ak, jkl -> ajl', F_k, g, dtype=float)

c = c[:(2 * J + 1), :, :]
d_jlm = np.einsum('ij, ilm -> jlm', v_hat, c, dtype=float)

p_am = np.einsum('ajl, jlm -> am', h_ajl, d_jlm, dtype=float)

W_theta_x = np.zeros(n, dtype=float)
W_theta_y = np.zeros(n, dtype=float)
vector_approx = np.empty([n, 4], dtype=float)


def eigenfunc_x(m):
    return lambda x, y: p_am[0, 0] if m == 0 else (
        p_am[0, m] * np.sqrt(2) * np.cos(m * np.angle(x + (1j) * y) / 2) if ((m % 2) == 0 and m != 0) else p_am[
                                                                                                               0, m] * np.sqrt(
            2) * np.sin((m + 1) * np.angle(x + (1j) * y) / 2))


def eigenfunc_y(m):
    return lambda x, y: p_am[1, 0] if m == 0 else (
        p_am[1, m] * np.sqrt(2) * np.cos(m * np.angle(x + (1j) * y) / 2) if ((m % 2) == 0 and m != 0) else p_am[
                                                                                                               1, m] * np.sqrt(
            2) * np.sin((m + 1) * np.angle(x + (1j) * y) / 2))


def W_x(args):
    return lambda x, y: sum(eigenfunc_x(a)(x, y) for a in args)


def W_y(args):
    return lambda x, y: sum(eigenfunc_y(a)(x, y) for a in args)


for i in range(0, n):
    W_theta_x[i] = W_x(list(range(0, 2 * I + 1)))(TRAIN_X[i], TRAIN_Y[i])
    W_theta_y[i] = W_y(list(range(0, 2 * I + 1)))(TRAIN_X[i], TRAIN_Y[i])
    vector_approx[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x[i], W_theta_y[i]])
print(W_theta_x)
print(W_theta_y)

X_2, Y_2, U_2, V_2 = zip(*vector_approx)

plt.figure()
ax = plt.gca()
ax.quiver(X_1, Y_1, U_1, V_1, angles='xy', scale_units='xy', scale=0.3, color='black')
ax.quiver(X_2, Y_2, U_2, V_2, angles='xy', scale_units='xy', scale=0.3, color='red')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

t = np.linspace(0, 2 * np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth=2.5, color='blue')

plt.draw()
plt.show()

plt.scatter(THETA_LST, -TRAIN_Y, color='black')
plt.scatter(THETA_LST, W_theta_x, color='red')
plt.show()

plt.scatter(THETA_LST, TRAIN_X, color='black')
plt.scatter(THETA_LST, W_theta_y, color='red')
plt.show()

# %%


# ODE solver applied to approximated vector fields
# with initial condition specified
# and the true system

# Define derivative function for the true system
def f_true(t, y):
    # dydt = [-np.sin(np.angle(y[0]+(1j)*y[1])), np.cos(np.angle(y[0]+(1j)*y[1]))]
    dydt = [-np.sin(np.arctan2(y[1], y[0])), np.cos(np.arctan2(y[1], y[0]))]
    return dydt
    
# Define time spans and initial values for the true system
tspan = np.linspace(500, 600, num = 1000)
yinit = [1.56, 1]

# Solve ODE under the true system
sol_true = solve_ivp(lambda t, y: f_true(t, y), 
                [tspan[0], tspan[-1]], yinit, t_eval=tspan, rtol = 1e-5)

# %% 
# Plot solutions to the true system
plt.figure(figsize = (8, 8))
plt.plot(sol_true.y.T[:, 0], sol_true.y.T[:, 1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solutions to ODE under the true system')
plt.show()

# %%
sidefig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (24, 16))
sidefig.suptitle('Solutions to ODE under the true system')

ax1.plot(sol_true.t, sol_true.y.T)
ax1.set_title('x- & y-coordinates prediction w.r.t. time t')

ax2.plot(sol_true.t, sol_true.y.T[:, 0], color = 'black')
ax2.set_title('x-coordinates prediction w.r.t. time t')

ax3.plot(sol_true.t, sol_true.y.T[:, 1], color = 'red')
ax3.set_title('x-coordinates prediction w.r.t. time t')

plt.show()
# %%

# %%
# Define derivative function for the SEC approximated system
def f_sec(t, y):
    dydt = [W_x(list(range(0,2*I+1)))(y[0], y[1]), W_y(list(range(0,2*I+1)))(y[0], y[1])]
    return dydt
    
# Define time spans and initial values for the true system
tspan = np.linspace(500, 600, num = 1000)
yinit = [1.56, 1]

# Solve ODE under the true system
sol_sec = solve_ivp(lambda t, y: f_sec(t, y), 
                [tspan[0], tspan[-1]], yinit, t_eval=tspan, rtol = 1e-5)

# %% 
# Plot solutions to the true system
plt.figure(figsize = (8, 8))
plt.plot(sol_sec.y.T[:, 0], sol_sec.y.T[:, 1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solutions to ODE under the SEC approximated system')
plt.show()

# %%
sidefig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (24, 16))
sidefig.suptitle('Solutions to ODE under the SEC approximated system')

ax1.plot(sol_sec.t, sol_sec.y.T)
ax1.set_title('x- & y-coordinates prediction w.r.t. time t')

ax2.plot(sol_true.t, sol_true.y.T[:, 0], color = 'black')
ax2.plot(sol_sec.t, sol_sec.y.T[:, 0], color = 'black')
ax2.set_title('x-coordinates prediction w.r.t. time t')

ax3.plot(sol_sec.t, sol_sec.y.T[:, 1], color = 'red')
ax3.set_title('x-coordinates prediction w.r.t. time t')

plt.show()
# %%

# %%
sidefig, (ax1, ax2) = plt.subplots(2, figsize = (48, 12))
sidefig.suptitle('Comparisons for solutions to ODE under the true and SEC approximated systems')

ax1.plot(sol_true.t, sol_true.y.T[:, 0], color = 'red')
ax1.plot(sol_sec.t, sol_sec.y.T[:, 0], color = 'blue')
ax1.set_title('x-coordinates prediction w.r.t. time t (true = red, SEC = blue)')

ax2.plot(sol_true.t, sol_true.y.T[:, 1], color = 'red')
ax2.plot(sol_sec.t, sol_sec.y.T[:, 1], color = 'blue')
ax2.set_title('x-coordinates prediction w.r.t. time t (true = red, SEC = blue)')

plt.show()
# %%


# %%
## MONTE CARLO SECTION

# Apply analysis operator T to obtain v_hat_prime
# Using Monte Carlo integration
v_hat_prime_mc = np.empty([2*J+1, 2*K+1], dtype = float)

for i in range(0, 2*J+1):
    for j in range(0, 2*K+1):
        f = double_prod(phis[i], dphis[j])
        v_hat_prime_mc[i, j] = monte_carlo_l2_integral(f)

v_hat_prime_mc = np.reshape(v_hat_prime_mc, ((2*J+1)*(2*K+1), 1))

# print(np.amax(v_hat_prime - v_hat_prime_mc))


# Compute c_ijk coefficients
# Using Monte Carlo integration
c_mc = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
for i in range(0, 2*I+1):
    for j in range(0, 2*I+1):
        for k in range(0, 2*I+1):
            f = triple_prod(phis[i], phis[j], phis[k])
            c_mc[i, j, k] = monte_carlo_l2_integral(f)

# print(np.amax(c - c_mc))


# %%
# Compute g_kij Riemannian metric coefficients
# Using Monte Carlo integration
g_mc = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for k in range(0, 2*I+1):
                                    g_mc[i,j,k] = (lamb[i] + lamb[j] - lamb[k])*c_mc[i,j,k]/2


# Compute G_ijkl entries for the Gram operator and its dual
# Using Monte Carlo integration
G_mc = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G_mc = np.einsum('ikm, jlm->ijkl', c_mc, g_mc, dtype = float)

G_mc = G_mc[:2*J+1, :2*K+1, :2*J+1, :2*K+1]
G_mc = np.reshape(G_mc, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))

G_dual_mc = np.linalg.pinv(G_mc)


# Apply dual Gram operator G^+ to obtain v_hat 
# Using quad integration
v_hat_mc = np.matmul(G_dual_mc, v_hat_prime_mc)
v_hat_mc = np.reshape(v_hat_mc, (2*J+1, 2*K+1))


# %%
# Apply oushforward map to v_hat to obtain approximated vector fields
# Using Monte Carlo integration
g_mc = g_mc[:(2*K+1), :, :]
h_ajl_mc = np.einsum('ak, jkl -> ajl', F_k, g_mc, dtype = float)

c_mc = c_mc[:(2*J+1), :, :]
d_jlm_mc = np.einsum('ij, ilm -> jlm', v_hat_mc, c_mc, dtype = float)

p_am_mc = np.einsum('ajl, jlm -> am', h_ajl_mc, d_jlm_mc, dtype = float)


W_theta_x_mc = np.zeros(n, dtype = float)
W_theta_y_mc = np.zeros(n, dtype = float)
vector_approx_mc = np.empty([n, 4], dtype = float)

def eigenfunc_x_mc(m):
         return lambda x, y: p_am_mc[0,0] if m == 0 else (p_am_mc[0, m]*np.sqrt(2)*np.cos(m*np.angle(x+(1j)*y)/2) if ((m % 2) == 0 and m != 0) else p_am_mc[0, m]*np.sqrt(2)*np.sin((m+1)*np.angle(x+(1j)*y)/2))

def eigenfunc_y_mc(m):
         return lambda x, y: p_am_mc[1,0] if m == 0 else (p_am_mc[1, m]*np.sqrt(2)*np.cos(m*np.angle(x+(1j)*y)/2) if ((m % 2) == 0 and m != 0) else p_am_mc[1, m]*np.sqrt(2)*np.sin((m+1)*np.angle(x+(1j)*y)/2))

def W_x_mc(args):
            return lambda x, y: sum(eigenfunc_x_mc(a)(x, y) for a in args)

def W_y_mc(args):
            return lambda x, y: sum(eigenfunc_y_mc(a)(x, y) for a in args)

for i in range(0, n):
            W_theta_x_mc[i] = W_x_mc(list(range(0,2*I+1)))(TRAIN_X[i], TRAIN_Y[i])
            W_theta_y_mc[i] = W_y_mc(list(range(0,2*I+1)))(TRAIN_X[i], TRAIN_Y[i])
            vector_approx_mc[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x_mc[i], W_theta_y_mc[i]])
print(W_theta_x_mc)
print(W_theta_y_mc)

X_3, Y_3, U_3, V_3 = zip(*vector_approx_mc)


plt.figure()
ax = plt.gca()
# ax.quiver(X_1, Y_1, U_1, V_1, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'black')
ax.quiver(X_3, Y_3, U_3, V_3, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'green')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth = 2.5, color = 'blue')

plt.draw()
plt.show()
# %%
