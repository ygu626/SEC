# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import multiprocess as mp
from scipy.integrate import quad
from scipy.integrate import solve_ivp


# number of non-constant eigenform pairs
L = 10    

I = 10
J = 10
K = 3

# Number of data points
n = 4 


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
    return (1/(2*np.pi))*quad(f, a, b, limit = 100)[0]
    
# (L2) Monte Carlo integration
def monte_carlo_l2_integral(f, a = 0, b = 2*np.pi, N = 10000):
    u = np.zeros(N)
    subsets = np.arange(0, N+1, N/2000)
    for i in range(0, 2000):
        start = int(subsets[i])
        end = int(subsets[i+1])
        u[start:end] = random.uniform(low = (i/2000)*b, high = ((i+1)/2000)*b, size = end - start)
    random.shuffle(u)

    integral = 0.0
    for j in u:
        integral += (1/N)*f(j)

    return integral


# Eigenvalues lambda_i
lamb = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
    if i == 0:
        lamb[i] = 0
    elif (i % 2) == 0 and i != 0:
        lamb[i] = i**2/4
    else:
        lamb[i] = (i+1)**2/4


# Eigenfunctions phi_i(theta)
def phi_basis(i):
    if i == 0:
        def phi(x):
            return 1
    elif (i % 2) == 1:
        def phi(x):
            return np.sqrt(2)*np.sin((i + 1) * x / 2)
    else:
        def phi(x):
            return np.sqrt(2)*np.cos(i * x / 2)
    return phi

phis = [phi_basis(i) for i in range(2*I+1)]


# Derivatives of eigenfunctions dphi_i(theta)
def dphi_basis(i):
    if i == 0:
        def dphi(x):
            return 0
    elif (i % 2) == 1:
        def dphi(x):
            return np.sqrt(2)*((i + 1) / 2)*np.cos((i + 1) * x / 2)
    else:
        def dphi(x):
            return -np.sqrt(2)*(i / 2)*np.sin(i * x / 2)
    return dphi

dphis = [dphi_basis(i) for i in range(2*I+1)]


# Data points and corresponding vector field on the unit circle
THETA_LST = list(np.arange(0, 2*np.pi, np.pi/(n/2)))
X_func = lambda theta: np.cos(theta)
Y_func = lambda theta: np.sin(theta)
TRAIN_X = np.array(X_func(THETA_LST))
TRAIN_Y = np.array(Y_func(THETA_LST))

TRAIN_V = np.empty([n, 4], dtype = float)
for i in range(0, n):
        TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i], TRAIN_X[i]])

X_1, Y_1, U_1, V_1 = zip(*TRAIN_V)


# Apply analysis operator T to obtain v_hat_prime
# Using Monte Carlo integration
p = mp.Pool()

def v_hat_prime_func_mc(i, j):
    return monte_carlo_l2_integral(double_prod(phis[i], dphis[j]))

v_hat_prime_mc = p.starmap(v_hat_prime_func_mc, 
                        [(i, j) for i in range(0, 2 * J + 1)
                         for j in range(0, 2 * K + 1)])
            
v_hat_prime_mc = np.reshape(np.array(v_hat_prime_mc), ((2*J+1)*(2*K+1), 1))


# Compute c_ijk coefficients
# Using Monte Carlo integration
p = mp.Pool()

def c_func_mc(i, j, k):
    return monte_carlo_l2_integral(triple_prod(phis[i], phis[j], phis[k]))

c_mc = p.starmap(c_func_mc, 
              [(i, j, k) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for k in range(0, 2 * I + 1)])
            
c_mc = np.reshape(np.array(c_mc), (2 * I + 1, 2 * I + 1, 2 * I + 1))
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
# %%

# %%
# Apply dual Gram operator G^+ to obtain v_hat 
# Using quad integration
v_hat_mc = np.matmul(G_dual_mc, v_hat_prime_mc)
v_hat_mc = np.reshape(v_hat_mc, (2*J+1, 2*K+1))

# %%
# Apply oushforward map to v_hat to obtain approximated vector fields
# Using Monte Carlo integration
F_k = np.zeros([2, 2*I+1], dtype = float)
F_k[1, 1] = 1/np.sqrt(2)
F_k[0, 2] = 1/np.sqrt(2)

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
ax.quiver(X_1, Y_1, U_1, V_1, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'black')
ax.quiver(X_3, Y_3, U_3, V_3, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'red')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth = 2.5, color = 'blue')

plt.draw()
plt.show()
# %%
