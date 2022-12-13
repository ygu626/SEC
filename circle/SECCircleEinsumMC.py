# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import multiprocess as mp
from scipy.integrate import quad
from scipy.integrate import solve_ivp


# Number of non-constant eigenform pairs
I = 10
J = 10
K = 3

# Number of data points
n = 8 

# Weight oaraneter
tau = 0

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
    
# (L2) Deterministic Monte Carlo integration
def monte_carlo_l2_integral(f, a = 0, b = 2*np.pi, N = 800):
    u = np.zeros(N)
    subsets = np.arange(0, N+1, N/400)
    for i in range(0, 400):
        start = int(subsets[i])
        end = int(subsets[i+1])
        u[start:end] = random.uniform(low = (i/400)*b, high = ((i+1)/400)*b, size = end - start)
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


# Data points and corresponding pushforward of vector field ("arrows") on the unit circle
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


# Embedding map F and its pushforward applied to vector field v
F = lambda theta: np.array([np.cos(theta), np.sin(theta)])
vF = lambda theta: np.array([-np.sin(theta), np.cos(theta)])

# Fourier coefficients F_ak pf F w.r.t. eigenfunctions phi_k
F_ak = np.zeros([2, 2*I+1], dtype = float)
F_ak[1, 1] = 1/np.sqrt(2)
F_ak[0, 2] = 1/np.sqrt(2)


# Compute c_ijp coefficients
# Using Monte Carlo integration
pool = mp.Pool()

def c_func_mc(i, j, p):
    return monte_carlo_l2_integral(triple_prod(phis[i], phis[j], phis[p]))

c_mc = pool.starmap(c_func_mc, 
              [(i, j, p) for i in range(0, 2 * I + 1)
                for j in range(0, 2 * I + 1)
                for p in range(0, 2 * I + 1)])
            
c_mc = np.reshape(np.array(c_mc), (2 * I + 1, 2 * I + 1, 2 * I + 1))


# Compute g_ijp Riemannian metric coefficients
# Using Monte Carlo integration
g_mc = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for p in range(0, 2*I+1):
                                    g_mc[i,j,p] = (lamb[i] + lamb[j] - lamb[p])*c_mc[i,j,p]/2


# Compute G_ijpq entries for the Gram operator and its dual
# Using Monte Carlo integration
G_mc = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G_mc = np.einsum('ipm, jqm->ijpq', c_mc, g_mc, dtype = float)

G_mc = G_mc[:2*J+1, :2*K+1, :2*J+1, :2*K+1]
# G_mc = np.reshape(G_mc, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))
# G_dual_mc = np.linalg.pinv(G_mc, rcond = (np.amax(lamb)*1e-3))


# Add in weighted frame elements
G_mc_weighted = np.zeros([2*J+1, 2*K+1, 2*J+1, 2*K+1], dtype = float)

for i in range(0, 2*J+1):
    for j in range(0, 2*K+1):
               for p in range(0, 2*J+1):
                           for q in range(0, 2*K+1):
                                G_mc_weighted[i, j, p, q] = np.exp(-tau*(lamb[j]+lamb[q]))*G_mc[i, j, p, q]

G_mc_weighted = np.reshape(G_mc_weighted, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))

G_dual_mc = np.linalg.pinv(G_mc_weighted, rcond = (np.amax(lamb)*1e-3))


# Deterministic Monte Carlo summation of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product(f, a = 0, b = 2*np.pi, N = 200):
    u = np.zeros(N)
    subsets = np.arange(0, N+1, N/100)
    for i in range(0, 100):
        start = int(subsets[i])
        end = int(subsets[i+1])
        u[start:end] = random.uniform(low = (i/100)*b, high = ((i+1)/100)*b, size = end - start)
    random.shuffle(u)
    
    sum = 0.0
    for j in u:
        sum += (1/N)*(f(j)*vF(j))

    return sum

# Compute b_am entries using Monte Carlo summation
pool = mp.Pool()

def b_func_mc(m):
    return monte_carlo_product(phis[m])

b_am_mc = pool.map(b_func_mc, 
              [m for m in range(0, 2 * I + 1)])
b_am_mc = np.array(b_am_mc).T

gamma_km_mc = np.einsum('ak, am -> km', F_ak, b_am_mc, dtype = float)

g_mc = g_mc[:(2*K+1), :, :]
eta_qlm_mc = np.einsum('qkl, km -> qlm', g_mc, gamma_km_mc, dtype = float)

v_hat_prime_mc = np.einsum('qlm, plm -> pq', eta_qlm_mc, c_mc, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime_mc[:, q] = np.exp(-tau*lamb[q])*v_hat_prime_mc[:, q]
    
v_hat_prime_mc = np.reshape(np.array(v_hat_prime_mc), ((2*J+1)*(2*K+1), 1))
# %%


# %%
# Apply analysis operator T to obtain v_hat_prime
# Using Monte Carlo integration with weights
# pool = mp.Pool()

# def v_hat_prime_func_mc(p, q):
#     return np.exp(-tau*lamb[q])*monte_carlo_l2_integral(double_prod(phis[p], dphis[q]))

# v_hat_prime_mc = pool.starmap(v_hat_prime_func_mc, 
#                         [(p, q) for p in range(0, 2 * J + 1)
#                          for q in range(0, 2 * K + 1)])
            
# v_hat_prime_mc = np.reshape(np.array(v_hat_prime_mc), ((2*J+1)*(2*K+1), 1))
# %%

# Apply dual Gram operator G^+ to obtain v_hat 
# Using quad integration
v_hat_mc = np.matmul(G_dual_mc, v_hat_prime_mc)
v_hat_mc = np.reshape(v_hat_mc, (2*J+1, 2*K+1))


# Apply oushforward map F_* of embedding F to v_hat to obtain approximated vector fields
# Using Monte Carlo integration with weights

# g_mc = g_mc[:(2*K+1), :, :]

g_mc_weighted = np.zeros([2*K+1, 2*I+1, 2*I+1], dtype = float)
for j in range(0, 2*K+1):
    g_mc_weighted[j, :, :] = np.exp(-tau*lamb[j])*g_mc[j, :, :]

h_ajl_mc = np.einsum('ak, jkl -> ajl', F_ak, g_mc_weighted, dtype = float)

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
ax.set_title('Comparisons of True and SEC Approximated Vector Fields')

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth = 2.5, color = 'blue')

plt.draw()
plt.show()


sidefig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sidefig.suptitle('Comparisons of True and SEC Approximated Vector Fields')

ax1.scatter(x = THETA_LST, y = -TRAIN_Y, color='black')
ax1.scatter(x = THETA_LST, y = W_theta_x_mc, color='red')
ax1.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax1.set_xlabel("Angle Theta")
ax1.set_ylabel("X-coordinates of Vector Fields")
ax1.set_title('X-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

ax2.scatter(x = THETA_LST, y = TRAIN_X, color='black')
ax2.scatter(x = THETA_LST, y = W_theta_y_mc, color='red')
ax2.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
ax2.set_xlabel("Angle Theta")
ax2.set_ylabel("Y-coordinates of Vector Fields")
ax2.set_title('Y-coordinates w.r.t. Angle Theta (true = black, SEC = red)')

plt.show()
# %%


# %%
# ODE solver applied to the SEC approximated vector fields
# with initial condition specified
# and the true system

# Define derivative function for the true system
def f_true(t, y):
    # dydt = [-np.sin(np.angle(y[0]+(1j)*y[1])), np.cos(np.angle(y[0]+(1j)*y[1]))]
    dydt = [-np.sin(np.arctan2(y[1], y[0])), np.cos(np.arctan2(y[1], y[0]))]
    return dydt


# Define time spans and initial values for the true system
tspan = np.linspace(0, 10000, num=1000)
yinit = [10, 24]

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


# Define derivative function for the SEC approximated system
def f_sec_mc(t, y):
    dydt = [W_x_mc(list(range(0, 2 * I + 1)))(y[0], y[1]), W_y_mc(list(range(0, 2 * I + 1)))(y[0], y[1])]
    return dydt


# Define time spans and initial values for the SEC approximated system
tspan = np.linspace(0, 10000, num=1000)
yinit = [10, 24]

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
