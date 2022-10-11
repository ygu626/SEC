#%%

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.integrate import solve_ivp


# number of non-constant eigenform pairs
L = 10    

I = 10
J = 10
K = 3


# Data points and corresponding vector field on the unit circle
THETA_LST = list(np.arange(0, 2*np.pi, np.pi/5))
X_func = lambda theta: np.cos(theta)
Y_func = lambda theta: np.sin(theta)
TRAIN_X = np.array(X_func(THETA_LST))
TRAIN_Y = np.array(Y_func(THETA_LST))

TRAIN_V = np.empty([10, 4], dtype = float)
for i in range(0, 10):
        TRAIN_V[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], -TRAIN_Y[i], TRAIN_X[i]])


# Plot the unit circle and the (training) vector field
X_1, Y_1, U_1, V_1 = zip(*TRAIN_V)
plt.figure()
ax = plt.gca()
ax.quiver(X_1, Y_1, U_1, V_1, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'red')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth = 2.5, color = 'blue')

# plt.draw()
# plt.show()

print(U_1)
print(V_1)


# Eigenvalues lambda_i
lamb = np.empty(2*I+1, dtype = float)
for i in range(0, 2*I+1):
    if i == 0:
        lamb[i] = 0
    elif (i % 2) == 0 and i != 0:
        lamb[i] = i**2/4
    else:
        lamb[i] = (i+1)**2/4


# Eigenfunctions phi_i(theta) and corresponding derivatives
phi_even = lambda i, x: np.sqrt(2)*np.cos(i*x/2)
phi_odd = lambda i, x: np.sqrt(2)*np.sin((i+1)*x/2)

dphi_even = lambda i, x: -np.sqrt(2)*(i/2)*np.sin(i*x/2)
dphi_odd = lambda i, x: np.sqrt(2)*((i+1)/2)*np.cos((i+1)*x/2)


# Apply analysis operator T to obtain v_hat_prime
v_hat_prime = np.empty([2*J+1, 2*K+1], dtype = float)

for i in range(0, 2*J+1):
    if i == 0:
        for j in range(0, 2*K+1):
            if j == 0:
                v_hat_prime[i,j] = 0
            elif (j % 2) == 0 and j != 0:
                inner_prod = lambda x: (1/(2*np.pi))*dphi_even(j,x)
                v_hat_prime[i,j] = quad(inner_prod, 0, 2*np.pi)[0]
            else:
                inner_prod = lambda x: (1/(2*np.pi))*dphi_odd(j,x)
                v_hat_prime[i,j] = quad(inner_prod, 0, 2*np.pi)[0]
    elif (i % 2) == 0 and i != 0:
        for j in range(0, 2*K+1):
            if j == 0:
                v_hat_prime[i,j] = 0
            elif (j % 2) == 0 and j != 0:
                inner_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*dphi_even(j,x)
                v_hat_prime[i,j] = quad(inner_prod, 0, 2*np.pi)[0]
            else:
                inner_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*dphi_odd(j,x)
                v_hat_prime[i,j] = quad(inner_prod, 0, 2*np.pi)[0]
    else:
        for j in range(0, 2*K+1):
            if j == 0:
                v_hat_prime[i,j] = 0
            elif (j % 2) == 0 and j != 0:
                inner_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*dphi_even(j,x)
                v_hat_prime[i,j] = quad(inner_prod, 0, 2*np.pi)[0]
            else:
                inner_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*dphi_odd(j,x)
                v_hat_prime[i,j] = quad(inner_prod, 0, 2*np.pi)[0]

v_hat_prime = np.reshape(v_hat_prime, ((2*J+1)*(2*K+1), 1))


# Compute c_ijk coefficients
c = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
for i in range(0, 2*I+1):
            if i == 0:
                for j in range(0, 2*I+1):
                    if j == 0:
                        for k in range(0, 2*I+1):
                              if k == 0:
                                    triple_prod = lambda x: 1/(2*np.pi)
                                    c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                              elif (k % 2) == 0 and k!= 0:
                                    triple_prod = lambda x: (1/(2*np.pi))*phi_even(k,x)  
                                    c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                              else:
                                    triple_prod = lambda x: (1/(2*np.pi))*phi_odd(k,x)  
                                    c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                    elif (j % 2) == 0 and j != 0:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(j,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(j,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(j,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                    else:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(j,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(j,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(j,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
            elif (i % 2) == 0 and i != 0:
                for j in range(0, 2*I+1):
                    if j == 0:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]                    
                    elif (j % 2) == 0 and j != 0:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_even(j,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_even(j,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_even(j,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                    else:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_odd(j,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_odd(j,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_even(i,x)*phi_odd(j,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
            else:
                for j in range(0, 2*I+1):
                    if j == 0:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                    elif (j % 2) == 0 and j != 0:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_even(j,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_even(j,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_even(j,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                    else:
                        for k in range(0, 2*I+1):
                            if k == 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_odd(j,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            elif (k % 2) == 0 and k != 0:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_odd(j,x)*phi_even(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]
                            else:
                                triple_prod = lambda x: (1/(2*np.pi))*phi_odd(i,x)*phi_odd(j,x)*phi_odd(k,x)
                                c[i,j,k] = quad(triple_prod, 0, 2*np.pi)[0]

# print(c[2,:,:])
# print(np.isnan(c).any())
# print(np.isinf(c).any())


# Compute g_kij Riemannian metric coefficients
g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for k in range(0, 2*I+1):
                                    g[i,j,k] = (lamb[i] + lamb[j] - lamb[k])*c[i,j,k]/2

# print(g[4,2,2])
# print(np.isnan(g).any())
# print(np.isinf(g).any())


c_new = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
for j in range(0, 2*I+1):
    for l in range(0, 2*I+1):
        for m in range(0, 2*I+1):
            c_new[j,l,m] = (1/2)*(lamb[j] + lamb[l] - lamb[m])*c[j,l,m]


# G_new = np.einsum('ikm, jlm -> ijkl', c, c_new, dtype = float)
# G_new = G_new[:2*J+1, :2*K+1, :2*J+1, :2*K+1]
# G_new = np.reshape(G_new, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))


# Compute G_ijkl entries for the Gram operator and its dual
G = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G = np.einsum('ikm, jlm->ijkl', c, g, dtype = float)

G = G[:2*J+1, :2*K+1, :2*J+1, :2*K+1]
G = np.reshape(G, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))

G_dual = np.linalg.pinv(G)
# G_dual = np.reshape(G_dual, (21, 7, 21, 7))

# print(G[2,:])
# print(np.isnan(G).any())
# print(np.isinf(G).any())
# print(G_dual[0,0,0,:])
# %%

# Apply dual Gram operator G^+ to obtain v_hat 
v_hat = np.matmul(G_dual, v_hat_prime)
v_hat = np.reshape(v_hat, (2*J+1, 2*K+1))

# v_hat = np.einsum('ijkl, kl->ij', G_dual, v_hat_prime, dtype = float)
# print(v_hat[1,:])
# %%


F_k = np.zeros([2, 2*I+1], dtype = float)
F_k[1, 1] = 1/np.sqrt(2)
F_k[0, 2] = 1/np.sqrt(2)

g = g[:(2*K+1), :, :]
h_ajl = np.einsum('ak, jkl -> ajl', F_k, g, dtype = float)

c = c[:(2*J+1), :, :]
d_jlm = np.einsum('ij, ilm -> jlm', v_hat, c, dtype = float)

p_am = np.einsum('ajl, jlm -> am', h_ajl, d_jlm, dtype = float)


W_theta_x = np.zeros(10, dtype = float)
W_theta_y = np.zeros(10, dtype = float)
vector_approx = np.empty([10, 4], dtype = float)

for i in range(0, 10):
    for m in range(0, 2*I+1):
        if m == 0:
            W_theta_x[i] += p_am[0, m]
            W_theta_y[i] += p_am[1, m]
        elif (m % 2) == 0 and m != 0:
            W_theta_x[i] += p_am[0, m]*phi_even(m, THETA_LST[i])
            W_theta_y[i] += p_am[1, m]*phi_even(m, THETA_LST[i])
        else:
            W_theta_x[i] += p_am[0, m]*phi_odd(m, THETA_LST[i])
            W_theta_y[i] += p_am[1, m]*phi_odd(m, THETA_LST[i])

            vector_approx[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], W_theta_x[i], W_theta_y[i]])

print(W_theta_x)
print(W_theta_y)
# %%

X_2, Y_2, U_2, V_2 = zip(*vector_approx)

plt.figure()
ax = plt.gca()
ax.quiver(X_1, Y_1, U_1, V_1, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'black')
ax.quiver(X_2, Y_2, U_2, V_2, angles = 'xy', scale_units = 'xy', scale = 0.3, color = 'red')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])

t = np.linspace(0, 2*np.pi, 100000)
ax.plot(np.cos(t), np.sin(t), linewidth = 2.5, color = 'blue')

plt.draw()
plt.show()


plt.scatter(THETA_LST, -TRAIN_Y, color = 'black')
plt.scatter(THETA_LST, W_theta_x, color = 'red')
plt.show()


plt.scatter(THETA_LST, TRAIN_X, color = 'black')
plt.scatter(THETA_LST, W_theta_y, color = 'red')
plt.show()


# ODE solver applied to approximated vector fields
# with initial condition specified
F = lambda theta, w: np.dot(np.array([[W_x(list(range(0,2*I+1)))(theta), 0], [0, W_y(list(range(0,2*I+1)))(theta)]]), w)

t_eval = np.arange(0, 10.01, 0.01)
sol = solve_ivp(F, [0, 10], [1, 1], t_eval=t_eval)

plt.figure(figsize = (12, 8))
plt.plot(sol.y.T[:, 0], sol.y.T[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
