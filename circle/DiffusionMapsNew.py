# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy.linalg import eig as eig


# Oarameters
epsilon = 0.15          # RBF bandwidth parameter
alpha = 1
N = 800                 # Number of training data points
l = 21                  # Number of basis functions

# (L2) Deterministic Monte Carlo integration
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

# training_lst = training_data.T.tolist()
# print(training_data.shape)
# %%


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

# print(np.amax(dist_matrix(training_data, training_data)))
# print(dist_matrix(training_data, training_data)[:3,:3])
# %%


# %%
# Heat kernel function
k = lambda x_1, x_2: np.exp(-dist_matrix(x_1, x_2)/(epsilon**2))

# Build kernel matrix K
K = k(training_data, training_data)
# %%


# %%
# Normalization function q that corresponds to diagonal matrix Q
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
print(K_hat[:3,:3])
# %%


# %%
# Normalization function d that corresponds to diagonal matrix D
d = make_normalization_func(k_hat, training_data)

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
print(P[:3,:3])
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
print(S[:3,:3])
# %%


# %%
# Solve eigenvalue problem for similarity matrix S
eigenvalues, eigenvectors = eig(S) 
index = eigenvalues.argsort()[::-1][:l]
Lamb = eigenvalues[index]
Phi = np.real(eigenvectors[:, index])

# Compute approximated 0-Laplacian eigengunctions
lamb = np.empty(l, dtype = float)
for i in range(0, l):
            lamb[i] = 4*(-np.log(np.real(Lamb[i]))/(epsilon**2)) 

print(lamb)         
# %%


# %%
# Normalize eigenfunctions Phi_j
Phi_normalized = np.empty([N, l], dtype = float)
for j in range(0, l):
    Phi_normalized[:, j] = np.real(Phi[:, j])*np.sqrt(N)

def make_varphi(k, x_train, lambs, phis):
    phi_lamb = phis / lambs
    def varphi(x):
        y = k(x, x_train) @ phi_lamb
        return y
    return varphi

Lamb_normalized = np.power(Lamb, 4)
varphi = make_varphi(p, training_data, Lamb, Phi_normalized)
# %%


# %%
# Get x values of the sine wave
time = u
time2 = u

# Amplitude of the sine wave is sine of a variable like time
amplitude = Phi_normalized[:, 2]
amplitude2 = np.real(varphi(training_data)[:, 2])

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