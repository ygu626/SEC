

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
# Functions specifying the coordinates in R3
# using the angles theat and rho for the latitude and meridian circles
X_func = lambda theta, rho: (a + b*np.cos(theta))*np.cos(rho)
Y_func = lambda theta, rho: (a + b*np.cos(theta))*np.sin(rho)
Z_func = lambda theta: b*np.sin(theta)

# N*N training data points corrdinates in the x, y and z coordinates
TRAIN_X = X_func(training_angle[0, :], training_angle[1, :])
TRAIN_Y = Y_func(training_angle[0, :], training_angle[1, :])
TRAIN_Z = Z_func(training_angle[0, :])

# N*N training data points containing all three coordinates of each point
training_data = np.vstack([TRAIN_X, TRAIN_Y, TRAIN_Z])


x = (a + b*np.cos(training_angle[0, :]))*np.cos(training_angle[1, :])
y = (a + b*np.cos(training_angle[0, :]))*np.sin(training_angle[1, :])
z = b*np.sin(training_angle[0, :])
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


# fig = plt.figure()

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.set_zlim(-3,3)
# ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
# ax1.view_init(36, 26)

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_zlim(-3,3)
# ax2.plot_surface(TRAIN_X, TRAIN_Y, TRAIN_Z, rstride=5, cstride=5, color='k', edgecolors='w')
# ax2.view_init(0, 0)
# ax2.set_xticks([])

# plt.show()


# Embedding map F and its pushforward F_* applied to vector field v
F = lambda theta, rho: np.array([(a + b*np.cos(theta))*np.cos(rho), (a + b*np.cos(theta))*np.sin(rho), a + b*np.sin(theta)])
v1F = lambda theta, rho: np.array([-b*np.sin(theta)*np.cos(rho) - (a + b*np.cos(theta))*np.sin(rho), -b*np.sin(theta)*np.sin(rho) + (a + b*np.cos(theta))*np.cos(rho), b*np.cos(theta)])



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
# %%


# %%
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
"""
Check accuracy of diffusion maps approximation
fir eigenvalues and eigenfunctions of 0-Laplacian
"""

# Check approximations for Laplacian eigenbasis agree with true eigenbasis
# by ploting against linear combinations of true eigenfunctions 

x_coords = training_angle[0, :]
y_coords = training_angle[1, :]

z_true = Phis_normalized[:, 1]
z_dm = np.real(varphi(training_data)[:, 1])


# Creating figure
fig = plt.figure(figsize = (10,10))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter3D(x_coords, y_coords, z_true, color = "blue")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter3D(x_coords, y_coords, z_dm, color = "red")

plt.title("3D scatter plot of diffusion maps approximation")
 
plt.show()
# %%
