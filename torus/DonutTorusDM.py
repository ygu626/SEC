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
# Apply the coninuous extensiom varphi to the training data set
cont_result = varphi(training_data)
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
z_dm = np.real(cont_result[:, 1])


# Creating figure
fig = plt.figure(figsize = (10,10))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter3D(x_coords, y_coords, z_true, color = "blue")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter3D(x_coords, y_coords, z_dm, color = "red")

plt.title("3D scatter plot of diffusion maps approximation")
 
plt.show()
# %%



# %%
"""
SEC approximation
for pushforward of vector fields on the 2-torus embedded in R3
"""


# Fourier coefficients F_ak pf F w.r.t. difusion maps approximated eigenvectors Phi_j
F_ak = (1/(N**2))*np.matmul(F(training_angle[0, :], training_angle[1, :]), Phis_normalized)
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
print(c[:2,:2,:2])
# %%


# %%
# Compute g_ijp Riemannian metric coefficients
# using Monte Carlo integration
g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
g_coeff = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)

for i in range(0, 2*I+1):
            for j in range(0, 2*I+1):
                        for p in range(0, 2*I+1):
                                    g_coeff[i,j,p] = (lambs_dm[i] + lambs_dm[j] - lambs_dm[p])/2

g = np.multiply(g_coeff, c)


# g = np.empty([2*I+1, 2*I+1, 2*I+1], dtype = float)
# for i in range(0, 2*I+1):
#             for j in range(0, 2*I+1):
#                         for p in range(0, 2*I+1):
#                                     g[i,j,p] = (lambs[i] + lambs[j] - lambs[p])*c[i,j,p]/2
         
print(g[:,:2,:2])
# %%


# %%
# Compute G_ijpq entries for the Gram operator and its dual
# using Monte Carlo integration
G = np.zeros([2*I+1, 2*I+1, 2*I+1, 2*I+1], dtype = float)
G = np.einsum('ipm, jqm -> ijpq', c, g, dtype = float)

G = G[:(2*J+1), :(2*K+1), :(2*J+1), :(2*K+1)]
G = np.reshape(G, ((2*J+1)*(2*K+1), (2*J+1)*(2*K+1)))

print(G[:2,:2])
# %%


# %%
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
threshold = 1/(0.01*np.max(s2))      # Threshold value for truncated SVD


# Compute duall Gram operator G* using pseudoinverse based on truncated singular values of G
G_dual = np.linalg.pinv(G, rcond = threshold)
# G_dual_mc = np.linalg.pinv(G_mc_weighted)
# %%



# %%
"""
Applying analysis operator T to the pushforwaed F_*v (instead of the vector field v)
using Monte Carlo integration
to obtain v_hat'
"""


# (L2) Deterministic Monte Carlo integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product(Phis, training_angle, N = 100):
    v_an = v1F(training_angle[0, :], training_angle[1, :])
    integral = (1/(N**2))*np.sum(Phis*v_an, axis = 1)
    
    return integral
# %%


# %%
# Compute b_am entries using (L2) deterministic Monte Carlo integral
pool = mp.Pool()

def b_func(m):
    return monte_carlo_product(Phis_normalized[:, m], training_angle)


b_am = pool.map(b_func, 
                [m for m in range(0, 2 * I + 1)])

b_am = np.array(b_am).T
# %%


# %%
# Apply analysis operator T to obtain v_hat_prime
# using pushforward vF of vector field v 
# and Monte Carlo integration with weights
gamma_km = np.einsum('ak, am -> km', F_ak, b_am, dtype = float)
# %%


# %%
g = g[:(2*K+1), :, :]


eta_qlm = np.einsum('qkl, km -> qlm', g, gamma_km, dtype = float)
# %%


# %%
c = c[:(2*J+1), :, :]


v_hat_prime = np.einsum('qlm, plm -> pq', eta_qlm, c, dtype = float)

for q in range(0, 2*K+1):
    v_hat_prime[:, q] = np.exp(-tau*lambs[q])*v_hat_prime[:, q]

# v_hat_prime = np.reshape(np.array(v_hat_prime), ((2*J+1), (2*K+1)))
v_hat_prime = np.reshape(v_hat_prime, ((2*J+1)*(2*K+1)))
# print(v_hat_prime[:3,:3])
# %%


# %%
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


h_ajl = np.einsum('ak, jkl -> ajl', F_ak, g_weighted, dtype = float)
# %%

# %%
# c = c[:(2*J+1), :, :]
d_jlm = np.einsum('ij, ilm -> jlm', v_hat, c, dtype = float)

p_am = np.einsum('ajl, jlm -> am', h_ajl, d_jlm, dtype = float)
# %%


# %%
W_theta_x = np.zeros(100, dtype = float)
W_theta_y = np.zeros(100, dtype = float)
W_theta_z = np.zeros(100, dtype = float)

vector_approx = np.empty([100, 6], dtype = float)

def W_theta(training_data):
    varphi_xyz = np.real(varphi(training_data)).T
    return np.matmul(p_am, varphi_xyz)

example_training = training_data[:, :100]
W_temp = W_theta(example_training)

W_theta_x = W_temp[0, :]
W_theta_y = W_temp[0, :]
W_theta_z = W_temp[0, :]

# %%
for i in range(0, 100):
    vector_approx[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], W_theta_x[i], W_theta_y[i], W_theta_z[i]])

# %%

# %%
def plot_vf(vector_approx):
    plt.clf()
    
    # Plot the dataset
    ax = plt.axes(projection ='3d')
   
    x = vector_approx[:, 0]
    y = vector_approx[:, 1]
    z = vector_approx[:, 2]
   
    a = vector_approx[:, 3]
    b = vector_approx[:, 4]
    c = vector_approx[:, 5]
    
    ax.quiver(x, y, z, a, b, c,length = 0.5, color = 'blue')
    
    plt.show()
    
plot_vf(vector_approx)
# %%

# %%
print(vector_approx[:20,2:5])
# %%


def W_x(x, y, z):
    varphi_xyz = np.real(varphi(np.reshape(np.array([x, y, z]), (3, 1))))
    return np.sum(p_am[0, :]*varphi_xyz)

def W_y(x, y, z):
    varphi_xyz = np.real(varphi(np.reshape(np.array([x, y, z]), (3, 1))))
    return np.sum(p_am[1, :]*varphi_xyz)

def W_z(x, y, z):
    varphi_xyz = np.real(varphi(np.reshape(np.array([x, y, z]), (3, 1))))
    return np.sum(p_am[2, :]*varphi_xyz)

# %%
print(training_data.shape)
a = training_data[:, :100]
b = varphi(a)
print(b.shape)
# %%

# %%
c = p_am*b
print(c.shape)
# %%



for i in range(0, 10):
    W_theta_x[i] = W_x(TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i])
    W_theta_y[i] = W_y(TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i])
    W_theta_z[i] = W_z(TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i])

    vector_approx[i, :] = np.array([TRAIN_X[i], TRAIN_Y[i], TRAIN_Z[i], W_theta_x[i], W_theta_y[i], W_theta_z[i]])

print(W_theta_x)
print(W_theta_y)
print(W_theta_z)
# %%


# %%
# Plotting SEC approximated vector fields in R3
# using the donut embedding

def plot_vf(vector_approx):
    plt.clf()
    
    # Plot the dataset
    ax = plt.axes(projection ='3d')
   
    x = vector_approx[:, 0]
    y = vector_approx[:, 1]
    z = vector_approx[:, 2]
   
    a = vector_approx[:, 3]
    b = vector_approx[:, 4]
    c = vector_approx[:, 5]
    
    ax.quiver(x, y, z, a, b, c,length =0.1)
    
    plt.show()
    
plot_vf(vector_approx)
# %%
