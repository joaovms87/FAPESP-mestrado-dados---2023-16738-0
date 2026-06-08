# COMPARISON with best methods - FAPESP report

import numpy as cp  # MUDAR PARA CUPY QUANDO FOR RODAR NA GPU
import matplotlib.pyplot as plt
from functools import reduce

# 4 spins Ising - spin space

# Constants
J = 1
h_0 = 1.1
DELTA = 2
tau = 10
n_spins = 4
dim = 2**n_spins

# Pauli matrices using CuPy
Id = cp.eye(2, dtype=cp.complex128)
sigma_x = cp.array([[0., 1.], [1., 0.]], dtype=cp.complex128)
sigma_z = cp.array([[1., 0.], [0., -1.]], dtype=cp.complex128)

def h(t):
    return h_0 if t == 0 else h_0 + DELTA * t / tau

def tensor(matrices):
    return reduce(cp.kron, matrices)

# Precompute tensor products of sigma_z
tensor_products_z = []
for j in range(n_spins):
    matrices = [Id] * n_spins
    matrices[j] = sigma_z
    tensor_products_z.append(tensor(matrices))

# Compute s_x with periodic boundary conditions
s_x = cp.zeros((dim, dim), dtype=cp.complex128)
for j in range(n_spins):
    matrices = [Id] * n_spins
    matrices[j] = sigma_x
    matrices[(j + 1) % n_spins] = sigma_x
    s_x += tensor(matrices)
s_x *= J

Sz_total = sum(tensor_products_z)

def H(t):
    h_val = h(t)
    s_z = sum(h_val * tpz for tpz in tensor_products_z)
    return -s_z - s_x

def f(r, t):
    return -1j * H(t) @ r

def rk4_step(f, r, t, u):
    k1 = u * f(r, t)
    k2 = u * f(r + 0.5 * k1, t + 0.5 * u)
    k3 = u * f(r + 0.5 * k2, t + 0.5 * u)
    k4 = u * f(r + k3, t + u)
    return r + (k1 + 2*k2 + 2*k3 + k4) / 6

def step_ratio(a, b, u, delta):
    diff = (a - b) / 30
    return u * delta / cp.linalg.norm(diff)

def update_u(u, p):
    return min(2*u, u * p**0.25)

# Initial conditions
u = 1e-6
delta = 1e-10
results = []
tpoints = []

# Ground state at t=0
H0 = H(0)
eigvals, eigvecs = cp.linalg.eigh(H0)
r = eigvecs[:, 0]
t = 0

while t < tau:
    tpoints.append(float(t))
    expectation = cp.vdot(r, Sz_total @ r).real
    results.append(expectation)

    r_half = rk4_step(f, r, t, u)
    r1 = rk4_step(f, r_half, t + u, u)
    r2 = rk4_step(f, r, t, 2*u)

    p = step_ratio(r1, r2, u, delta)
    if p >= 1:
        r = r1
        t += 2 * u
    u = update_u(u, p)

# Convert results back to NumPy for plotting IF USING GPU
"""
results_np = cp.asnumpy(cp.array(results))
tpoints_np = cp.asnumpy(cp.array(tpoints))
"""

t_spins = tpoints.copy()
results_spins = results.copy()




##### ISING 4 SPINS - FERMIONS ####

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.integrate import solve_ivp

# Constants
J = 1
h_0 = 1.1
DELTA = 2
tau = 10
n_spins = 4
dim = 2**n_spins

# Pauli matrices
Id = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
sigma_z = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)

def h(t):
    return h_0 if t == 0 else h_0 + DELTA * t / tau

def tensor(matrices):
    return reduce(np.kron, matrices)

def H(k, t):
    h_val = h(t)
    return np.array([[2*(h_val - J*np.cos(k)), -2*1j*J*np.sin(k)],
                     [2*1j*J*np.sin(k), -2*(h_val - J*np.cos(k))]])

def f(t, r):
    return -1j * H(k, t) @ r


# Possible values of k (p=0)
k_list = np.array([(2*n-1)*np.pi/n_spins for n in range(1, int(n_spins/2)+1)])

# Ground state eigenvector for each k
r_list = []
for k in k_list:
    H0 = H(k, 0)
    eigvals, eigvecs = np.linalg.eigh(H0)
    r = eigvecs[:, 0]
    r_list.append(r)

# Solve the evolution for each k
# Time span
t_span = (0, tau)
t_eval = np.linspace(*t_span, 1000)

s_matrix = np.zeros([len(k_list), len(t_eval)], float)
for j, k in enumerate(k_list):
    # Get the initial state for this k
    r0 = r_list[j]
    # Solve using solve_ivp (real-valued interface workaround)
    sol = solve_ivp(
        f, t_span, r0,
        t_eval=t_eval, method='RK45'
    )
    all_r = sol.y
    all_O_k = np.vstack([2*all_r[0], np.zeros(len(all_r[0]))])
    s_matrix[j] = np.array([np.vdot(all_r[:,i], all_O_k[:,i]).real for i in range(len(all_r[0]))])

results = n_spins - 2*np.sum(s_matrix, axis=0)



# Convert results back to NumPy for plotting
"""
results_np = cp.asnumpy(cp.array(results))
tpoints_np = cp.asnumpy(cp.array(tpoints))
"""

t_fermions = t_eval
results_fermions = results

# Plotting
fig, ax = plt.subplots()
ax.plot(t_spins, results_spins, 'r-', label='Liouville equation')
ax.plot(t_fermions, results_fermions, 'k--', label='Fermions')
ax.set_title('Comparison between the methods: 4 spins')
ax.set_ylabel(r'$\langle \sum_j \sigma_j^z \rangle$')
ax.set_xlabel(r'$t J/\hbar$')
ax.legend(fancybox=False, edgecolor='black')
ax.grid(True)
ax.set_box_aspect(0.7)
plt.show()
