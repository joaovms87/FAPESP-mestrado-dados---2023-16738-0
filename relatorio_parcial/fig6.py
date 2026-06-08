import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.integrate import solve_ivp

# Constants
J = 1
h_0 = 1.1
DELTA = 2
tau = 10
n_spins = 1000
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
ax.plot(t_eval, results, 'k-')
ax.set_title('1000-spin transverse-field Ising chain - fermions method')
ax.set_ylabel(r'$\langle \sum_j \sigma_j^z \rangle$')
ax.set_xlabel(r'$tJ/\hbar$')
ax.grid(True)
ax.set_box_aspect(0.7)
plt.show()
