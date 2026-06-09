import numpy as np
from numpy.random import normal
import scipy.linalg as sl
from scipy.optimize import fsolve
from scipy.stats import entropy
import matplotlib.pyplot as plt

plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

n = 100
w = 2*n
d_0 = 0.7
DELTA = 0.3
beta = 1.0

tau = 1

V = normal(0, np.sqrt(1/10/int(n)*d_0**2/16/np.pi**2), (int(n), int(n)))

p1 = 1/(2-2**(1/3))
p2 = 1-2*p1

def d(t):
  return d_0 + t/tau*DELTA

def Ham(t):
  aux = np.zeros([2*n, 2*n], float)
  for i in range(n):
    aux[i, i] = i/(n-1)*d(t)
  for j in range(n):
    aux[n+j, n+j] = j/(n-1)*d(t)
  aux[:n, n:] = V.copy()
  aux[n:, :n] = V.copy().T
  return aux

def H_diag(t):
  aux = np.empty(2*n)
  for i in range(n):
    aux[i] = i/(n-1)*d(t)
  for j in range(n):
    aux[n+j] = j/(n-1)*d(t)
  return aux

def rho_eq(t, b):
  eigenvalues, Q = sl.eigh(Ham(t))
  aux = Q @ np.diag(np.exp(-b*eigenvalues)) @ Q.conj().T
  return aux/np.trace(aux)

def rho_ad(t):
    eigenvalues, eigenvectors = sl.eigh(Ham(t))
    eigenvalues0, eigenvectors0 = sl.eigh(Ham(0))
    p = 0
    for i in range(w):
        v = eigenvectors[:, i].copy()
        v /= np.linalg.norm(v)
        p += np.exp(-beta*eigenvalues0[i])*np.outer(v, v)

    return p / np.trace(p)



def step(r, t, h):
  U1 = Q @ np.diag(np.exp(-1j*p1*values*h/2)) @ Q.conj().T
  U2 = Q @ np.diag(np.exp(-1j*(p1+p2)*values*h/2)) @ Q.conj().T
  diag1 = np.exp(-1j*p1*H_diag(t+h/2)*h)
  diag2 = np.exp(-1j*p2*H_diag(t+h/2)*h)

  U = U1 @ (diag1[:, None] * U2 @ (diag2[:, None] * U2) * diag1) @ U1
  return U @ r @ U.conj().T


def D(p1, p2):
    return np.trace(p1 @ sl.logm(p1) - p1 @ sl.logm(p2))


def equation_rho(b):
    return np.trace((rho - rho_eq(tau,b)) @ Ham(tau))

w = 2*n

H_1 = np.zeros([2*n, 2*n], float)
H_1[:n, n:] = V.copy()
H_1[n:, :n] = V.copy().T

values, Q = sl.eigh(H_1)

lambda2 = 1/n**2*np.sum(V**2)
tau_th = (d_0+DELTA)/(4*np.pi*n*lambda2)

# list of the process durations that will be investigated
taupoints = np.logspace(np.log10(tau_th/100), np.log10(tau_th*10), 20)

Dad_points = np.empty(len(taupoints), complex)
beta_tilde = np.empty(len(taupoints), complex)
DeltaS_points = np.empty(len(taupoints), complex)
#D1 = np.empty(len(taupoints), complex)
#D2 = np.empty(len(taupoints), complex)
#D_tilde = np.empty(len(taupoints), complex)
Wex_points = np.empty(len(taupoints), complex)
tr_points = np.empty(len(taupoints), complex)
obs_entropy = np.empty(len(taupoints), complex)
diag_entropy = np.empty(len(taupoints), complex)
obs_entropy_traditional = np.empty(len(taupoints), complex)

Sobs_coarse = np.empty(len(taupoints), complex)
Strad_coarse = np.empty(len(taupoints), complex)


tau = 1.0
rho0 = rho_eq(0, beta)


for q, tau in enumerate(taupoints):
        t = 0
        r = rho0.copy()
        print(tau)
        if tau < 10:
          u = 0.05
        elif tau < 1000:
          u = 0.05
        else:
          u = 0.05
        while t<tau:
              r = step(r, t, u)
              t += u


        print('EVOLUTION OK')

        rho = r
        r_ad = rho_ad(tau)
        H = Ham(tau)

        beta_tilde[q] = fsolve(equation_rho, beta)[0]
        r_tilde = rho_eq(tau, beta_tilde[q])

        Wex_points[q] = np.einsum('ij,ji->', rho, H) - np.einsum('ij,ji->', r_ad, H)
        #D1[q] = D(r_ad, r_tilde)
        #D2[q] = D(r_tilde, r_ad)
        Dad_points[q] = D(rho, r_ad)
        DeltaS_points[q] = - np.trace(r_tilde @ sl.logm(r_tilde)) + np.trace(rho0 @ sl.logm(rho0))
        #D_tilde[q] = D(rho, rho_eq(tau, beta_tilde[q]))
        tr_points[q] = np.trace(rho)

        evals, evecs = sl.eigh(H)
        evals0, evecs0 = sl.eigh(Ham(0))

        n_measurements = 10
        DeltaE_0 = -(evals0[0] - evals0[-1])/n_measurements
        DeltaE = -(evals[0] - evals[-1])/n_measurements

        a = 0
        b = 0
        Mx_list = []
        Mx_list0 = []
        for i in range(n_measurements):
            Mx_aux = np.zeros_like(H)
            Mx_aux0 = np.zeros_like(H)
            e_fin = evals[0] + (i+1)*DeltaE
            e_fin0 = evals0[0] + (i+1)*DeltaE_0
            while evals[a] < e_fin:
                Mx_aux += np.outer(evecs[:, a], evecs[:, a].conj())
                a += 1
            Mx_list.append(Mx_aux)
            while evals0[b] < e_fin0:
                Mx_aux0 += np.outer(evecs0[:, b], evecs0[:, b].conj())
                b += 1
            Mx_list0.append(Mx_aux0)



        coarse_p_rho = []
        coarse_p_prior = []
        coarse_p0 = []
        coarse_Id = []
        coarse_Id0 = []
        for i in range(n_measurements):
            Mx = Mx_list[i]
            Mx_0 = Mx_list0[i]
            coarse_p_rho.append(np.trace(Mx @ rho))
            coarse_p_prior.append(np.trace(Mx @ rho_eq(tau, beta_tilde[q])))
            coarse_p0.append(np.trace(Mx_0 @ rho0))

            coarse_Id.append(np.trace(Mx))
            coarse_Id0.append(np.trace(Mx_0))

        coarse_p_rho = np.real(np.array(coarse_p_rho))
        coarse_p_prior = np.real(np.array(coarse_p_prior))
        coarse_p0 = np.real(np.array(coarse_p0))


        p_tilde = np.real(np.exp(-beta_tilde[q]*evals))
        p_tilde = p_tilde/sum(p_tilde)
        p_rho = np.real(np.diag(evecs.conj().T @ rho @ evecs))

        p_0 = np.real(np.exp(-beta*sl.eigvalsh(Ham(0))))

        obs_entropy[q] = entropy(p_tilde) - entropy(p_rho, p_tilde) - entropy(p_0)

        diag_entropy[q] = entropy(p_rho) - entropy(p_0)


        p_ones = np.ones(len(p_rho))/len(p_rho)
        obs_entropy_traditional[q] = - entropy(p_rho, p_ones) + entropy(p_0, p_ones)



        Sobs_coarse[q] = entropy(p_tilde) - entropy(coarse_p_rho, coarse_p_prior) - entropy(p_0)
        Strad_coarse[q] = - entropy(coarse_p_rho, coarse_Id) + entropy(coarse_p0, coarse_Id0)

tau_th = (d_0 + DELTA) / (4 * np.pi * n * lambda2)


def del_H():
    aux = np.zeros([2 * int(n), 2 * int(n)], float)
    for i in range(int(n)):
        aux[i, i] = i / (int(n) - 1) * d(t)
    for j in range(int(n)):
        aux[int(n) + j, int(n) + j] = j / (int(n) - 1) * d(t)
    return aux


E2_matrix = np.zeros([2 * int(n), 2 * int(n)], float)
F_matrix = np.zeros([2 * int(n), 2 * int(n)], float)

aux = []

for t in [0, tau]:
    # Diagonalize Hamiltonian
    values, vectors = sl.eigh(Ham(t))

    # Build E2_matrix using broadcasting
    diffs = values[:, None] - values[None, :]
    E2_matrix = diffs ** 2
    np.fill_diagonal(E2_matrix, np.inf)  # avoid i=i

    # Build F_matrix using one matrix multiplication
    F_matrix = np.abs(vectors.conj().T @ del_H() @ vectors)
    # print(F_matrix)

    # Compute X
    X = DELTA * F_matrix / E2_matrix

    aux.append(X.max() / tau_th)

print(f'tau_ad/tau_th = {np.real(max(aux))}\n')

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints / tau_th, np.real(obs_entropy), 'k^', label=r'$\Delta S_{obs}$')
ax.loglog(taupoints / tau_th, np.real(diag_entropy), 'rv', label=r'$\Delta S_{diag}$')
ax.loglog(taupoints / tau_th, np.real(Dad_points), 'bx', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints / tau_th, np.real(obs_entropy_traditional), 'y.', label=r'$\Delta S_{obs, trad}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau/\tau_{rel}$')
# ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
ax.set_xlim(1e-2, 1e1)
ax.set_ylim(top=1e-4)
ax.set_box_aspect(0.85)
plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints / tau_th, np.real(Sobs_coarse), 'k^', label=r'$\Delta S_{obs}$')
ax.loglog(taupoints / tau_th, np.real(diag_entropy), 'rv', label=r'$\Delta S_{diag}$')
ax.loglog(taupoints / tau_th, np.real(Dad_points), 'bx', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints / tau_th, np.real(Strad_coarse), 'y.', label=r'$\Delta S_{obs, trad}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau/\tau_{rel}$')
# ax.set_title(r'Coarse-grained measurement', fontsize=15)
ax.set_xlim(1e-2, 1e1)
ax.set_ylim(top=1e-4)
ax.set_box_aspect(0.85)
plt.show()
