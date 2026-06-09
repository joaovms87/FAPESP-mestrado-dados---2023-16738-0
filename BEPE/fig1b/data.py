import numpy as np
import scipy.linalg as sl
from scipy.optimize import fsolve
from scipy.stats import entropy


DELTA = 1  # total variation of gamma1
beta = 1
h2 = 1
h3 = 1


def h(t):  # protocol for gamma1 variation
    if t==0:
        return h_0
    else:
        return h_0 + DELTA/tau*t


def partition(b, eigenvalues):
    return sum(np.exp(-b*eigenvalues))


def rho_eq(t, b):
    eigenvalues, eigenvectors = sl.eigh(Ham(t))
    Z = partition(b, eigenvalues)
    p = 0
    for i in range(8):
        v = eigenvectors[:, i].copy()
        v /= np.linalg.norm(v)
        p += np.exp(-b*eigenvalues[i])*np.outer(v, v)

    return p/Z


def rho_ad(t):
    eigenvalues, eigenvectors = sl.eigh(Ham(t))
    eigenvalues0, eigenvectors0 = sl.eigh(Ham(0))
    Z = partition(beta, eigenvalues0)
    p = 0
    for i in range(8):
        v = eigenvectors[:, i].copy()
        v /= np.linalg.norm(v)
        p += np.exp(-beta*eigenvalues0[i])*np.outer(v, v)

    return p / Z


def Ham(t):
    h1 = h(t)
    H_t = np.zeros([8, 8], complex)
    H_t[0, 0], H_t[1, 1], H_t[2, 2], H_t[3, 3], H_t[4, 4], H_t[5, 5], H_t[6, 6], H_t[7, 7] = -h1-h2-h3, h1-h2-h3, -h1+h2-h3, -h1-h2+h3, -h1+h2+h3, h1-h2+h3, h1+h2-h3, h1+h2+h3
    H_t[0, 4], H_t[0, 5], H_t[0, 6] = -1, -1, -1
    H_t[1, 2], H_t[1, 3], H_t[1, 7] = -1, -1, -1
    H_t[2, 1], H_t[2, 3], H_t[2, 7] = -1, -1, -1
    H_t[3, 1], H_t[3, 2], H_t[3, 7] = -1, -1, -1
    H_t[4, 0], H_t[4, 5], H_t[4, 6] = -1, -1, -1
    H_t[5, 0], H_t[5, 4], H_t[5, 6] = -1, -1, -1
    H_t[6, 0], H_t[6, 4], H_t[6, 5] = -1, -1, -1
    H_t[7, 1], H_t[7, 2], H_t[7, 3] = -1, -1, -1
    return H_t


def D(p1, p2):  # relative entropy between states p1 and p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)
    return np.trace(prod1-prod2)


def f(p, t):  # array of functions at the right-hand-side of the simultaneous equations
    return - 1j * (Ham(t) @ p - p @ Ham(t))


# The next functions are for the adaptive Runge-Kutta method
def ratio(a, b):
    aux = (a-b)/30
    eps = np.linalg.norm(aux)
    return u*delta/eps

def new_u(u, p):
    u_new = u*p**0.25
    if u_new>2*u:
        return 2*u
    else:
        return u_new

def equation_rho(b):
    return np.trace((rho - rho_eq(tau, b)) @ H)
# Initial guess
initial_guess = beta


# list of the process durations that will be investigated
taupoints = np.logspace(-2, 4, 40)
l = len(taupoints)
# each entry corresponds to a process with tau in taupoints:
Dad_points = np.empty(l, complex)   # D[rho(tau)||rho_ad(tau)]
beta_tilde = np.empty(l, complex)
obs_entropy = np.empty(len(taupoints), complex)
diag_entropy = np.empty(len(taupoints), complex)
obs_entropy_traditional = np.empty(len(taupoints), complex)


delta = 1e-11  # required accuracy per unit time

for h_0 in [1.1]:
    # array of rho elements at time 0; necessary to apply adaptive Runge-Kutta method
    rho0 = rho_eq(0, beta)
    for j, tau in enumerate(taupoints):
        t = 0
        N = 100
        u = tau/N  # Initial step size
        print(tau)

        # Solve system of differential equations by the adaptive Runge-Kutta method
        r = rho0.copy()
        while t < tau:
            # two steps of size u
            k1 = u * f(r, t)
            k2 = u * f(r + 0.5 * k1, t + 0.5 * u)
            k3 = u * f(r + 0.5 * k2, t + 0.5 * u)
            k4 = u * f(r + k3, t + u)
            r1 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            k1 = u * f(r1, t + u)
            k2 = u * f(r1 + 0.5 * k1, t + 1.5 * u)
            k3 = u * f(r1 + 0.5 * k2, t + 1.5 * u)
            k4 = u * f(r1 + k3, t + 2 * u)
            r1 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # one step of size 2h
            k1 = 2 * u * f(r, t)
            k2 = 2 * u * f(r + 0.5 * k1, t + u)
            k3 = 2 * u * f(r + 0.5 * k2, t + u)
            k4 = 2 * u * f(r + k3, t + 2 * u)
            r2 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            p = ratio(r1, r2)
            if p >= 1:  # precision greater than required
                r = r1  # keep the most precise result
                t = t + 2 * u  # next t
            # if p>=1 is not satisfied, repeat everything without going to next t
            u = new_u(u, p)  # changes h regardless of the value of p

        rho = r


        r_ad = rho_ad(tau)
        H = Ham(tau)

        # compute \tilde\beta
        beta_tilde[j] = fsolve(equation_rho, initial_guess)[0]
        r_tilde = rho_eq(tau, beta_tilde[j])

        evals, evecs = sl.eigh(H)
        p_tilde = np.real(np.exp(-beta_tilde[j] * evals))
        p_tilde = p_tilde / sum(p_tilde)
        p_rho = np.real(np.diag(evecs.conj().T @ rho @ evecs))

        # compute D[rho(tau)||rho_ad(tau)] and save it
        Dad_points[j] = D(rho, rho_ad(tau))

        p_0 = np.real(np.exp(-beta * sl.eigvalsh(Ham(0))))

        obs_entropy[j] = entropy(p_tilde) - entropy(p_rho, p_tilde) - entropy(p_0)

        diag_entropy[j] = entropy(p_rho) - entropy(p_0)

        p_ones = np.ones(len(p_rho)) / len(p_rho)
        obs_entropy_traditional[j] = - entropy(p_rho, p_ones) + entropy(p_0, p_ones)

        # save everything
        np.savetxt(f'D_ad beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', Dad_points)
        np.savetxt(f'S_obs beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', obs_entropy)
        np.savetxt(f'S_diag beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', diag_entropy)
        np.savetxt(f'S_obs_trad beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', obs_entropy_traditional)
        np.savetxt(f'beta_tilde beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', beta_tilde)
