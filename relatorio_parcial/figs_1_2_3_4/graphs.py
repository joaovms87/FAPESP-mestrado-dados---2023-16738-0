import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import cmath
from numpy.linalg import eigh

plt.rcParams['errorbar.capsize'] = 3
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 20
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.weight'] = 'bold'


#################### 3 SPINS 1 VARIA  ######################


beta = 1
h0 = 1.1
h2 = 1
h3 = 1
DELTA = 1  # total variation of h

Wpoints = np.loadtxt(r'D:\PythonScripts\3spins\1 variando\RK\W beta=1 h_0=1.1 h2=1 h3=1 DELTA=1', complex)
WADpoints = np.loadtxt(r'D:\PythonScripts\3spins\1 variando\RK\W_ad beta=1 h_0=1.1 h2=1 h3=1 DELTA=1', complex)
Dpoints = np.loadtxt(r'D:\PythonScripts\3spins\1 variando\RK\D beta=1 h_0=1.1 h2=1 h3=1 DELTA=1', complex)
D2points = np.loadtxt(r'D:\PythonScripts\3spins\1 variando\RK\D2 beta=1 h_0=1.1 h2=1 h3=1 DELTA=1', complex)

Wex = np.real(Wpoints - WADpoints)
taupoints = np.logspace(-2, 4, 100)


# D vs Wex
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Wex, 'r.', label=r'$W_{ex}(\tau)$')
ax.loglog(taupoints, Dpoints, 'k.', label=r'$D[\rho(\tau)||\rho_{eq}(\tau)]$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('$W_{ex}J^{-1}$ ou $D$')
ax.set_title(r'3 spins: quantificadores em função de $\tau$', pad=10)
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
ax.set_box_aspect(0.7)
plt.savefig(f'D:\MESTRADO\EQM/3spins Wex vs D.png', bbox_inches='tight')


# Wex vs Dad
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Wex, 'r.', label=r'$W_{ex}(\tau)$')
ax.loglog(taupoints, D2points, 'b.', label=r'$D_{ad}(\tau)$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('$W_{ex}J^{-1}$ or $D_{ad}$')
ax.set_title(r'3-spin chain: $\tau$-dependence', pad=10)
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
ax.set_box_aspect(0.7)
plt.text(0.9e2, 0.8e-5, r'$\sim \tau^{-2}$', fontsize=28)
plt.savefig(f'D:\MESTRADO\EQM/3spins Wex vs Dad.png', bbox_inches='tight')




################################ LZ ####################################

# Wex vs D vai ser com B0 = -2
# Wex vs Dad vai ser com B0 = -2
# Comparação APT vai ser com B0 = -1

B0 = -2
beta = 1

Wpoints = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\W; B0={B0}; beta={beta}', complex)
WADpoints = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\W_ad; B0={B0}; beta={beta}', complex)
Dpoints = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\D; B0={B0}; beta={beta}', complex)
D2points = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\D2; B0={B0}; beta={beta}', complex)

Wex = np.real(Wpoints - WADpoints)
taupoints = np.logspace(-2, 4, 100)


# D vs Wex
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Wex, 'r.', label=r'$W_{ex}(\tau)$')
ax.loglog(taupoints, Dpoints, 'k.', label=r'$D[\rho(\tau)||\rho_{eq}(\tau)]$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('$W_{ex}J^{-1}$ ou $D$')
ax.set_title(r'LZ: quantificadores em função de $\tau$', pad=10)
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
ax.set_box_aspect(0.7)
plt.savefig(f'D:\MESTRADO\EQM/LZ Wex vs D B0={B0}.png', bbox_inches='tight')


# Wex vs Dad
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Wex, 'r.', label=r'$W_{ex}(\tau)$')
ax.loglog(taupoints, D2points, 'b.', label=r'$D_{ad}(\tau)$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('$W_{ex}J^{-1}$ or $D_{ad}$')
ax.set_title(r'LZ: $\tau$-dependence', pad=10)
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
ax.set_box_aspect(0.7)
plt.text(2e2, 1e-4, r'$\sim \tau^{-2}$', fontsize=28)
plt.savefig(f'D:\MESTRADO\EQM/LZ Wex vs Dad B0={B0}.png', bbox_inches='tight')



# COMPARAÇÃO APT
beta = 1
DELTA = 2

def B(t):  # protocol for B variation
    if t > 0:
        return B0 + DELTA/tau*t
    else:
        return B0


def theta(B):  # a useful parameter dependent on B
    return 0.5*np.arctan2(1, B)


def E(B):  # absolute value of the eigen-energy of the system
    return np.sqrt(B**2 + 1)

def ratio_M_E(t):
    return -DELTA/tau/4/E(B(t))**3


def integrand(t):
    return np.sqrt(1+B(t)**2)


def phi(m, n, t):
    return (m-n)*2*quad(integrand, 0, t)[0]


def C1(m, n, t):
    return 1j*(ratio_M_E(t) - cmath.exp(1j*phi(m, n, t))*ratio_M_E(0))


def avg_abs_squared_C1(m, n, t):
    return abs(ratio_M_E(t))**2 + abs(ratio_M_E(0))**2 - 2*avg_cos*abs(ratio_M_E(t)*ratio_M_E(0))



for B0 in [-2, -1]:

    Wpoints = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\W; B0={B0}; beta={beta}', complex)
    WADpoints = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\W_ad; B0={B0}; beta={beta}', complex)
    Dpoints = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\D; B0={B0}; beta={beta}', complex)
    D2points = np.loadtxt(f'D:\PythonScripts\Landau_Zener\AdapiveRK\D2; B0={B0}; beta={beta}', complex)

    Wex = np.real(Wpoints - WADpoints)
    taupoints = np.logspace(-2, 4, 100)

    WexNUM = np.real(Wpoints-WADpoints)
    # list of the process durations that will be investigated
    n = 100
    taupoints = np.logspace(-2, 4, n)
    WexAPT = np.empty(n, float)
    WexAPT_avg = np.empty(n, float)
    D2APT = np.empty(n, float)
    D2APT_avg = np.empty(n, float)


    phi_list = np.empty(int(1e4), float)
    sample_points = np.linspace(1e-2, 1e4, int(1e4))
    for k, s in enumerate(sample_points):
        tau = s
        phi_list[k] = phi(2, 1, s)
    cosine = np.cos(phi_list)
    avg_cos = np.average(cosine)


    for i, tau in enumerate(taupoints):
        C = C1(2, 1, tau)
        Wex = 2*E(B(tau))*np.tanh(beta*E(B0))*(abs(C))**2
        D = 2*beta*E(B0)*np.tanh(beta*E(B0))*(abs(C))**2
        WexAPT[i] = Wex
        D2APT[i] = D

        C2_avg = avg_abs_squared_C1(2, 1, tau)
        Wex_avg = 2*E(B(tau))*np.tanh(beta*E(B0))*C2_avg
        WexAPT_avg[i] = Wex_avg
        D_avg = 2*beta*E(B0)*np.tanh(beta*E(B0))*C2_avg
        D2APT_avg[i] = D_avg



    # Wex comparison: 10^(-2) - 10^4

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, WexNUM, 'r.', label=r'$W_{ex}(\tau)$ num.')
    ax.loglog(taupoints, WexAPT, 'k.', label=r'$W_{ex}(\tau)$ APT')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel('$W_{ex} J^{-1}$')
    ax.set_title(r'LZ: $W_{ex}(\tau)$ numerical vs APT', pad=10)
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_box_aspect(0.7)
    plt.savefig(f'D:\MESTRADO\EQM\Wex comparação APT B0={B0}.png', bbox_inches='tight')

    """
    # Wex comparison: averaging the phase
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, WexNUM, 'r.', label=r'Num. solution')
    ax.loglog(taupoints, WexAPT_avg, 'k-', label=r'$2^{nd}$ order APT (phase-averaged)')
    ax.legend(fancybox=False, edgecolor='black', loc='upper right')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'$\tau$-dependence of the excess work')
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_box_aspect(1)
    plt.savefig(f'D:\MESTRADO\Comparação_Num_vs_APT\LZ\B0={B0}/log(Wex)_log(tau) phase-averaged,B0={B0},beta={beta}.png', bbox_inches='tight')
    """


    # D_ad comparison: 10^(-2) - 10^4

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, D2points, 'r.', label=r'$D_{ad}(\tau)$ num.')
    ax.loglog(taupoints, D2APT, 'k.', label=r'$D_{ad}(\tau)$ APT')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel(r'$D_{ad}$')
    ax.set_title(r'LZ: $D_{ad}(\tau)$ numerical vs APT', pad=10)
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_box_aspect(0.7)
    plt.savefig(f'D:\MESTRADO\EQM/Dad comparação APT B0={B0}.png', bbox_inches='tight')
