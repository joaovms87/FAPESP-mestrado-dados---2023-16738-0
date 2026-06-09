import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat


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
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



taupoints = np.logspace(-1, 3, 12)
n_points = [12]
beta = 0.1
J = 1.0
hx = 1.1
hz0, Delta_h = 0.1, 0.4
h1, hL = 0.25, -0.25
h_0 = hz0
DELTA = Delta_h



W_points = np.real(np.loadtxt(f'W.txt', complex))[2, :]
Wad_points = np.real(np.loadtxt(f'W_ad.txt', complex))[2, :]
Wex_points = W_points - Wad_points



for i, tau in enumerate(taupoints):
    if tau > 10:
        break

fit1, cov1 = np.polyfit(np.log(taupoints[i:]), np.log(Wex_points[i:]), deg=1, cov=True)
print(f'Wex fit...')
print(ufloat(fit1[0], np.sqrt(cov1[0, 0])))
W_fit1 = taupoints**fit1[0] * np.exp(fit1[1])


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Wex_points), 'r.', label=r'$W_{ex}$')
ax.loglog(taupoints, W_fit1, 'k--', label=r'Fit for $\tau > 10$')
#ax.loglog(taupoints, np.real(Wex_points[1,:]), 'k.', label=f'$L = {n_points[1]}$')
#ax.loglog(taupoints, np.real(Wex_points[2,:]), 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Wex_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$J^{-1}W_{ex}$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1/spins12_Wex_fit.png', dpi=400, bbox_inches='tight')