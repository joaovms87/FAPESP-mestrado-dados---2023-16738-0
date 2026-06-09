import matplotlib.pyplot as plt
import numpy as np

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


taupoints = np.logspace(-2, 4, 40)
l = len(taupoints)

DELTA = 1  # total variation of gamma1
beta = 1
h2 = 1
h3 = 1

h_0 = 1.1

obs_entropy = np.loadtxt(f'S_diag beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
diag_entropy = np.loadtxt(f'S_diag beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
Dad_points = np.loadtxt(f'D_ad beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
obs_entropy_traditional = np.loadtxt(f'S_obs_trad beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(obs_entropy), 'k^', label=r'$\Delta S_{obs}$')
ax.loglog(taupoints, np.real(diag_entropy), 'rv', label=r'$\Delta S_{diag}$')
ax.loglog(taupoints, np.real(obs_entropy_traditional), 'y.', label=r'$\Delta S_{obs, trad}$')
ax.loglog(taupoints, np.real(Dad_points), 'bx', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau \cdot J/\hbar$')
#ax.set_xlim(1e-1, 1e3)
ax.set_box_aspect(0.85)
plt.savefig('/Users/ospite/Desktop/joao_research/spins3 S_obs comparison.png', bbox_inches='tight', dpi=400)


print(abs(obs_entropy - Dad_points)/obs_entropy)
print()
print(abs(obs_entropy - diag_entropy)/obs_entropy)
print()
print(abs(obs_entropy - obs_entropy_traditional)/obs_entropy)
