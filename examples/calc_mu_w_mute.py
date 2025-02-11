import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import trapz
from scipy import interpolate
import sys

_E_BINS = np.logspace(1.9, 14, 122)  # Bin edges in MeV
_E_WIDTHS = np.diff(_E_BINS)  # Bin widths
ENERGIES = np.sqrt(_E_BINS[1:] * _E_BINS[:-1])  # Bin centers

print(ENERGIES)
print(ENERGIES.shape)

colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

cr_mod = 'GSF'
had_mod = 'SIBYLL2.3c'
year = 2012
day = 1

surface_flux_file = f"/data/user/aalves/calc_fluxes/fluxes_per_zenith_{year}_{day}.pkl"
with open(surface_flux_file, "rb") as f:
    sfd = pickle.load(f)
print(sfd.keys())
print(sfd['E_grid'])
print(len(sfd['E_grid']))


survival_file='/data/user/lpyras/mute_output/survival_probabilities/ice_0.93_100_Survival_Probabilities.txt'
file = open(survival_file, "r")
n_lines = len(file.read().splitlines())
file.close()

if n_lines != 121 * 10 * 121:
    raise ValueError(f"Survival probabilities file does not match the expected number of lines: {n_lines} != {121 * 10 * 121}")

survival_probability_tensor = np.reshape(
	np.loadtxt(survival_file)[:, 3],
	(121, 10, 121)
)
new_height = np.linspace(0, 70000, 1000)
new_X_density = np.logspace(0, 3, 1000)

spl_height_prod = np.zeros((len(sfd['zenith_center']),len(new_height)))
spl_density_prod = np.zeros((len(sfd['zenith_center']),len(new_X_density)))
fig2, ax3 = plt.subplots(1,1,figsize=(12,6))
for j, E_threshold in enumerate([1, 500, 1000]):
	fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
	for i, zen_key in enumerate(sfd['zenith_center']):
		zen = f"{np.rad2deg(zen_key):.2f}"
		X_density = sfd[cr_mod][had_mod]['x_density'][zen]
		height = sfd[cr_mod][had_mod]['h_height'][zen]
		height_center = (height[1:] + height[:-1])/2
		X_density_column = X_density * np.cos(zen_key)
		X_density_column_center = (X_density_column[1:] + X_density_column[:-1])/2
		bin_width_X = np.diff(X_density)

		flux = sfd[cr_mod][had_mod]['muon_flux_per_X'][zen]
		print(f)
	
		print(f"flux shape: {flux.shape}")
		energy = sfd[cr_mod][had_mod]['E_grid'][zen]

		# Construct a matrix of energy bin widths
		#widths = np.repeat(_E_WIDTHS, flux.shape[1]).reshape((len(energy), flux.shape[1])) #121x121
		# Calculate the underground flux tensor 
		#u_fluxes_from_convolution = (survival_probability_tensor.dot((s_fluxes.T * constants._E_WIDTHS).T) / widths)

		#surv_prob = survival_probabilities[:,i,:] # surface energy, zenith angle, underground energy
		N_mu = []
		energy_mask = energy > E_threshold
        # atmospheric layers
		for idx in range(flux.shape[0]):
			underground_flux = np.dot(survival_probability_tensor[:, i, :], flux[idx, :])
			sum_flux = trapz(underground_flux[energy_mask], energy[energy_mask])
			N_mu.append(sum_flux)

		N_mu = np.array(N_mu)
		N_mu_per_slice = N_mu[1:] - N_mu[:-1]
		N_mu_center = (N_mu[1:] + N_mu[:-1])/2
		N_mu_prod_X =  N_mu_per_slice * X_density_column_center * bin_width_X

		spline_density = interpolate.UnivariateSpline(X_density_column_center, N_mu_prod_X, s=0,ext=1)
		spline_height = interpolate.UnivariateSpline(np.flip(height_center), np.flip(N_mu_prod_X), s=0, ext=1)

		spl_height_prod[i,:] = spline_height(new_height)
		spl_density_prod[i,:] = spline_density(new_X_density)

		ax1.plot(X_density_column_center, N_mu_prod_X,marker='x',ls='',color=colors[i],label=np.round(np.cos(zen_key),2))
		ax2.plot(height_center, N_mu_prod_X,marker='x',ls='',color=colors[i],label=np.round(np.cos(zen_key),2))

		#Normalize 
		#ax1.plot(X_density_column_center, N_mu_prod_X/(trapz(N_mu_prod_X,X_density_column_center)),marker='o',ls='',color=colors[i],label=np.round(np.cos(zen_key),2))
		#ax2.plot(height_center, N_mu_prod_X/(trapz(N_mu_prod_X,height_center)), marker='o',ls='',color=colors[i],label=np.round(np.cos(zen_key),2))
		ax1.legend()
		ax2.legend()
		ax1.plot(new_X_density, spl_density_prod[i,:], color=colors[i])
		ax2.plot(new_height, spl_height_prod[i,:], color=colors[i])
		if np.round(np.cos(zen_key),2) == 0.95:
			ax3.plot(X_density_column_center, N_mu_prod_X/(trapz(N_mu_prod_X ,X_density_column_center)),marker='o',ls='',color=colors[j],label=f'{np.round(np.cos(zen_key),2)} {E_threshold} GeV')

			#ax3.plot(X_density_column_center, N_mu_prod_X,marker='x',ls='',color=colors[j],label=np.round(np.cos(zen_key),2))
			#ax3.plot(new_X_density, spl_density_prod[i,:], color=colors[j],label=f"{E_threshold} GeV")
			ax3.legend()

	ax1.plot(new_X_density, np.sum(spl_density_prod,axis=0)*0.1,color='k')
	ax2.plot(new_height, np.sum(spl_height_prod,axis=0)*0.1, color='k')
	ax1.grid()
	ax2.grid()
	ax1.set_ylabel(rf"X $d N_\mu / d X (E_\mu$ > {E_threshold} GeV)")
	ax1.set_xlabel(r'X$_\mathrm{density}$ [g/cm$^2$]')
	ax1.set_xscale('log')
	#ax1.set_xlim(1e-1,5e2)
	ax2.set_ylabel(rf"X $d N_\mu / d X (E_\mu$ > {E_threshold} GeV)")
	ax2.set_xlabel(r'Height [m]')

	ax3.set_ylabel(rf"X $d N_\mu / d X (E_\mu$ > E$_\mathrm{{label}}$ GeV)")
	ax3.set_xlabel(r'X$_\mathrm{density}$ [g/cm$^2$]')
	ax3.set_xscale('log')
	ax3.grid()


	fig1.tight_layout()
	fig1.savefig(f"plots/N_mu_underground_energy_{E_threshold}GeV_{year}_{day}.pdf", dpi=300, bbox_inches='tight')
fig2.tight_layout()
fig2.savefig(f"plots/N_mu_underground_095_energy_{year}_{day}.pdf", dpi=300, bbox_inches='tight')
