#!/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/bin/python3
import sys
sys.path.append('/home/lpyras/utah/mute_fork/mute/')
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as scii
import datetime
import mute
import mute.constants as mtc
import mute.surface as mts
import mute.underground as mtu
import argparse
import os
import pickle

argparser = argparse.ArgumentParser()
argparser.add_argument('--day', type=int, default=4)
argparser.add_argument('--year', type=int, default=2012)
argparser.add_argument('--out_dir', type=str, default="/data/user/lpyras/mute_output/")
argparser.add_argument('--hadronic_model', type=str, default="SIBYLL2.3c")
argparser.add_argument('--primary_model', type=str, default="GSF")
args = argparser.parse_args()

day = args.day + 1
year = args.year
cr_mod = args.primary_model
had_mod = args.hadronic_model


def get_month(day_num, year):
    date = datetime.datetime.strptime(f"{year}-{int(day_num):03d}", "%Y-%j")
    res = date.strftime("%B")
    return res

vertical_depth = 2. * (0.93/0.997) 
print(f"Vertical depth: {vertical_depth:.2f} km w.e.")

mtc.set_verbose(1)
mtc.set_output(True)
mtc.set_lab("IceCube")
mtc.set_density(0.93)
mtc.set_overburden("flat")
mtc.set_vertical_depth(vertical_depth)
mtc.set_medium("ice")
mtc.set_n_muon(10000)
mtc.set_directory(args.out_dir)

month = get_month(day, args.year)

s_fluxes = np.zeros((len(mtc.ENERGIES), len(mtc.ANGLES_FOR_S_FLUXES)))
input_file = f"/data/user/aalves/calc_fluxes_pkl/fluxes_per_zenith_{year}_{day}.pkl"
with open(input_file, "rb") as f:
    data = pickle.load(f)
for i_zen, zen in enumerate(data[cr_mod][had_mod]['muon_flux'].keys()):
    MCeq_energies = data[cr_mod][had_mod]['E_grid'][zen]
    MCeq_flux = data[cr_mod][had_mod]['muon_flux'][zen]
    s_fluxes[:, i_zen] = MCeq_flux * 1e-3
print('s_fluxes', s_fluxes.shape)
#s_fluxes   = mts.calc_s_fluxes(primary_model=args.primary_model, interaction_model=args.hadronic_model, atmosphere = "AIRS", location = "SouthPole", month = month, force=True, year=args.year, day=day)
s_tot_flux = mts.calc_s_tot_flux(s_fluxes = s_fluxes, force=True)
print('s_tot_flux', s_tot_flux)
u_fluxes = mtu.calc_u_fluxes(s_fluxes = s_fluxes, full_tensor=True, output=True, output_file_name=f"{args.out_dir}underground/underground_fluxes_{had_mod}_{cr_mod}_{year}_{day}.pkl", force=True)
print('u_fluxes', u_fluxes.shape) #(28, 91, 10) slant depth, energy bins, surface zenith angles. 
#store constants._SLANT_DEPTHS[x], constants.ENERGIES[u], constants.ANGLES_FOR_S_FLUXES[j], u_fluxes[x, u, j],
u_tot_flux = mtu.calc_u_tot_flux(u_fluxes = u_fluxes, force=True)
print('u_tot_flux', u_tot_flux)
