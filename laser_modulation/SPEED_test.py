# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:15:23 2024

@author: arjun
"""

from define_laser_lattice import *
from phase_space_manipulation import *
from tracking_functions import *

##### natural constants #####
c = const.c                       # speed of light
e_charge = const.e                # electron charge
m_e = const.m_e                   # electron mass in eV/c^2
Z0 = 376.73                       # impedance of free space in Ohm
epsilon_0 = const.epsilon_0       # vacuum permittivity

e_E = 1.492e9 * e_charge    # electron energy in J
e_gamma = e_E/m_e/c**2    # Lorentz factor
sigma_E = 7e-4

        
##### Simulation parameter #####
slicelength = 10e-6     # length of simulated bunch slice in m
tstep = 5e-12           # timestep in s
N_e = int(5e4)          # number of electrons

bunch_test = define_bunch(Test=True,E0=e_E)
bunch_init = define_bunch(Test=False,E0=e_E,dE=sigma_E,N=N_e,slicelength=slicelength)

##### defining Laser 1 #####
l1_wl = 400e-9   # wavelength of the laser
l1_sigx = 1.5e-3   # sigma width at the focus
l1_fwhm =45e-15  # pulse length 
l1_E = 3e-3 #1.5      # pulse energy

##### defining Laser 2 #####
l2_wl = 400e-9
l2_sigx = 1.0e-3
l2_fwhm = 45e-15
l2_E = 0 #3.0e-3

# Calculate the delay for the second seed pulse
p = [ 6.29880952e-04, -8.26071429e-02,  1.10083333e+01]

Delay_SecondPulse = lambda I : (p[0] * I**2 + p[1] * I + p[2] ) - 3

IC1 = 0                 # Current for the first chicane
delay_z = Delay_SecondPulse(IC1) * 1e-6    # Corresponding R56 value in microns

#%%
## First calculate energy modualtion amplitude and the optimum chicane currents
##### defining the magnetic configuration#####

#lattice = Lattice(filename='fieldfiles/M2_400_C2_300_R_200/M2_400_C2_300_R_200_R52mod_35A_input.txt', plot= 1)
lattice = Lattice(filename='input_files/U250.LTT', plot= 1)

l1 = Laser("input_files/Laser1.LSR")
l2 = Laser("input_files/Laser2.LSR")

#%%
elec_test , track_x, track_z = lsrmod_track(lattice,l2,bunch_test,tstep=tstep,zlim=6,disp_Progress=False,plot_track=True)
z , dE = calc_phasespace(elec_test,e_E,plot=True)
A22 = abs(min(dE))
print(A22)
print("R56 : ",(z[-2]-z[-1])/7e-4*1e6)
print("R52 z_diff : ",(z[-4]-z[-3])*1e9)
print("R51 z_diff : ",(z[-6]-z[-5])*1e9)
R52_l = (track_z.T[-3] - track_z.T[-4])/4e-5
print("R52 :" , (R52_l[2910] - R52_l[2320])*1e6)

#%%

elec = define_bunch(E0=e_E,dE=sigma_E,N=5e4,slicelength=10e-6)
print("\nTracking through the lattice...")
elec_M1 = lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep,get_R512=False)
z , dE = calc_phasespace(elec_M1,e_E,plot=True)

#%%
plt.figure()
wl = np.linspace(20e-9,250e-9,1001)
b = calc_bn(z,wl)                     #calculating bunching factor
plt.plot(wl,b)
plt.show()
