# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:23:24 2022

@author: arjun
"""


import os
#Changing the working directory to the source directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.io
import scipy.interpolate
from SPEED_functions import *

##### natural constants #####
c = const.c                       # speed of light
e_charge = const.e                # electron charge
m_e = const.m_e                   # electron mass in eV/c^2
Z0 = 376.73                       # impedance of free space in Ohm
epsilon_0 = const.epsilon_0       # vacuum permittivity

e_E = 1.5e9 * e_charge    # electron energy in J
e_gamma = e_E/m_e/c**2    # Lorentz factor
sigma_E = 7e-4

        
##### Simulation parameter #####
slicelength = 20e-6     # length of simulated bunch slice in m
tstep = 5e-12           # timestep in s
N_e = int(5e4)          # number of electrons

bunch_test = define_bunch(Test=True,E0=e_E)
bunch_init = define_bunch(Test=False,E0=e_E,dE=sigma_E,N=N_e,slicelength=slicelength)
elec = np.copy(bunch_init)

##### defining Laser 1 #####
l1_wl = 800e-9   # wavelength of the laser
l1_sigx = 1e-3   # sigma width at the focus
l1_fwhm =45e-15  # pulse length 
l1_E = 4e-3      # pulse energy

##### defining Laser 2 #####
l2_wl= 400e-9
l2_sigx= 1e-3
l2_fwhm=45e-15
l2_E= 4e-3

Delay_SecondPulse = lambda I : (8.57242860e-04* I**2 + -1.17140062e-01* I + 1.61857761e+01)/2   # Calculate the delay for the second seed pulse


IC1 = 750                    # Current for the first chicane
delay_z = Delay_SecondPulse(IC1) * 1e-6    # Corresponding R56 value in microns

## First calculate energy modualtion amplitude and the optimum chicane currents
##### defining the magnetic configuration#####

lattice = Lattice(E0= 1500, l1= l1_wl, l2= l2_wl, h=5, c1= IC1 , c2= 600, plot= 1)

l1= Laser(wl=l1_wl,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=l1_E,focus=1.125,M2=1,pulsed=1,phi=0e10)
l2= Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.3125,X0=0.0e-3,Z_offset=delay_z,M2=1,pulsed=1,phi=0e10)

#### Test Tracking through Modulators
elec_test= lsrmod_track(lattice,l1,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=True)
A11=(max(dE))

elec_test= lsrmod_track(lattice,l2,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=True)
A22=(max(dE))
print("A1= ",A11,"\t A2= ",A22)

R56_1_opt , R56_2_opt = calc_R56(A11, A22, m=3)

#%%
###### Calculate the curresponding chicane currents and define a new lattice object
######################## 
#r56 = (0.869e-3 *I**2)+ (-6.236e-3* I) + 2.353 ;      % (1st chicane) Current to R56 relation in SPEED mode (Benedikt)
#r56 = (0.203e-3 *I**2)+ (-35.5e-3* I) + 4.01  ;       % (2nd chicane) Current to R56 relation in SPEED mode (Benedikt) 
########################

IC1 = max(np.roots([0.869e-3 , -6.236e-3 , -R56_1_opt*1e6 + 2.353]))
IC2 = max(np.roots([0.203e-3 , -35.5e-3 , -R56_2_opt*1e6 + 4.01]))

delay_z = Delay_SecondPulse(IC1) * 1e-6 

lattice = Lattice(E0= 1500, l1= l1_wl, l2= l2_wl, h=5, c1= IC1 , c2= IC2 , plot= 1)
l2= Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.3125,X0=0.0e-3,Z_offset=delay_z,M2=1,pulsed=1,phi=0e10)

elec_test= lsrmod_track(lattice,l1,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=True)
A11=(max(dE))

elec_test= lsrmod_track(lattice,l2,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=True)
A22=(max(dE))
print("A1= ",A11,"\t A2= ",A22)

R56_1_opt , R56_2_opt = calc_R56(A11, A22, m=3)

#%%

elec = define_bunch(E0=e_E,dE=sigma_E,N=1e5,slicelength=slicelength)
print("\nTracking through the lattice...")
elec_M1= lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep)
z,dE=calc_phasespace(elec_M1,e_E,plot=True)

plt.figure()
wl = np.linspace(20e-9,250e-9,1001)
b = calc_bn(z,wl)                     #calculating bunching factor
plt.plot(wl,b)
#plot_slice(z, wl, n_slice=100)

#%%
########## Trying out different R56_2 values ###########
elec = define_bunch(E0=e_E,dE=sigma_E,N=1e4,slicelength=slicelength)
IC2_list = np.linspace(IC2-50,IC2+50,11)
bmax = []
for IC2 in IC2_list:
    lattice = Lattice(E0= 1500, l1= l1_wl, l2= l2_wl, h=5, c1= IC1 , c2= IC2 , plot= 0)
    elec_M1= lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep)
    z,dE=calc_phasespace(elec_M1,e_E,plot=False)
    b = calc_bn(z,wl)
    bmax.append(max(b))                    
    plt.plot(wl,b)

plt.figure()
plt.plot(IC2_list, bmax)
