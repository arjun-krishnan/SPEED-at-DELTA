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
from scipy.integrate import simpson
from scipy import interpolate
from SPEED_functions import *
from tqdm import tqdm
import h5py

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
slicelength = 20e-6     # length of simulated bunch slice in m
tstep = 5e-12           # timestep in s
N_e = int(5e4)          # number of electrons

bunch_test = define_bunch(Test=True,E0=e_E)
bunch_init = define_bunch(Test=False,E0=e_E,dE=sigma_E,N=N_e,slicelength=slicelength)
elec = np.copy(bunch_init)

##### defining Laser 1 #####
l1_wl = 800e-9   # wavelength of the laser
l1_sigx = 0.8e-3   # sigma width at the focus
l1_fwhm =45e-15  # pulse length 
l1_E = 2.6e-3      # pulse energy

##### defining Laser 2 #####
l2_wl = 400e-9
l2_sigx = 0.7e-3
l2_fwhm = 45e-15
l2_E = 2.5e-3

# Calculate the delay for the second seed pulse
p = [ 6.29880952e-04, -8.26071429e-02,  1.10083333e+01]

Delay_SecondPulse = lambda I : (p[0] * I**2 + p[1] * I + p[2] ) - 3

IC1 = 800                  # Current for the first chicane
delay_z = Delay_SecondPulse(IC1) * 1e-6    # Corresponding R56 value in microns

## First calculate energy modualtion amplitude and the optimum chicane currents
##### defining the magnetic configuration#####

lattice = Lattice(E0= 1492, l1= l1_wl, l2= l2_wl, h=5, c1= IC1 , c2= 635, plot= 1)
#lattice = Lattice(E0= 1492, l1= l1_wl, l2= 0, h=0, c1= 0 , c2= 0, plot= 1)

l1 = Laser(wl=l1_wl,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=l1_E,focus=1.125,M2=1,pulsed=1,phi=0e10)
l2 = Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.125,X0=0.0e-3,Z_offset=delay_z ,M2=1,pulsed=1,phi=0e10)

#### Test Tracking through Modulators
elec_test = lsrmod_track(lattice,l1,bunch_test,tstep=tstep,disp_Progress=False)
z , dE = calc_phasespace(elec_test,e_E,plot=True)
A11 = abs(min(dE))

elec_test = lsrmod_track(lattice,l2,bunch_test,tstep=tstep,disp_Progress=False)
z , dE = calc_phasespace(elec_test,e_E,plot=True)
A22 = abs(min(dE))
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

IC1, IC2 = 470, 610         # Comment these out if you dont want to give manual entries for C1 and C2

delay_z = Delay_SecondPulse(IC1) * 1e-6 

lattice = Lattice(E0= 1492, l1= l1_wl, l2= l2_wl, h=5, c1= IC1 , c2= IC2 , plot= 1)

l2 = Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.3125,X0=0.0e-3,Z_offset=delay_z,M2=1,pulsed=1,phi=0e10)

# elec_test = lsrmod_track(lattice,l1,bunch_test,tstep=tstep,disp_Progress=False)
# z,dE = calc_phasespace(elec_test,e_E,plot=True)
# A11 = (max(dE))

# elec_test = lsrmod_track(lattice,l2,bunch_test,tstep=tstep,disp_Progress=False)
# z , dE = calc_phasespace(elec_test,e_E,plot=True)
# A22 = (max(dE))
# print("A1= ", A11 ,"\t A2= ", A22)

# R56_1_opt , R56_2_opt = calc_R56(A11, A22, m=4)

#%% 

elec = define_bunch(E0=e_E,dE=sigma_E,N=5e5,slicelength=100e-6)
print("\nTracking through the lattice...")
elec_M1 = lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep)
z , dE = calc_phasespace(elec_M1,e_E,plot=True)

#%%
plt.figure()
wl = np.linspace(20e-9,250e-9,1001)
b = calc_bn(z,wl)                     #calculating bunching factor
plt.plot(wl,b)

wl_h      = 160e-9
slice_len = 800e-9
z_slice , b_slice = plot_slice(z,[wl_h], slice_len)
z_slice , b_slice = z_slice[1:-2] , b_slice[1:-2] 

#%% Calculating the temporal power profile of the radiation

beam_cur = 5e-3     # 5 mA
rev_t    = 115.2/c  # revolution time
bunch_Q  = beam_cur * rev_t
Ne_bunch = bunch_Q / e_charge

bunch_fwhm = 80e-12 *c                    # bunch length 80 ps in meters
bunch_sig  = bunch_fwhm / 2.3548
z_bunch    = np.arange(-1.5*bunch_fwhm, 1.5*bunch_fwhm , slice_len)
t_bunch    = z_bunch / c
t_slice    = slice_len / c
gaus       = (1/bunch_sig/np.sqrt(2*np.pi)) * np.exp(-z_bunch**2/2/bunch_sig**2)
bunch_dens = Ne_bunch * gaus

f_dens     = interpolate.interp1d(z_bunch, bunch_dens)


#Ne_slice   = f_dens(z_slice) * slice_len
#P = b_slice**2 * Ne_slice**2  # * P0 (power emitted from a single electron)

Ne_bunch_full = f_dens(z_bunch) * slice_len
bn_incoherent = 1/np.sqrt(Ne_bunch_full)
#np.mean(b_slice[0:10])
P_incoherent  = bn_incoherent**2 * Ne_bunch_full**2
plt.plot(t_bunch * 1e15 , P_incoherent / max(P_incoherent) , '-g' , label = "incoherent" )
E_incoherent = simpson(P_incoherent, dx = t_slice)


N_pad = len(z_bunch) - len(z_slice)
bn_left , bn_right = np.zeros(int(N_pad/2)) , np.zeros(int(N_pad/2))
if N_pad % 2 == 1:
    bn_right = bn_incoherent * np.ones(int(N_pad/2)+1)
bn_coherent = np.concatenate((bn_left, b_slice - np.mean(b_slice[0:10]) , bn_right)) 
bn_coherent += bn_incoherent 
P_coherent  = bn_coherent**2 * Ne_bunch_full**2
plt.plot(t_bunch * 1e15 , P_coherent / max(P_incoherent) , '-r' , label = "coherent", alpha = 0.8)
E_coherent = simpson(P_coherent, dx = t_slice)

plt.xlim(-150 , 150)
plt.xlabel("t (fs)")
plt.ylabel("Power (arb. u.)")
plt.legend()
plt.grid()

print("Integrated power: \n Incoherent \t : ", E_incoherent ,"\n Coherent \t\t : ", E_coherent)
print("Coherent to incoherent ratio :" , E_coherent/E_incoherent)

#%%
########## Exporting a particle distribution file for Genesis #########

elec_genesis = np.copy(elec_M1)
elec_genesis[2] = elec_genesis[2] - np.mean(elec_genesis[2])

ref_wl= wl_h
x = elec_genesis[0]
y = elec_genesis[1]
px = elec_genesis[3] / m_e / c
py = elec_genesis[4] / m_e / c
t = (elec_genesis[2] - min(elec_genesis[2])) / ref_wl * 2*np.pi
gamma = np.sqrt(m_e**2 * c**4 + elec_genesis[5]**2 * c**2) / m_e / c**2

plt.figure()
plt.plot(t,gamma,',r')

N_slice = round(max(t)/2/np.pi)

f = h5py.File('../Genesis/beam_in.par.h5', 'w')
f.create_dataset('beamletsize',data= 1)
f.create_dataset('one4one',data= 1)
f.create_dataset('refposition',data= 0.0)
f.create_dataset('slicecount',data= N_slice)
f.create_dataset('slicelength',data= ref_wl)
f.create_dataset('slicespacing',data= ref_wl)

prefix = "slice000"
for i in range(N_slice):
    ii = (t >= i*2*np.pi) & (t < (i+1)*2*np.pi)
    data_n = "{:03n}".format(i+1)
    N_par = len(gamma[ii])
    
    g = f.create_group(prefix+data_n)
    g.create_dataset('gamma',data= gamma[ii])
    g.create_dataset('current',data= N_par*e_charge/(ref_wl/c))
    g.create_dataset('x',data= x[ii])
    g.create_dataset('y',data= y[ii])
    g.create_dataset('px',data= px[ii])
    g.create_dataset('py',data= py[ii])
    
    tau = t[ii]- i*2*np.pi
    g.create_dataset('theta',data= tau)

f.close()

#%%
############ Plot a bunching heatmap for different chicane currents ############

harm = np.arange(3,21,2)
wl_h = 800e-9 / harm 
#[800e-9/3 , 800e-9/5 , 800e-9/7 , 800e-9/9, 800e-9/11, 800e-9/13] 

elec =  define_bunch(E0=e_E,dE=sigma_E,N=1e4,slicelength = 20e-6)

C1I = np.linspace(200, 800, 15)
C2I = np.linspace(200, 800, 15)

#wls = [np.linspace(wl_h[0]-5e-9, wl_h[0]+5e-9, 101) , np.linspace(wl_h[1]-5e-9, wl_h[1]+5e-9, 101) , np.linspace(wl_h[2]-5e-9, wl_h[2]+5e-9, 101)]
l1 = Laser(wl=l1_wl,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=l1_E,focus=1.125,M2=1,pulsed=False,phi=0e10)

bn_map = []
for C1 in tqdm(C1I):
    b = [] #[[],[],[]]
    delay_z = Delay_SecondPulse(C1) * 1e-6  
    l2= Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.125,Z_offset=delay_z, pulsed=False)
    for C2 in C2I:
        print("C1 : ", C1, " C2 : ", C2)
        lattice = Lattice(E0= 1492, l1= l1_wl, l2= l2_wl, h = 5 , c1= C1 , c2= C2 , plot= 0)
        elec_M1= lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep)
        z,dE=calc_phasespace(elec_M1,e_E,plot=False)
        #for i in range(3):
        b.append(calc_bn(z,wl_h))
#        bmax[i].append(max(b))
        
    bn_map.append(b)

bn_map = np.array(bn_map)
bn_map = np.transpose(bn_map,axes=(2,1,0))

for i in range(len(wl_h)):
    plt.figure()
    plt.contourf(C1I, C2I,bn_map[i],50)
    plt.colorbar(label="bn")
    plt.title(str(int(wl_h[i]*1e9))+" nm")
    plt.xlabel("Chicane 1 current (A)")
    plt.ylabel("Chicane 2 current (A)")
    

#%%
########## Trying out different R56_2 values ###########

elec = define_bunch(E0=e_E,dE=sigma_E,N=1e4,slicelength=slicelength)
IC2_list = np.linspace(IC2+50,IC2+100,11)
wl = np.linspace(20e-9, 220e-9, 1001)
bmax = []
for C2 in IC2_list:
    lattice = Lattice(E0= 1492, l1= l1_wl, l2= l2_wl, h=5, c1= IC1 , c2= C2 , plot= 0)
    elec_M1 = lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep)
    z,dE=calc_phasespace(elec_M1,e_E,plot=False)
    b = calc_bn(z,wl)
    bmax.append(max(b))                    
    plt.plot(wl,b)

plt.figure()
plt.plot(IC2_list, bmax)

#%%
########## Trying out different R56_1 values ###########

elec = define_bunch(E0=e_E,dE=sigma_E,N=1e4,slicelength=slicelength)
IC1_list = np.linspace(IC1-50,IC1+50,11)
bmax = []
for C1 in IC1_list:
    lattice = Lattice(E0= 1500, l1= l1_wl, l2= l2_wl, h=7, c1= C1 , c2= IC2 , plot= 0)
    l2= Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.3125,X0=0.0e-3,Z_offset=Delay_SecondPulse(C1)*1e-6 ,M2=1,pulsed=1,phi=0e10)
    elec_M1= lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep)
    z,dE=calc_phasespace(elec_M1,e_E,plot=False)
    b = calc_bn(z,wl)
    bmax.append(max(b))                    
    plt.plot(wl,b)

plt.figure()
plt.plot(IC1_list, bmax)

