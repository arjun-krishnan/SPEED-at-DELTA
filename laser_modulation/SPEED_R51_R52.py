# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:33:33 2023

@author: arjun
"""

from define_laser_lattice import *
from phase_space_manipulation import *
from tracking_functions import *
from tqdm import tqdm
from glob import glob

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
elec = np.copy(bunch_init)

##### defining Laser 1 #####
l1_wl = 400e-9   # wavelength of the laser
l1_sigx = 1.0e-3   # sigma width at the focus
l1_fwhm =45e-15  # pulse length 
l1_E = 2.0e-3 #1.5      # pulse energy

##### defining Laser 2 #####
l2_wl = 400e-9
l2_sigx = 1.0e-3
l2_fwhm = 45e-15
l2_E = 4.0e-3

# Calculate the delay for the second seed pulse
p = [ 6.29880952e-04, -8.26071429e-02,  1.10083333e+01]

Delay_SecondPulse = lambda I : (p[0] * I**2 + p[1] * I + p[2] ) - 3

IC1 = 0                 # Current for the first chicane
delay_z = Delay_SecondPulse(IC1) * 1e-6    # Corresponding R56 value in microns

#%%
## First calculate energy modualtion amplitude and the optimum chicane currents
##### defining the magnetic configuration#####

lattice = Lattice(filename='fieldfiles/M2_400_C2_500_R_200/M2_400_C2_500_R_200_R52mod_30A_input.txt', plot= 1)
#lattice = Lattice(filename='fieldfiles/M1_400_C1_100_M2_200_input.txt', plot= 1)

l1 = Laser(wl=l1_wl,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=l1_E,focus=0.93,M2=1,pulsed=0,phi=0e10)
l2 = Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.125,X0=1.0e-3,\
            M2=1,pulsed=0,phi=0e10) #Z_offset=delay_z*10

#%%
    
elec_test , R51, R52, track_z = lsrmod_track(lattice,l1,bunch_test,tstep=tstep,zlim=6,disp_Progress=False,get_R512=True)

plt.plot(track_z, R51)
plt.plot(track_z, R52)

#%%
elec_test , track_x, track_z = lsrmod_track(lattice,l1,bunch_test,tstep=tstep,zlim=6,disp_Progress=False,plot_track=True)
z , dE = calc_phasespace(elec_test,e_E,plot=True)
A22 = abs(min(dE))
print(A22)
print("R56 : ",(z[-2]-z[-1])/7e-4*1e6)
print("R52 z_diff : ",(z[-4]-z[-3])*1e9)
print("R51 z_diff : ",(z[-6]-z[-5])*1e9)
R52_l = (track_z.T[-3] - track_z.T[-4])/4e-5
print("R52 :" , (R52_l[2910] - R52_l[2320])*1e6)

#%% Simulating the particle track and R52 values 

#l1 = Laser(wl=400e-9,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=l1_E,focus=0.93,M2=1,pulsed=0,phi=0e10)

files = glob("fieldfiles/M2_400_C2_500_R_200/*A_input.txt")

IA = np.arange(0,55,5)
#IA = np.concatenate((np.arange(-5,-55,-5),np.arange(0,55,5)))
trackz = []
R51_list, R52_list = [], []

R52_modlist, R51_modlist = [],[]

for i,f in enumerate(tqdm(files[:])):
    lattice = Lattice(filename=f, plot=0)
    elec_test , R51, R52, track_z = lsrmod_track(lattice,l1,bunch_test,tstep=tstep,disp_Progress=False,get_R512=True)
    #z , dE = calc_phasespace(elec_test,e_E,plot=0)
    #R52_l = (track_z.T[-3] - track_z.T[-4])/4e-5
    plt.figure(0)
    plt.plot(track_z,R51*1e3,label=str(IA[i])+" A")
    trackz.append(track_z.T[1])
    R51_list.append(R51)
    
    plt.figure(1)
    plt.plot(track_z,R52*1e3,label=str(IA[i])+" A")
    trackz.append(track_z.T[1])
    R52_list.append(R52)
    
    #plt.plot(track_z.T[0],track_x.T[0]*1e3,label=str(IA[i])+" A")
    R52_modlist.append((R52[2953] - R52[2284])*1e6)       # R52 of the second chicane
    R51_modlist.append((R51[2953] - R51[2284])*1e6)
  #  R52_modlist_1.append((R52_l[1750] - R52_l[905])*1e6)        # R52 of the first chicane
    #print(R52)

plt.legend() #title="Additional current for coils 27 & 30")
plt.xlabel("s (m)")
plt.ylabel("R$_{52}$ (mm)")
plt.grid()

plt.figure(0)
plt.legend() #title="Additional current for coils 27 & 30")
plt.xlabel("s (m)")
plt.ylabel("R$_{51}$ (mm)")
plt.grid()


plt.figure()
plt.plot(IA, R52_modlist)
plt.xlabel("additional current for coils 27 and 30 (A)")
plt.ylabel("R52 difference between M2 and Radiator ($\mu m$)")
R52_zero = np.polyfit(R52_modlist, IA,deg=1)[1]
plt.title("R52 = 0 at "+str(np.round(R52_zero,2))+" A", loc='right')
plt.title("M1_800_C1_450_R_200", loc='left')
plt.grid()

plt.figure()
plt.plot(IA, R51_modlist)
plt.xlabel("additional current for coils 27 and 30 (A)")
plt.ylabel("R51 difference between M2 and Radiator ($\mu m$)")
R51_zero = np.polyfit(R51_modlist, IA,deg=1)[1]
plt.title("R51 = 0 at "+str(np.round(R51_zero,2))+" A", loc='right')
plt.title("M1_800_C1_450_R_200", loc='left')
plt.grid()

#%%

plt.figure()
plt.plot(IA, R52_modlist_1)
plt.xlabel("additional current for coils 11 and 16 (A)")
plt.ylabel("R52 difference between M1 and M2 ($\mu m$)")
R52_zero = np.polyfit(R52_modlist_1, IA,deg=1)[1]
plt.title("R52 = 0 at "+str(np.round(R52_zero,2))+" A", loc='right')
plt.title("M1_800_C1_450_M2_400_C2_500_R_200", loc='left',fontsize=10)
plt.grid()

#%% Simulating the energy modulation amplitude for each U250 setup for different horizontal position of laser waist

#x0 = np.linspace(1e-3,1.37e-3,11)
x0 = np.linspace(-0.17e-3,0.4e-3,15)
l2_E = 2.5e-3


Emod = []
for i,f in enumerate(tqdm(files)):
    lattice = Lattice(filename=f, plot=0)
    l2 = Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.125,X0=0,\
               Z_offset=delay_z*10 ,M2=1,pulsed=0,phi=0e10)
    l1 = Laser(wl=800e-9,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=2.9e-3,focus=0.93,X0=0.2e-3,M2=1,pulsed=0,phi=0e10)
    elec_test , track_x, track_z = lsrmod_track(lattice,l1,bunch_test,tstep=tstep,disp_Progress=False,plot_track=True,zlim=1.35)
    z , dE = calc_phasespace(elec_test,e_E,plot=False)
    Emod.append(abs(min(dE)))

Emod = np.array(Emod)
plt.plot(IA, Emod*100)
plt.xlabel("additional current for coils 11 and 16 (A)")
plt.ylabel("2nd energy modulation (%)")
plt.grid()

#%% Simulating the phase space and bunching factor

l2_E = 0.5
l2 = Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.125,X0=0.0e-3,\
           Z_offset=delay_z*10 ,M2=1,pulsed=0,phi=0e10)
    
elec = define_bunch(E0=e_E,dE=sigma_E,N=0.5e5,slicelength=5e-6)
wl = np.array([400e-9/2,400e-9/3,400e-9/4])
bn = []
for i,f in enumerate(tqdm(files)):
    lattice = Lattice(filename=f, plot=0)
    elec_M1 = lsrmod_track(lattice,l2,elec,tstep=tstep,zlim=4.25)
    z , dE = calc_phasespace(elec_M1,e_E,plot=0)
    b = calc_bn(z,wl)
    bn.append(b)

bn = np.array(bn)
plt.plot(IA, bn)
plt.legend(["200 nm","133 nm","100 nm"])
plt.xlabel("additional current for coils 27 and 30 (A)")
plt.ylabel("bunching factor")
plt.grid()

#%%

#l1 = Laser(wl=800e-9,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=8.5e-3,focus=0.93,X0=0.2e-3,M2=1,pulsed=0,phi=0e10)
l2 = Laser(wl=400e-9,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=3.5e-3,focus=3.125,X0=0.0e-3,\
           Z_offset=delay_z*0 ,M2=1,pulsed=0,phi=0e10)
elec = define_bunch(E0=e_E,dE=sigma_E,N=0.5e5,slicelength=10e-6)
#x_off = np.linspace(-0.17e-3,0.21e-3,11)
x_off = np.linspace(-0.17e-3,0.4e-3,15)

files = glob("fieldfiles/M1_400_C1_100_M2_200/*A_input.txt")
h = np.arange(4,11)
wl = 800e-9/h
#np.linspace(50e-9,250e-9,1001)
bn = []
for i,f in enumerate(tqdm(files)):
    lattice = Lattice(filename=f, plot=0)
    l2 = Laser(wl=400e-9,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=3.5e-3,focus=3.125,X0=1e-3,\
               Z_offset=delay_z*0 ,M2=1,pulsed=0,phi=0e10)
    elec_M1 = lsrmod_track(lattice,l1,elec,Lsr2=None,tstep=tstep,zlim=3)
    z , dE = calc_phasespace(elec_M1,e_E,plot=False)
    b = calc_bn(z,wl)                     #calculating bunching factor
    bn.append(b)
    #plt.plot(wl*1e9,b,label=str(IA[i])+" A")

bn = np.array(bn)
plt.plot(IA, bn[:,:-3])
plt.legend(np.round(wl*1e9,1))
plt.xlabel("additional current for coils 11 and 16 (A)")
plt.ylabel("bunching factor")
plt.legend(np.round(wl*1e9,1), title="wavelength (nm)")
plt.grid()

#%%

lattice = Lattice(filename='fieldfiles/M1_800_C1_450_M2_400_C2_500_R_200/M1_800_C1_450_M2_400_C2_500_R_200_R52mod_05A_input.txt', plot= 0)
l1 = Laser(wl=800e-9,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=8.5e-3,focus=0.93,X0=0.2e-3,M2=1,pulsed=0,phi=0e10)
l2 = Laser(wl=400e-9,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=3.5e-3,focus=3.125,X0=0.0e-3,\
           Z_offset=delay_z*0 ,M2=1,pulsed=0,phi=0e10)
    
elec = define_bunch(E0=e_E,dE=sigma_E,N=0.25e5,slicelength=5e-6)
print("\nTracking through the lattice...")
elec_M1 = lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep,zlim=4.25)
z , dE = calc_phasespace(elec_M1,e_E,plot=True)

#%%

plt.figure()
wl = np.linspace(50e-9,250e-9,1001)
b = calc_bn(z,wl)                     #calculating bunching factor
plt.plot(wl*1e9,b)
plt.xlabel("wavelength (nm)")
plt.ylabel("bn")
plt.grid()