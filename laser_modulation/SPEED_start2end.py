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
c=const.c           # speed of light
e_charge=const.e    # electron charge
m_e=const.m_e       # electron mass in eV/c^2
Z0=376.73           # impedance of free space in Ohm
epsilon_0= const.epsilon_0 # vacuum permittivity

#e_E=1.492e9*e_charge    # electron energy in J
e_E=1.5e9*e_charge    # electron energy in J
e_gamma=e_E/m_e/c**2    # Lorentz factor
sigma_E=7e-4
#sigma_E= 4.69e-4
#sigma_E= 2e-4
        
##### Simulation parameter #####
slicelength=20e-6    # length of simulated bunch slice in m
tstep=5e-12         # timestep in s
N_e=int(1e5)        # number of electrons

bunch_test=define_bunch(Test=True,E0=e_E)
bunch_init=define_bunch(Test=False,E0=e_E,dE=sigma_E,N=N_e,slicelength=slicelength)
elec=np.copy(bunch_init)

##### defining Laser 1 #####
l1_wl= 800e-9   # wavelength of the laser
l1_sigx= 1e-3   # sigma width at the focus
l1_fwhm=45e-15  # pulse length 
l1_E= 0e-3      # pulse energy

##### defining Laser 2 #####
l2_wl= 400e-9
l2_sigx= 1e-3
l2_fwhm=45e-15
l2_E= 0e-3

C1_R56 = lambda I : 8.57242860e-04* I**2 + -1.17140062e-01* I + 1.61857761e+01

R56_1_I= 300                    # Current for the first chicane
R56_1_set = C1_R56(R56_1_I)     # Corresponding R56 value in microns

##### defining the magnetic configuration#####
lattice = Lattice(E0= 1500, l1= l1_wl, l2= l2_wl, h=5, c1= R56_1_I, c2= 0, plot= 1)

l1= Laser(wl=l1_wl,sigx=1*l1_sigx,sigy=l1_sigx/1,pulse_len=l1_fwhm,pulse_E=l1_E,focus=1.125,M2=1,pulsed=1,phi=0e10)
l2= Laser(wl=l2_wl,sigx=1*l2_sigx,sigy=1*l2_sigx,pulse_len=l2_fwhm,pulse_E=l2_E,focus=3.3125,X0=0.0e-3,Z_offset=(R56_1_set/2)*1e-6+0e-6,M2=1,pulsed=1,phi=0e10)

#### Test Tracking through Modulators
elec_test= lsrmod_track(lattice,l1,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=True)
A11=(max(dE))

elec_test= lsrmod_track(lattice,l2,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=True)
A22=(max(dE))
print("A1= ",A11,"\t A2= ",A22)

R56_1_opt , R56_2_opt = calc_R56(A11, A22, m=5)


#%%

print("\n\nTracking through the lattice...")
elec_M1= lsrmod_track(lattice,l1,elec,Lsr2=l2,tstep=tstep)
#plt.figure()
z,dE=calc_phasespace(elec_M1,e_E,plot=True)


#%%
plt.figure()
wl=np.linspace(20e-9,250e-9,1001)
b=calc_bn(z,wl)                     #calculating bunching factor
plt.plot(wl,b)
print(max(b))
plot_slice(z, wl, n_slice=100)

#%%

#### Test Tracking through Modulators
elec_test= lsrmod_track(mod1,l1,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=False)
#plt.plot(z,dE,',r')
A11=(max(dE))

#elec_test= lsrmod_track(mod2,l2,bunch_test,tstep=tstep,disp_Progress=False)
#z,dE=calc_phasespace(elec_test,e_E,plot=False)
#A22=(max(dE))
#plt.plot(z,dE,',b')


elec_test= lsrmod_track(mod2,l2,bunch_test,tstep=tstep,disp_Progress=False)
z,dE=calc_phasespace(elec_test,e_E,plot=False)
A22=(max(dE))
print("A1= ",A11,"\t A2= ",A22)
plt.plot(z,dE,',g')

#Calculating the best choice of R56 for this specific A1 and A2
r56_1,r56_2=calc_R56(A11, A22,dE=sigma_E,K=l1_wl/l2_wl,m=8,n=-1)

#%%

print("\n\nTracking through the lattice...")
elec_M1= lsrmod_track(mod1,l1,elec,Lsr2=l2,tstep=tstep)
plt.figure()
z,dE=calc_phasespace(elec_M1,e_E,plot=True)

elec_C1= lsrmod_track(chic1,l1,elec_M1,Lsr2=l2,tstep=tstep)
plt.figure()
z,dE=calc_phasespace(elec_C1,e_E,plot=True)

elec_M2= lsrmod_track(mod2,l1,elec_C1,Lsr2=l2,tstep=tstep)
plt.figure()
z,dE=calc_phasespace(elec_M2,e_E,plot=True)

elec_C2= lsrmod_track(chic2,l1,elec_M2,Lsr2=l2,tstep=tstep)
plt.figure()
z,dE=calc_phasespace(elec_C2,e_E,plot=True)



#%%

'''    
bn=[]
r1=np.linspace(rr2-5e-6,rr2+2.5e-6,11)
for r in r1: 
    p_end2= track_chicane(p_mod2,R56=r,isr=False)
    tau2=-p_end2[4]*c
    b=[]
    wl=np.linspace(20.4e-9,20.6e-9,101)
    b=calc_bn(tau2,wl)
    bn.append(max(b))
plt.plot(r1,bn)  
print(max(bn),r1[bn.index(max(bn))])


bunch=pd.DataFrame({"x":elec[0],"y":elec[1],"z":elec[2],"px":elec[3],"py":elec[4],"pz":elec[5]})
write_results(bunch,"Bunch_after_mod1.txt")
     


mod1= Modulator(periodlen=0.20,periods=9,laser_wl=l1_wl,e_gamma=e_gamma)
EE=np.linspace(0.2e-3,4e-3,10)
l1_wl=800e-9
l1_sigx=1.0e-3
l1_w0=l1_sigx*np.sqrt(2)
l1_fwhm= 40e-15
A1_el,A1_py=[],[]
for E1 in EE:
    l1_E= E1
    l1_P_max=l1_E/(0.94*l1_fwhm)

    modify_lattice("lattice_test.lte",Bu=mod1_Bmax,P0=l1_P_max,w0=l1_w0)
    subprocess.run(["mpiexec","-np","7","Pelegant","run_test.ele"])
    dat=sdds.SDDS(0)
    dat.load("run_test.out")
    dE=(np.array(dat.columnData[5][0])*e_m*c**2/e_E)-1
    #tau=-(np.array(dat.columnData[4][0])*c)
    A1_el.append(max(dE))
    #plt.plot(tau-np.mean(tau),dE,',b')
    l1= Laser(wl=l1_wl,sigx=l1_sigx,pulse_len=l1_fwhm,pulse_E=l1_E,focus=mod1.len/2,M2=1.0,pulsed=False)
    elec_test= lsrmod_track(mod1,l1,bunch_test,tstep=tstep)
    z,dE=plot_phasespace(elec_test)
    A1_py.append(max(dE))
A_rat=np.array(A1_el)/np.array(A1_py)
plt.figure()
plt.plot(EE,A_rat)
'''
