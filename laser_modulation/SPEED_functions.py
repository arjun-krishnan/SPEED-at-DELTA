#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:58:49 2021

@author: arjun

Issues: When the chicane 1 strength is changed, Z_offset also needs to be changed. Otherwise we dont see any energy modulation 

"""
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy import interpolate,special
import matplotlib.pyplot as plt
from time import time
#from  numba import jit
from pathlib import Path
import sys
import h5py
from scipy.interpolate import RegularGridInterpolator
#import sdds


##### natural constants #####
c = const.c                     # speed of light
e_charge = const.e              # electron charge
m_e = const.m_e                 # electron mass in eV/c^2
Z0 = 376.73                     # impedance of free space in Ohm
epsilon_0 = const.epsilon_0     # vacuum permittivity
mu0 = const.mu_0                # vacuum permeability


#@jit(parallel = True)
def calc_bn(tau0, wl, printmax = True):
    wl = np.asarray(wl).reshape(-1,)
    bn = []
    for i in range(len(wl)):
        z = np.sum(np.exp(-1j * 2 * np.pi * (tau0 / wl[i])))
        bn.append(abs(z) / len(tau0))
    
    index = np.where(bn == max(bn))
    wl_max = wl[index]
    if printmax:
        print("Maximum bunching factor is", np.round(max(bn),4) , " at " , np.round(wl_max[0]*1e9,2) , " nm")
    return(np.array(bn))


def plot_slice(z, wl, slice_len=0, n_slice=40):
    if slice_len != 0:
        n_slice = int((max(z) - min(z)) / slice_len)

    zz = np.linspace(min(z), max(z), n_slice)
    i = 1
    bn, z_slice = [], []
    while i < len(zz):
        z1, z2 = zz[i - 1], zz[i]
        z_slice.append(np.mean([z1, z2]))
        slice_zz = z[(z >= z1) * (z < z2)]
        #print(len(slice_zz))
        if len(slice_zz) == 0:
            bn.append(0)
        else:
            bn.append(max(calc_bn(slice_zz, wl, printmax = False)))
        i += 1
    
    z_slice = np.array(z_slice) - np.mean(z_slice)
    bn      = np.array(bn)
    
    plt.figure()
    plt.plot(z_slice, bn)
    return(z_slice, bn)
    
def write_results(bunch,file_path):
    print("Writing to "+file_path+" ...")
    file = Path(file_path)
    if file.is_file():
        ch = input("The file already exist! Overwrite? (Y/N)")
        if ch == 'y':
            bunch.to_csv(file_path)
    else:
        bunch.to_csv(file_path)


def define_bunch(Test=False, N=1e4, slicelength=8e-6, E0=1.492e9 * e_charge, dE=7e-4, lattice='del008', R56_dE=0.0007,
                 R51_dx=4e-4, R52_dxp=4e-5):
    N_e = int(N)  # number of electrons

    ##### electron parameter #####
    e_E = E0  # electron energy in J
    energyspread = dE

    if lattice == 'eehg':
        # These values are suitable for the big EEHG lattice
        alphaX = 8.811383e-01  # 1.8348
        alphaY = 8.972460e-01  # 0.1999
        betaX = 13.546
        betaY = 13.401
        emitX = 1.6e-8
        emitY = 1.6e-9
        Dx = 0.0894
        Dxprime = -4.3065e-9

    elif lattice == 'del008':
        # The lattice parameters at the beginning of U250 del008 model
        alphaX = 1.92938
        alphaY = 0.210161
        betaX = 6.69295
        betaY = 13.5857
        Dx = -0.0894
        Dxprime = -4.3065e-9
        emitX = 1.6e-8
        emitY = 1.6e-9

    elif lattice == 'del21':
        # The lattice parameters at the beginning of U250 del21 model
        alphaX = 1.18911
        alphaY = 0.260189
        betaX = 5.378
        betaY = 11.4688
        Dx = 0.00377785
        Dxprime = -0.00714072
        emitX = 1.6e-8
        emitY = 1.6e-9

    else:
        print("Unknown input for lattice! Please check!")

    if (Test):
        slicelength = 8e-6
        N_e = int(1e4)
        energyspread = 0e-4
        emitX = 0
        emitY = 0
        Dx = 0
        Dxprime = 0

    # print(slicelength)
    CS_inv_x = np.abs(np.random.normal(loc=0, scale=emitX * np.sqrt(2 * np.pi), size=N_e))
    CS_inv_y = np.abs(np.random.normal(loc=0, scale=emitY * np.sqrt(2 * np.pi), size=N_e))
    phase_x = np.random.rand(N_e) * 2 * np.pi
    phase_y = np.random.rand(N_e) * 2 * np.pi

    # generate random electron parameters according to beam parameters
    elec0 = np.zeros((6, N_e))
    elec0[4] = (np.random.rand(1, N_e) - 0.5) * slicelength  # / c
    elec0[5] = np.random.normal(loc=0, scale=energyspread, size=N_e)  # /e_m/c**2
    elec0[0] = np.sqrt(CS_inv_x * betaX) * np.cos(phase_x) + elec0[5, :] * Dx
    elec0[1] = -np.sqrt(CS_inv_x / betaX) * (alphaX * np.cos(phase_x) + np.sin(phase_x)) + elec0[5, :] * Dxprime
    elec0[2] = np.sqrt(CS_inv_y * betaY) * np.cos(phase_y)
    elec0[3] = -np.sqrt(CS_inv_y / betaY) * (alphaY * np.cos(phase_y) + np.sin(phase_y))

    # Adding two particles with only energy difference to calculate the R56
    for i in range(6):
        if i == 5:
            elec0[i][-2], elec0[i][-1] = 0.0, R56_dE
        else:
            elec0[i][-2], elec0[i][-1] = 0.0, 0.0
        if i == 1:
            elec0[i][-4], elec0[i][-3] = 0.0, R52_dxp
        else:
            elec0[i][-4], elec0[i][-3] = 0.0, 0.0
        if i == 0:
            elec0[i][-6], elec0[i][-5] = 0.0, R51_dx
        else:
            elec0[i][-6], elec0[i][-5] = 0.0, 0.0

    # changing to parameter style: [x,y,z,px,py,pz] in laboratory frame
    elec = coord_change(elec0, e_E)
    np.save("e_dist.npy", elec)
    return (elec)


def coord_change(elec_dummy,e_E):
    elec = np.zeros((6,len(elec_dummy[0])))
    elec[0,:] = elec_dummy[0,:]
    elec[1,:] = elec_dummy[2,:]
    elec[2,:] = elec_dummy[4,:]
    p_elecs   = np.sqrt(((1+elec_dummy[5,:])*e_E)**2-m_e**2*c**4)/c
    elec[5,:] = p_elecs/(np.sqrt(1/np.cos(elec_dummy[1,:])**2+np.tan(elec_dummy[3,:])**2))
    elec[4,:] = elec[5,:]*np.tan(elec_dummy[3,:])
    elec[3,:] = elec[5,:]*np.tan(elec_dummy[1,:])
    return(elec)     

    

class Laser:
    def __init__(self, wl, sigx, sigy, pulse_len, pulse_E, focus, X0=0.0, Z_offset=0, M2=1.0, pulsed=True, D2=0, D3=0):
        self.wl = wl  # Wavelength (m)
        self.sigx = sigx  # Horizontal beam width (m)
        self.sigy = sigy  # Vertical beam width (m)
        self.pulse_len = pulse_len  # FWHM pulse duration (s)
        self.E = pulse_E  # Pulse energy (J)
        self.P_max = self.E / (0.94 * self.pulse_len)
        I0 = (2 * self.P_max) / (np.pi * 4 * sigx * sigy)  # Peak intensity
        self.E0 = np.sqrt(2 * Z0 * I0)  # Peak electric field

        self.M2 = M2
        self.k = 2 * np.pi / self.wl  # Wavenumber (1/m)
        self.omega0 = 2 * np.pi * c / self.wl  # Central angular frequency (rad/s)
        self.sigz = np.sqrt(2) * self.pulse_len * c / 2.3548  # Longitudinal sigma (m)
        self.zRx = np.pi * (2 * self.sigx) ** 2 / (self.M2 * self.wl)  # Rayleigh length (x)
        self.zRy = np.pi * (2 * self.sigy) ** 2 / (self.M2 * self.wl)  # Rayleigh length (y)

        self.beamsize_x = lambda z: self.sigx * np.sqrt(1 + (z / self.zRx) ** 2)
        self.beamsize_y = lambda z: self.sigy * np.sqrt(1 + (z / self.zRy) ** 2)

        self.X0 = X0  # X offset
        self.Z_offset = Z_offset  # Z offset
        self.focus = focus  # Focus position
        self.pulsed = pulsed  # Pulsed flag

        # Spectral phase parameters
        self.D2 = D2  # GDD (s²)
        self.D3 = D3  # TOD (s³)

        # Precompute dispersed temporal profile
        self.t_array, self.E_temporal = self.precompute_temporal_profile()

    def precompute_temporal_profile(self):
        # Convert FWHM pulse duration to Gaussian sigma
        sigma_t = self.pulse_len / (2 * np.sqrt(2 * np.log(2)))

        # Adjust time window to account for dispersion-induced broadening
        if self.D2 != 0 or self.D3 != 0:
            # Estimate broadening from GDD (quadratic in D2)
            sigma_t_broadened = np.sqrt(sigma_t ** 2 + (self.D2 / (2 * sigma_t)) ** 2)
            time_window = 16 * sigma_t_broadened  # Expanded window
        else:
            time_window = 12 * sigma_t  # Default for transform-limited

        # Use sufficient points to resolve phase variations
        Nt = 16384  # Increased to 16384 points for better frequency resolution
        t = np.linspace(-time_window / 2, time_window / 2, Nt)
        dt = t[1] - t[0]

        # Initial Gaussian envelope (without carrier)
        envelope = np.exp(-t ** 2 / (2 * sigma_t ** 2))
        # Analytic signal with carrier frequency
        E_time = envelope * np.exp(1j * self.omega0 * t)

        # FFT to frequency domain
        E_freq = np.fft.fftshift(np.fft.fft(E_time))
        omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nt, dt))

        # Spectral phase terms (expand around central frequency)
        delta_omega = omega - self.omega0
        phase = (
                + 0.5 * self.D2 * delta_omega ** 2
                + (1 / 6) * self.D3 * delta_omega ** 3
        )

        # Apply phase and inverse FFT
        E_freq_phased = E_freq * np.exp(-1j * phase)
        E_time_phased = np.fft.ifft(np.fft.ifftshift(E_freq_phased))

        # Normalize to preserve energy (not peak amplitude)
        original_energy = np.sum(np.abs(E_time) ** 2) * dt
        new_energy = np.sum(np.abs(E_time_phased) ** 2) * dt
        E_temporal = np.real(E_time_phased) * np.sqrt(original_energy / new_energy)

        return t, E_temporal

    def E_field(self, X, Y, Z, T):
        Zdif_x = Z - self.focus
        Zdif_y = Z - self.focus
        X = X - self.X0

        # Beam curvature
        R_x = Zdif_x * (1 + (self.zRx / Zdif_x) ** 2)  # if Zdif_x != 0 else np.inf
        R_y = Zdif_y * (1 + (self.zRy / Zdif_y) ** 2)  # if Zdif_y != 0 else np.inf

        # Spatial envelope
        central_E_field = self.E0 * self.sigx / self.beamsize_x(Zdif_x)
        spatial_factor = np.exp(
            -(Y / self.beamsize_y(Zdif_y)) ** 2
            - (X / self.beamsize_x(Zdif_x)) ** 2
        )

        # Temporal factor (retarded time)
        tau = T - (Z + self.Z_offset) / c
        temporal_factor = np.interp(tau, self.t_array, self.E_temporal)

        # Combine spatial and temporal factors
        envelope = spatial_factor * temporal_factor if self.pulsed else spatial_factor

        # Spatial phase terms (Gouy phase, wavefront curvature)
        phase = np.cos(
            - self.k * X ** 2 / (2 * R_x)
            - self.k * Y ** 2 / (2 * R_y)
            + np.arctan(Zdif_x / self.zRx)  # Gouy phase (x)
            + np.arctan(Zdif_y / self.zRy)  # Gouy phase (y)
        )

        return central_E_field * envelope * phase

        
class Lattice:
    def __init__(self, E0= 1492,l1= 800e-9, l2= 400e-9, h= 4, c1= 800, c2= 800, filename = None, is2d=True, plot = True):
        
        self.is2d = is2d
        self.len = 5.74925
        windings = 48 # upper plus lower coil
        gap = 0.05 
        period = 0.25
        yoke = 0.08
        edge = 0.02    # softness parameter for magnet edge
        drift = 0.5   # befor and after magnetic structure
        dl = 0.0005   # longitudinal interval
        E0 = E0    # beam energy in MeV
        e_gamma = E0/0.511
        
        if filename != None:
            
            df = pd.read_csv(filename,sep='\t')
            self.l = np.array(df['z']) / 1000 
            self.b = np.array(df['By']) 
            self.len = self.l[-1] 
            
            if is2d:
                
                df = pd.read_csv("fieldfiles/M2_279A_transverse_field.txt",skiprows=3,header=None,sep='\t')
                x1 = np.array(df[0], dtype='float32')
                B1 = np.array(df[1], dtype='float32')
                B_norm = (B1/B1[300]).reshape(-1,1)
                B_2D = B_norm * self.b
                self.B_func = RegularGridInterpolator((x1, self.l), B_2D, method='linear', bounds_error=False, fill_value=0)
            
            else:
                self.B_func = interpolate.interp1d(self.l,self.b)
                
            if plot == True:
                plt.plot(self.l,self.b)
            return
        
        if l1 == 0:   
            IM1 = 0
        else:
            laser_wl = l1
            K = np.sqrt(4 * laser_wl * e_gamma**2 / period - 2)
            B = 2 * np.pi * K * m_e * c / (e_charge * period)
            factor = mu0 * windings / gap
            IM1 = 1.14 * B / factor  # Current for the first modulator. The factor 1.14 is the correction term to match the CST simulations
        
        if l2 == 0:
            IM2 = 0
        else:
            laser_wl = l2
            K = np.sqrt(4 * laser_wl * e_gamma**2 / period - 2)
            B = 2 * np.pi * K * m_e * c / (e_charge * period)
            factor = mu0 * windings / gap
            IM2 = 1.14 * B / factor  # Current for the second modulator
        
        if h == 0:
            IR1 = 0
        else:
            laser_wl = 800e-9 / h
            K = np.sqrt(4 * laser_wl * e_gamma**2 / period - 2)
            B = 2 * np.pi * K * m_e * c / (e_charge * period)
            factor = mu0 * windings / gap
            IR1 = 1.14 * B / factor  # Current for the radiator
        

        ID1 = IM1/2
        ID2 = IM2/2
        ID3 = IM2/2
        ID4 = IR1/2
        IC1 = c1     # power supply for chicane 1 ("Danfysik")
        IC2 = c2     # power supply for chicane 2 ("Danfysik")
    
        # coil current
        curr=[ -ID1 , IM1 , -IM1 ,  IM1 , -IM1 , IM1 , -IM1 , IM1 , \
               -IC1 , -IC1, -ID1, IC1 , IC1 , IC1 , IC1 , -ID2 , -IC1 , -IC1 , \
               IM2 , -IM2 , IM2 , -IM2 , IM2 , -IM2 , IM2 , \
              -IC2 , -ID3 , IC2 , IC2 , -ID4 , -IC2 , \
               IR1 , -IR1 , IR1 , -IR1 , IR1 , -IR1 , ID4 ]

        factor = mu0 * windings / gap * 1.19     #The factor 1.19 is the correction term to match the CST simulaitons
        b0 = np.zeros(len(curr))
        for m in range(len(curr)):
            b0[m] = factor * curr[m]
                
        
        nm = len(curr)
        magnet = period / 2
        magnet2 = magnet / 2
        yoke2 = yoke / 2
        len_range = nm * magnet + 2 * drift
        
        nl = int(len_range / dl)
        self.l = np.zeros(nl)
        for k in range(nl):
            self.l[k] = (k - 0.5) * dl
        
        self.b = np.zeros(nl)
        
        for m in range(nm):
            l1 = drift + (m - 1) * magnet  # Magnet center
            for k in range(nl):
                self.b[k] = self.b[k] + b0[m] / (np.exp((l1 + magnet2 - yoke2 - self.l[k]) / edge) + 1)
        
            for k in range(nl):
                self.b[k] = self.b[k] + b0[m] / (np.exp((self.l[k] - magnet2 - yoke2 - l1) / edge) + 1)
        
        if plot:
            plt.plot(self.l, self.b)
        
        self.B_func = interpolate.interp1d(self.l, self.b)


def calc_phasespace(bunch,e_E,plot=False):
    p = np.sqrt(np.sum(bunch[3:]**2 , axis=0))
    E = np.sqrt(m_e**2 * c**4 + p**2 * c**2)
    dEE = E/e_E - 1
    z = np.copy(bunch[2,:]) 
    
    if plot:
        plt.figure()
        plt.plot((z-np.mean(z)) * 1e6 , dEE ,',')
        plt.xlabel('z ($\mu m$)')
        plt.ylabel('$\Delta E/E_0$')
        plt.tight_layout()
    return(z,dEE)    


def rotate_particles(p, angle):
    # Reference particle (last column)
    reference_particle = p[:, -1]
    
    # Translate particles to make reference particle the origin
    translated_p = p - reference_particle[:, np.newaxis]
    
    # Rotation matrix for rotating around the y-axis by angle alpha
    R_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    # Apply the rotation
    rotated_p = np.dot(R_y, translated_p)
    
    # Translate back to the original position
    #rotated_p += reference_particle[:, np.newaxis]
    
    return rotated_p


    
def lsrmod_track(Mod, Lsr, e_bunch, Lsr2=None, tstep=1e-12, zlim=None, 
                 plot_track=False, disp_Progress=True, 
                 get_R512=True, R51_dx=4e-4, R52_dxp=4e-5):
    

    N_e = len(e_bunch[0])
    bunch = np.copy(e_bunch)
    z_0 = np.mean(bunch[2])
    
    bunch[2] -= z_0
    z_mean = np.mean(bunch[2]) 
    count = 0
    progressrate = 10
    progress = 0 

    t = 0

    starttime = time()
    EE = []
    track_x = [np.copy(bunch[0][-6:])]
    track_z = [np.copy(bunch[2][-6:])]
  #  dZZ = []
  #  ZZ = []
    
    R51, R52 = [0.0], [0.0]
    
    if zlim == None:
        zlim = Mod.len
    
    while z_mean < z_0 + zlim:
        if disp_Progress:
            if progress < (z_mean) / zlim * progressrate:
                elapsed = time() - starttime
                sys.stdout.write('\r Progress: ' + str(progress) + '/' + str(progressrate) + " \t ETA: " + str(np.round(elapsed/60 * (progressrate / (progress+0.01) - 1), 2)) + " mins ")
                sys.stdout.flush()
                progress += 1
    
        z = np.copy(bunch[2])
        z_mean = np.mean(z)
    #    ZZ.append(z_mean)
    
        p_field = bunch[3:]
        p_vec = np.sqrt(np.sum(p_field**2, axis=0))
        gamma_vec = np.sqrt((p_vec / m_e / c) ** 2 + 1)
    
        Efield_x_vec = Lsr.E_field(bunch[0],bunch[1],bunch[2],t)
        if Lsr2 != None: 
            Efield_x_vec = Lsr.E_field(bunch[0],bunch[1],bunch[2],t) + Lsr2.E_field(bunch[0],bunch[1],bunch[2],t)
        EE.append(Efield_x_vec[0])

        try:
            if Mod.is2d:
                pos = np.array([bunch[0], bunch[2]]).T
                Bfield_y_vec = Mod.B_func(pos) + Efield_x_vec / c
            else:    
                Bfield_y_vec = Mod.B_func(z) + Efield_x_vec / c
        except:
            Bfield_y_vec = Efield_x_vec / c
    
        dp_x_vec = (Efield_x_vec - p_field[2] * Bfield_y_vec / m_e / gamma_vec) * e_charge * tstep
        dp_y_vec = np.zeros(N_e)
        dp_z_vec = p_field[0] * Bfield_y_vec / m_e / gamma_vec * e_charge * tstep
    
    
        p_new = bunch[3:] + [dp_x_vec , dp_y_vec , dp_z_vec]
        p_vec_new = np.sqrt(np.sum(p_new**2 , axis=0))
        gamma_vec_new = np.sqrt((p_vec_new / m_e / c)**2 + 1)                           
        spatial_new = bunch[0:3,:] + p_new / m_e / gamma_vec_new * tstep   
    
        comoving_angle = p_new[0, -1] / p_new[2, -1]
        rotated_pos = rotate_particles(spatial_new, -comoving_angle)
        
        R51.append((rotated_pos[2,-6] - rotated_pos[2,-5]) / R51_dx)
        R52.append((rotated_pos[2,-4] - rotated_pos[2,-3]) / R52_dxp)
        
        bunch[0:3] = np.copy(spatial_new)
        bunch[3:] = np.copy(p_new)
        
        t += tstep
        
        track_x.append(np.copy(bunch[0][-6:]))
        track_z.append(np.copy(bunch[2][-6:]))
        #track_x[0].append(bunch[0][-2])
        #track_z[0].append(bunch[2][-2])
        #track_x[1].append(bunch[0][-1])
        #track_z[1].append(bunch[2][-1])
        
   #     dz = (t * c - np.mean(bunch[2]))
  #      dZZ.append(dz)
  
    else:
        if disp_Progress:
            print('Progress: '+str(progress)+'/'+str(progressrate))

    if plot_track == True:
        track_x = np.array(track_x)
        track_z = np.array(track_z)
        return(bunch,track_x,track_z)
    
    if get_R512 == True:
        return bunch, np.array(R51), np.array(R52), np.array(track_z)[:,-1], np.array(track_x)[:,-1]
    
    endtime = time()
    print("\nRuntime:  " , np.round(endtime-starttime,2) , " sec")
    return bunch
        
def chicane_track(bunch_in, R56, R51=0, R52=0, isr=False):
    RM = pd.read_csv("TM.txt", usecols=range(1, 7))
    RR = np.array(RM)
    RR[4, 0], RR[4, 1] = R51, R52    
    RR[4, 5] = R56

    pp = np.sum(bunch_in[3:]**2)**0.5
    dE = ((pp**2 * c**2) + (m_e**2 * c**4))**0.5 / e_charge - 1492e6
    MM = np.asarray([[bunch_in[0]], [np.arctan(bunch_in[3] / bunch_in[5])], [bunch_in[1]], [np.arctan(bunch_in[4] / bunch_in[5])], [bunch_in[2]], [dE / 1492e6]])
    p_mod = MM.transpose((2, 0, 1))
    p_end = np.matmul(RR, p_mod)
    elec_dummy = p_end.transpose((2, 1, 0))[0]
    # convert to parameter style: [x,y,z,px,py,pz] in laboratory frame
    bunch_out = coord_change(elec_dummy)
    return bunch_out

def calc_R56(A11, A22, dE=7e-4, K=2, m=21, n=-1, wl=800e-9):
    A1, A2 = A11 / dE, A22 / dE
    B2 = (m + (0.81 * m**(1 / 3))) / ((K * m + n) * A2)
    R56_2 = B2 / (2 * np.pi / wl) / dE
    rr2 = R56_2  # optimal R56(2)
    print("\nOptimum R56 values:")
    print("R56(2) =", np.round(R56_2 * 1e6, 2), "microns")
    R56_1 = np.linspace(50e-6, 2000e-6, 1000)
    bn = []
    for R in R56_1:
        B1 = R * (2 * np.pi / wl) * dE
        bn.append(abs(special.jv(m, -(K * m + n) * A2 * B2) * special.jv(n, (A1 * (n * B1 + ((K * m + n) * B2)))) * np.exp(-0.5 * (n * B1 + (K * m + n) * B2)**2)))
    
    bmax = max(bn)
    i = bn.index(bmax)
    r11 = R56_1[i]
    bn[i] = 0
    bmax = max(bn)
    i = bn.index(bmax)
    r12 = R56_1[i]
    rr1 = r12 if r12 < r11 else r11  # optimal R56(1)
    print("R56(1) =", np.round(rr1 * 1e6), "microns")
    return rr1, rr2
    
'''
## This code section is to include incoherent syncrotron radiation induced energy spread for the particles. This works only for dipoles    
    if isr:
        L=0.37
        d=0.10      
        alpha=np.sqrt(R56/((4*L/3)+2*d))
        #B=(e_gamma*e_beta*e_m*c*np.sin(alpha))/(L*e_charge)
        rho=L/alpha
        sigE=4*np.sqrt((55*const.alpha*(((const.h/(2*np.pi))*const.c)**2)*(e_gamma**7)*L)/((2*24*(3**0.5))*(rho)**3))
        print(sigE/e_E)
        SR_dE=np.random.normal(loc=0,scale=sigE/e_E,size=len(elec2[5]))
        elec2[5]+=SR_dE
    elec2[5]=(elec2[5]+1)*e_E/m_e/c**2
    #return(elec2)


import h5py
from scipy.interpolate import RegularGridInterpolator

file = h5py.File("Mod2_400_Rad_133_B-Field_10.h5",'r')
Bf = np.array(file['/B-Field'])
pos = np.array(file['/Position'])
Bf = Bf.view('<f4').reshape(-1, 3)
pos = pos.view('<f8').reshape(-1, 3)

x = np.unique(pos[:, 0])
y = np.unique(pos[:, 1])

Bx = Bf[:, 0].reshape((len(y), len(x)))
By = Bf[:, 1].reshape((len(y), len(x)))
Bz = Bf[:, 2].reshape((len(y), len(x)))

Bz_interp = RegularGridInterpolator((x, y), Bz.T, bounds_error=False, fill_value=None)
sample_pos = np.array([np.linspace(-2000, 2000, 1000), np.zeros(1000)]).T

B_test = Bz_interp(sample_pos)
plt.plot(B_test)

            if Path(filename).suffix in ['.csv', '.txt']:
                self.is2d = False
                df = pd.read_csv(filename,sep='\t')
                self.l = np.array(df['z']) / 1000 
                self.b = np.array(df['By']) 
                self.len = self.l[-1]   
                
                elif Path(filename).suffix == '.h5':
                    with h5py.File(filename) as f:
                        self.is2d = True
                        x = np.array(f['x']) / 1000
                        self.l = np.array(f['z']) / 1000
                        self.b = np.array(f['By']) 
                        self.len = self.l[-1]
                        self.B_func = RegularGridInterpolator((x, self.l), self.b, method='linear', bounds_error=False, fill_value=0)
                        if plot == True:
                            plt.plot(self.l,self.b[np.where(x==0)[0][0]])
                        return
            
                
'''

