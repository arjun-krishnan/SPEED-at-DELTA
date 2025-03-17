# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:15:20 2025

@author: arjun
"""

import numpy as np
import pandas as pd
import scipy.constants as const
from scipy import interpolate, special
import matplotlib.pyplot as plt
from time import time
import numba as nb
from pathlib import Path
import sys
import h5py
from scipy.interpolate import RegularGridInterpolator
import cupy as cp  # Main GPU array library
import numpy as np  # Only for CPU fallback
from cupyx.scipy.interpolate import Akima1DInterpolator as cupy_interp1d


##### natural constants #####
c = const.c                     # speed of light
e_charge = const.e              # electron charge
m_e = const.m_e                 # electron mass in eV/c^2
Z0 = 376.73                     # impedance of free space in Ohm
epsilon_0 = const.epsilon_0     # vacuum permittivity
mu0 = const.mu_0                # vacuum permeability

def calc_bn(tau0, wl, printmax = True):
    wl = cp.asarray(wl).reshape(-1,)
    #tau0 = tau0.get()
    bn = []
    for i in range(len(wl)):
        z = np.sum(cp.exp(-1j * 2 * cp.pi * (tau0 / wl[i])))
        bn.append(abs(z) / len(tau0))

    bn = cp.asarray(bn)
    index = cp.where(bn == cp.max(bn))
    wl_max = wl[index]
    if printmax:
        print("Maximum bunching factor is", cp.round(max(bn),4) , " at " , cp.round(wl_max[0]*1e9,2) , " nm")
    return(bn.get())


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
        

def define_bunch_gpu(Test=False, N=1e4, slicelength=8e-6, E0=1.492e9*e_charge, 
                    dE=7e-4, lattice='del008', R56_dE=0.0007, 
                    R51_dx=4e-4, R52_dxp=4e-5):
    """Generate particle bunch directly on GPU with CuPy"""
    N_e = int(N)
    ##### electron parameter #####
    e_E = E0   # electron energy in J
    energyspread = dE 
        
    if lattice == 'eehg':
        #These values are suitable for the big EEHG lattice
        alphaX = 8.811383e-01 #1.8348
        alphaY = 8.972460e-01 #0.1999
        betaX = 13.546
        betaY = 13.401
        emitX = 1.6e-8
        emitY = 1.6e-9   
        Dx    = 0.0894
        Dxprime = -4.3065e-9 
    
    elif lattice == 'del008':
        # The lattice parameters at the beginning of U250 del008 model
        alphaX = 1.92938
        alphaY = 0.210161
        betaX = 6.69295
        betaY = 13.5857  
        Dx    = -0.0894
        Dxprime = -4.3065e-9 
        emitX = 1.6e-8
        emitY = 1.6e-9 
        
    elif lattice == 'del21':
        # The lattice parameters at the beginning of U250 del21 model
        alphaX = 1.18911
        alphaY = 0.260189
        betaX = 5.378
        betaY = 11.4688
        Dx    = 0.00377785
        Dxprime = -0.00714072
        emitX = 1.6e-8
        emitY = 1.6e-9 
    
    else:
        print("Unknown input for lattice! Please check!")
        
    if(Test):
        slicelength = 8e-6 
        N_e = int(1e4)
        energyspread = 0e-4
        emitX = 0
        emitY = 0  
        Dx = 0
        Dxprime = 0
        
    # =============================================
    # GPU-optimized random number generation
    # =============================================
    # Generate random numbers directly on GPU
    CS_inv_x = cp.abs(cp.random.normal(0, cp.sqrt(2*cp.pi)*emitX, N_e))
    CS_inv_y = cp.abs(cp.random.normal(0, cp.sqrt(2*cp.pi)*emitY, N_e))
    phase_x = cp.random.rand(N_e)*2*cp.pi
    phase_y = cp.random.rand(N_e)*2*cp.pi

    # Initialize array directly on GPU
    elec0 = cp.zeros((6, N_e), dtype=cp.float64)
    
    # =============================================
    # GPU-accelerated coordinate initialization
    # =============================================
    elec0[4] = (cp.random.rand(N_e) - 0.5) * slicelength
    elec0[5] = cp.random.normal(0, energyspread, N_e)
    
    # Beam optics calculations on GPU
    sqrt_betaX = cp.sqrt(CS_inv_x * betaX)
    elec0[0] = sqrt_betaX * cp.cos(phase_x) + elec0[5] * Dx
    elec0[1] = -cp.sqrt(CS_inv_x / betaX) * (alphaX * cp.cos(phase_x) + cp.sin(phase_x)) + elec0[5] * Dxprime
    
    sqrt_betaY = cp.sqrt(CS_inv_y * betaY)
    elec0[2] = sqrt_betaY * cp.cos(phase_y)
    elec0[3] = -cp.sqrt(CS_inv_y / betaY) * (alphaY * cp.cos(phase_y) + cp.sin(phase_y))

    # =============================================
    # Special particles for diagnostics (GPU)
    # =============================================
    # Last 6 particles are special markers
    indices = cp.array([-6, -5, -4, -3, -2, -1])
    
    # Set positions and momenta
    elec0[0, indices] = cp.array([0.0, R51_dx, 0.0, 0.0, 0.0, 0.0], dtype=cp.float64)
    elec0[1, indices] = cp.array([0.0, 0.0, 0.0, R52_dxp, 0.0, 0.0], dtype=cp.float64)
    elec0[5, indices] = cp.array([0.0, 0.0, 0.0, 0.0, 0.0, R56_dE], dtype=cp.float64)

    # =============================================
    # GPU-optimized coordinate transformation
    # =============================================
    elec = cp.zeros((6, N_e), dtype=cp.float64)
    p_elecs = cp.sqrt(((1+elec0[5])*E0)**2 - (m_e**2*c**4)) / c
    tan_vals = cp.tan(elec0[3])
    
    elec[5] = p_elecs / cp.sqrt(1/cp.cos(elec0[1])**2 + tan_vals**2)
    elec[4] = elec[5] * tan_vals
    elec[3] = elec[5] * cp.tan(elec0[1])
    
    # Position coordinates
    elec[0] = elec0[0]
    elec[1] = elec0[2]
    elec[2] = elec0[4]

    # Save directly in GPU format
    cp.save("e_dist_gpu.npy", elec)
    
    return elec


class LaserGPU:
    def __init__(self, wl, sigx, sigy, pulse_len, pulse_E, focus, 
                X0=0.0, Z_offset=0, M2=1.0, pulsed=True, D2=0, D3=0):
        # Convert all parameters to GPU-compatible types
        self.wl = cp.float64(wl)
        self.sigx = cp.float64(sigx)
        self.sigy = cp.float64(sigy)
        self.pulse_len = cp.float64(pulse_len)
        self.E = cp.float64(pulse_E)
        self.focus = cp.float64(focus)
        self.X0 = cp.float64(X0)
        self.Z_offset = cp.float64(Z_offset)
        self.M2 = cp.float64(M2)
        self.D2 = cp.float64(D2)
        self.D3 = cp.float64(D3)
        self.pulsed = pulsed
        
        self.P_max = self.E / cp.float64(0.94 * self.pulse_len)
        I0 = cp.float64((2 * self.P_max) / (cp.pi * 4 * sigx * sigy))  # Peak intensity
        self.E0 = cp.sqrt(2 * Z0 * I0)  # Peak electric field

        # Precompute constants on GPU
        self.k = cp.float64(2 * cp.pi / self.wl)
        self.omega0 = cp.float64(2 * cp.pi * c / self.wl)
        self.sigz = cp.sqrt(cp.float64(2)) * self.pulse_len * c / cp.float64(2.3548)
        
        # Rayleigh lengths
        self.zRx = cp.pi * (cp.float64(2) * self.sigx)**2 / (self.M2 * self.wl)
        self.zRy = cp.pi * (cp.float64(2) * self.sigy)**2 / (self.M2 * self.wl)

        self.beamsize_x = lambda z: self.sigx * cp.sqrt(1 + (z / self.zRx) ** 2)
        self.beamsize_y = lambda z: self.sigy * cp.sqrt(1 + (z / self.zRy) ** 2)
        
        # Precompute temporal profile on GPU
        self.t_array, self.E_temporal = self._precompute_temporal_profile_gpu()

    def _precompute_temporal_profile_gpu(self):
        """GPU-accelerated temporal profile calculation"""
        # Use CuPy constants and math functions
        sqrt_2 = cp.sqrt(cp.float64(2))
        log_2 = cp.log(cp.float64(2))
        
        # Calculate sigma_t using GPU math
        sigma_t = self.pulse_len / cp.float64(2 * cp.sqrt(2 * cp.log(2)))
        time_window = cp.float64(16) * sigma_t
        Nt = 16384

        t = cp.linspace(-time_window/2, time_window/2, Nt, dtype=cp.float64)
        dt = t[1] - t[0]

        if not self.pulsed:
            E_temporal = 1
            # Add the logic here
        else:
            # Gaussian envelope with carrier frequency
            envelope = cp.exp(-t**2 / (cp.float64(2) * sigma_t**2))
            E_time = envelope * cp.exp(1j * self.omega0 * t)
    
            # FFT on GPU
            E_freq = cp.fft.fftshift(cp.fft.fft(E_time))
            omega = cp.float64(2 * cp.pi) * cp.fft.fftshift(cp.fft.fftfreq(Nt, dt))
    
            # Dispersion terms
            delta_omega = omega - self.omega0
            phase = (cp.float64(0.5) * self.D2 * delta_omega**2 + 
                    cp.float64(1/6) * self.D3 * delta_omega**3)
    
            # Apply phase and inverse FFT
            E_freq_phased = E_freq * cp.exp(-1j * phase)
            E_time_phased = cp.fft.ifft(cp.fft.ifftshift(E_freq_phased))

            # Normalize to preserve energy (not peak amplitude)
            original_energy = cp.sum(cp.abs(E_time) ** 2) * dt
            new_energy = cp.sum(cp.abs(E_time_phased) ** 2) * dt
            E_temporal = cp.real(E_time_phased) * np.sqrt(original_energy / new_energy)
        
        return t, E_temporal 

    def E_field(self, X, Y, Z, T):
        """GPU-accelerated electric field calculation"""
        # Convert all inputs to CuPy arrays if not already
        X = cp.asarray(X, dtype=cp.float64)
        Y = cp.asarray(Y, dtype=cp.float64)
        Z = cp.asarray(Z, dtype=cp.float64)
        T = cp.asarray(T, dtype=cp.float64)

        # Beam parameters relative to focus
        Zdif_x = Z - self.focus
        Zdif_y = Z - self.focus
        X = X - self.X0

        # Spatial envelope
        central_E_field = self.E0 * self.sigx / self.beamsize_x(Zdif_x)
        spatial_factor = cp.exp(
            -(Y / self.beamsize_y(Zdif_y)) ** 2
            - (X / self.beamsize_x(Zdif_x)) ** 2)

        # Temporal profile
        tau = T - (Z + self.Z_offset)/c
        temporal_factor = cp.interp(tau, self.t_array, self.E_temporal)

        # Phase terms (vectorized)
        R_x = Zdif_x * (cp.float64(1) + (self.zRx/Zdif_x)**2)
        R_y = Zdif_y * (cp.float64(1) + (self.zRy/Zdif_y)**2)
        
        phase = cp.cos(
            -self.k * (X**2)/(cp.float64(2)*R_x) -
            self.k * (Y**2)/(cp.float64(2)*R_y) +
            cp.arctan(Zdif_x/self.zRx) +
            cp.arctan(Zdif_y/self.zRy)
        )
     
        return central_E_field * spatial_factor * temporal_factor * phase

    

class LatticeGPU:
    def __init__(self, E0= 1492,l1= 800e-9, l2= 400e-9, h= 4, c1= 800, c2= 800, filename = None, is2d=False, plot = True):
        
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
        
        self.B_func = cupy_interp1d(cp.asarray(self.l), cp.asarray(self.b))


def lsrmod_track_gpu(Mod, Lsr, e_bunch, Lsr2=None, tstep=1e-12, zlim=None,
                    plot_track=False, disp_Progress=True, 
                    get_R512=True, R51_dx=4e-4, R52_dxp=4e-5):
    
    # Convert input to GPU array if not already
    if not isinstance(e_bunch, cp.ndarray):
        bunch = cp.asarray(e_bunch, dtype=cp.float64)
    else:
        bunch = cp.copy(e_bunch)
    N_e = bunch.shape[1]
    
    # Constants on GPU
    m_e_gpu = cp.float64(const.m_e)
    c_gpu = cp.float64(const.c)
    e_charge_gpu = cp.float64(const.e)
    
    # Initialize tracking variables on GPU
    z_0 = cp.mean(bunch[2])
    bunch[2] -= z_0
    progressrate = 10
    progress = 0
    t = 0.0
    
    # GPU memory pre-allocation
    track_x = [cp.copy(bunch[0, -6:])]
    track_z = [cp.copy(bunch[2, -6:])]
    R51, R52 = [0.0], [0.0]
    
    # Unified memory for magnetic field
    if zlim is None:
        zlim = cp.float64(Mod.len)
    z_target = z_0 + zlim
    
    # Precompute gamma conversion factor
    inv_mc = 1 / (m_e_gpu * c_gpu)
    
    # Main tracking loop
    starttime = time()
    while cp.mean(bunch[2]) < z_target:
        # Progress reporting (minimal CPU sync)
        if disp_Progress and progress < (cp.mean(bunch[2]).get() / zlim * progressrate):
            progress += 1
            elapsed = time() - starttime
            sys.stdout.write('\r Progress: ' + str(progress) + '/' + str(progressrate) + " \t ETA: " + str(np.round(elapsed/60 * (progressrate / (progress+0.01) - 1), 2)) + " mins ")
            sys.stdout.flush()
        
        # Current coordinates
        z = bunch[2]
        p = bunch[3:]
        
        # Field calculations
        E = Lsr.E_field(bunch[0], bunch[1], z, t)
        if Lsr2 is not None:
            E += Lsr2.E_field(bunch[0], bunch[1], z, t)
        
        B = Mod.B_func(cp.stack([bunch[0], z], axis=1)).T if Mod.is2d else Mod.B_func(z)
        B += E / c_gpu

        # Lorentz force calculation (vectorized)
        gamma = cp.sqrt(1 + cp.sum(p**2, axis=0) * inv_mc**2)
        dp = cp.empty_like(p)
        dp[0] = (E - p[2]*B/(m_e_gpu*gamma)) * e_charge_gpu * tstep
        dp[1] = cp.zeros(N_e, dtype=cp.float64)
        dp[2] = p[0]*B/(m_e_gpu*gamma) * e_charge_gpu * tstep
        
        # Update momentum and position
        p_new = p + dp
        gamma_new = cp.sqrt(1 + cp.sum(p_new**2, axis=0) * inv_mc**2)
        bunch[3:] = p_new
        bunch[:3] += p_new / (m_e_gpu * gamma_new) * tstep

        track_x.append(np.copy(bunch[0][-6:]))
        track_z.append(np.copy(bunch[2][-6:]))
        
        t += tstep

    if plot_track == True:
        track_x = cp.array(track_x)
        track_z = cp.array(track_z)
        return(bunch,track_x,track_z)
    
    if disp_Progress:
        print(f'\nProgress: {progress}/{progressrate}')
    
    print(f"\nRuntime: {time()-starttime:.2f} sec")
    return bunch


def calc_phasespace(bunch,e_E,plot=False):
    p = cp.sqrt(cp.sum(bunch[3:]**2 , axis=0))
    E = cp.sqrt(m_e**2 * c**4 + p**2 * c**2)
    dEE = E/e_E - 1
    z = cp.copy(bunch[2,:]) 
    
    if plot:
        plt.figure()
        plt.plot((z-np.mean(z)).get() * 1e6 , dEE.get() ,',')
        plt.xlabel('z ($\mu m$)')
        plt.ylabel('$\Delta E/E_0$')
        plt.tight_layout()
    return(z,dEE)   


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
    