# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:58:02 2024

@author: arjun
"""
import numpy as np
import pandas as pd
import sys
from time import time
import scipy.constants as const
from scipy import special
from SPEED_functions import coord_change

##### natural constants #####
c = const.c                     # speed of light
e_charge = const.e              # electron charge
m_e = const.m_e                 # electron mass in eV/c^2
Z0 = 376.73                     # impedance of free space in Ohm
epsilon_0 = const.epsilon_0     # vacuum permittivity
mu0 = const.mu_0                # vacuum permeability


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
    # rotated_p += reference_particle[:, np.newaxis]

    return rotated_p


def lsrmod_track(Mod, Lsr, e_bunch, Lsr2=None, tstep=1e-12, zlim=None, plot_track=False, disp_Progress=True,
                 get_R512=True, R51_dx=4e-4, R52_dxp=4e-5):

    N_e = len(e_bunch[0])
    bunch = np.copy(e_bunch)
    z_0 = np.mean(bunch[2])
    bunch[2] -= z_0
    z_mean = np.mean(bunch[2]) 
    
    progressrate = 10
    progress = 0 
    t = 0

    starttime = time()
    EE = []
    track_x = [np.copy(bunch[0][-6:])]
    track_z = [np.copy(bunch[2][-6:])]

    R51, R52 = [0.0], [0.0]

    if zlim == None:
        zlim = Mod.len
    
    while z_mean < z_0 + zlim:
        if disp_Progress:
            if progress < (z_mean) / zlim * progressrate:
                elapsed = time() - starttime
                sys.stdout.write('\r Progress: ' + str(progress) + '/' + str(progressrate) + " \t ETA: " + 
                                 str(np.round(elapsed/60 * (progressrate / (progress+0.01) - 1), 2)) + " mins ")
                sys.stdout.flush()
                progress += 1
    
        z = np.copy(bunch[2])
        z_mean = np.mean(z)
        
        Efield_x_vec = Lsr.E_field(bunch[0], bunch[1], bunch[2], t)
        if Lsr2 != None: 
            Efield_x_vec += Lsr2.E_field(bunch[0], bunch[1], bunch[2],t)
        EE.append(Efield_x_vec[0])

        try:
            Bfield_y_vec = Mod.B_func(z) + Efield_x_vec / c
        except:
            Bfield_y_vec = Efield_x_vec / c
        
        p_field = bunch[3:]
        p_vec = np.sqrt(np.sum(p_field**2, axis=0))
        gamma_vec = np.sqrt((p_vec / m_e / c) ** 2 + 1)

        dp_x_vec = (Efield_x_vec - p_field[2] * Bfield_y_vec / m_e / gamma_vec) * e_charge * tstep
        dp_y_vec = np.zeros(N_e)
        dp_z_vec = p_field[0] * Bfield_y_vec / m_e / gamma_vec * e_charge * tstep

        p_new = bunch[3:] + [dp_x_vec , dp_y_vec , dp_z_vec]
        p_vec_new = np.sqrt(np.sum(p_new**2 , axis=0))
        gamma_vec_new = np.sqrt((p_vec_new / m_e / c)**2 + 1)    
                       
        spatial_new = bunch[0:3,:] + p_new / m_e / gamma_vec_new * tstep

        comoving_angle = p_new[0, -1] / p_new[2, -1]
        rotated_pos = rotate_particles(spatial_new, -comoving_angle)

        R51.append((rotated_pos[2, -6] - rotated_pos[2, -5]) / R51_dx)
        R52.append((rotated_pos[2, -4] - rotated_pos[2, -3]) / R52_dxp)

        bunch[0:3] = np.copy(spatial_new)
        bunch[3:] = np.copy(p_new)
        
        t += tstep
        
        track_x.append(np.copy(bunch[0][-6:]))
        track_z.append(np.copy(bunch[2][-6:]))

    if disp_Progress:
            print('Progress: '+str(progress)+'/'+str(progressrate))

    endtime = time()
    print("\nRuntime:  " , np.round(endtime-starttime,2) , " sec")

    if plot_track == True:
        track_x = np.array(track_x)
        track_z = np.array(track_z)
        return(bunch,track_x,track_z)

    if get_R512 == True:
        return bunch, np.array(R51), np.array(R52), np.array(track_z)[:, -1]

    return bunch  
        
def chicane_track(bunch_in, R56, R51=0, R52=0, isr=False):
    RM = pd.read_csv("TM.txt", usecols=range(1, 7))
    RR = np.array(RM)
    RR[4, 0], RR[4, 1] = R51, R52    
    RR[4, 5] = R56

    # pp = np.sum(bunch_in[3:]**2)**0.5
    pp = np.linalg.norm(bunch_in[3:])
    dE = np.sqrt((pp**2 * c**2) + (m_e**2 * c**4)) / e_charge - 1492e6
    MM = np.array([
        [bunch_in[0]], 
        [np.arctan(bunch_in[3] / bunch_in[5])], 
        [bunch_in[1]], 
        [np.arctan(bunch_in[4] / bunch_in[5])], 
        [bunch_in[2]], 
        [dE / 1492e6]
        ])
    
    p_mod = MM.transpose((2, 0, 1))
    p_end = np.matmul(RR, p_mod)
    elec_dummy = p_end.transpose((2, 1, 0))[0]
    # convert to parameter style: [x,y,z,px,py,pz] in laboratory frame
    bunch_out = coord_change(elec_dummy)
    return bunch_out

def calc_R56(A11, A22, dE=7e-4, K=2, m=21, n=-1, wl=800e-9):
    A1, A2 = A11 / dE, A22 / dE
    B2 = (m + (0.81 * m**(1 / 3))) / ((K * m + n) * A2)
    R56_2 = B2 / (2 * np.pi / wl) / dE  # optimal R56(2)
    
    print("\nOptimum R56 values:")
    print("R56(2) =", np.round(R56_2 * 1e6, 2), " microns")
    
    R56_list = np.linspace(50e-6, 2000e-6, 1000)
    bn = []
    for R in R56_list:
        B1 = R * (2 * np.pi / wl) * dE
        bn.append(abs(special.jv(m, -(K * m + n) * A2 * B2) * special.jv(n, (A1 * (n * B1 + ((K * m + n) * B2)))) * np.exp(-0.5 * (n * B1 + (K * m + n) * B2)**2)))
    
    i = np.argmax(bn)
    r1 = R56_list[i]
    bn[i] = 0
    i = np.argmax(bn)
    r2 = R56_list[i]
    R56_1 = r2 if r2 > r1 else r1  # optimal R56(1)
    print("R56(1) =", np.round(R56_1 * 1e6), " microns")
    return R56_1, R56_2
    