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
from  numba import jit
import pathlib
import sys
#import sdds


##### natural constants #####
c=const.c       # speed of light
e_charge=const.e # electron charge
m_e=const.m_e      # electron mass in eV/c^2
Z0=376.73          # impedance of free space in Ohm
epsilon_0= const.epsilon_0 # vacuum permittivity
mu0= const.mu_0

#e_E=1.492e9*e_charge   # electron energy in J
#e_E=1.0e9*e_charge
#e_gamma=e_E/m_e/c**2 # Lorentz factor

@jit(parallel=True)
def calc_bn(tau0,wl):
    bn=[]
    for i in range(len(wl)):
        z=np.sum(np.exp(-1j*2*np.pi*(tau0/wl[i])))
        bn.append(abs(z)/len(tau0))
    return(np.array(bn))

def plot_slice(z,wl,slice_len=0,n_slice=40):
    
    if slice_len != 0:
        n_slice= int((max(z)-min(z))/slice_len)
        
    zz=np.linspace(min(z),max(z),n_slice)
    i=1
    bn,z_slice=[],[]
    while(i<len(zz)):
        z1,z2=zz[i-1],zz[i]
        z_slice.append(np.mean([z1,z2]))
        slice_zz=z[(z>=z1)*(z<z2)]
        if len(slice_zz)==0:
            bn.append(0)
        else:
            bn.append(max(calc_bn(slice_zz,wl)))
        i+=1
    plt.figure()
    plt.plot(z_slice,bn)

def write_results(bunch,file_path):
    print("Writing to "+file_path+" ...")
    file=pathlib.Path(file_path)
    if file.is_file():
        ch=input("The file already exist! Overwrite? (Y/N)")
        if ch=='y':
            bunch.to_csv(file_path)
    else:
        bunch.to_csv(file_path)
        
class Laser:
    def __init__(self,wl,sigx,sigy,pulse_len,pulse_E,focus,X0=0.0,Z_offset=0,M2=1.0,pulsed=True,phi=0):
        self.wl= wl     # wavelength in m
        self.sigx= sigx # sigma width of horizontal focus in m
        self.sigy= sigy # sigma width of vertical focus in m
        self.pulse_len= pulse_len # FWHM pulse lengths in s
        self.E= pulse_E # pulse energy in J
        self.P_max=self.E/(0.94*self.pulse_len)
        self.M2= M2
        self.k= 2*np.pi/self.wl # wavenumber in 1/m
        self.omega= 2*np.pi*c/self.wl # angular frequency in rad/s
        self.sigz= np.sqrt(2)*self.pulse_len*c/2.3548 # sigma width of pulse length in m
        self.zRx=np.pi*(2*self.sigx)**2/(self.M2*self.wl) # horizontal Rayleigh length in m
        self.zRy=np.pi*(2*self.sigy)**2/(self.M2*self.wl) # vertical Rayleigh length in m
        self.beamsize_x=lambda z: self.sigx*(np.sqrt(1+z**2/(self.zRx**2))) # horizontal beam size at position z in m
        self.beamsize_y=lambda z: self.sigy*(np.sqrt(1+z**2/(self.zRy**2))) # vertical beam size at position z in m
        self.E0= 2**-0.25*np.pi**-0.75*np.sqrt(Z0*self.E/(self.sigx*self.sigy*self.sigz/c)) * 1.2 # factor to make the modulation amplitude equal to elegant simulations
        #self.w0=2*sigx
        #self.E0= np.sqrt((4*self.P_max*Z0)/(np.pi*self.w0**2))
        self.X0 = X0
        self.Z_offset= Z_offset
        self.focus=focus
        self.pulsed=pulsed
        self.phi= phi   # spectral phase of the laser
        
    def E_field(self,X,Y,Z,T):
        Zdif_x = Z-self.focus # Distance of electron to focus (mod1_center)
        Zdif_y = Z-self.focus
        X = X- self.X0
        Zlas= c*T - self.Z_offset  # Position of the laser pulse center
        R_x=Zdif_x*(1+(self.zRx/Zdif_x)**2)
        R_y=Zdif_y*(1+(self.zRy/Zdif_y)**2)
        central_E_field=self.E0*self.sigx/(self.beamsize_x(Zdif_x))
        if self.pulsed:
            offaxis_pulsed_factor=np.exp(-(Y/self.beamsize_y(Zdif_y))**2-(X/self.beamsize_x(Zdif_x))**2-((Z-Zlas)/(2*self.sigz))**2)
        else:
            offaxis_pulsed_factor=np.exp(-(Y/self.beamsize_y(Zdif_y))**2-(X/self.beamsize_x(Zdif_x))**2)
        phase=np.cos(self.k*(Z)+(self.phi*(const.c*T-Z)**2)-self.omega*T-self.k/2*X**2/R_x-self.k/2*Y**2/R_y)#+np.arctan(Zdif/l1_zRx))
        return (central_E_field*offaxis_pulsed_factor*phase)
        #return (central_E_field*phase)
        
class Lattice:
    def __init__(self, E0= 1500,l1= 800e-9, l2= 400e-9, h= 4, c1= 800, c2= 800, filename = None, plot = True):
        self.len= 5.74925
        windings=48 # upper plus lower coil
        gap=0.05 
        period=0.25
        yoke=0.08
        edge=0.02    # softness parameter for magnet edge
        drift=0.5   # befor and after magnetic structure
        dl=0.0005   # longitudinal interval
        E0= E0    # beam energy in MeV
        e_gamma = E0/0.511
        
        if filename != None:
            df=pd.read_csv(filename,sep='\t')
            self.l=np.array(df['z'])/1000 #[i_0:i_lim] #[:3220]
            self.b=np.array(df['By']) #[i_0:i_lim] #[:3220]
            self.len= self.l[-1]        
            self.B_func= interpolate.interp1d(self.l,self.b)
            if plot == True:
                plt.plot(self.l,self.b)
            return
        
        if l1 == 0:
            IM1= 0
        else:
            laser_wl= l1
            K= np.sqrt(4*laser_wl*e_gamma**2/period-2);
            B = 2*np.pi*K*m_e*c/(e_charge*period);
            factor=mu0*windings/gap;
            IM1 =  1.365* B/factor;  # current for first modulator

        if l2 == 0:
            IM2= 0
        else:        
            laser_wl= l2
            K= np.sqrt(4*laser_wl*e_gamma**2/period-2);
            B = 2*np.pi*K*m_e*c/(e_charge*period);
            factor=mu0*windings/gap;
            IM2 =  1.365* B/factor;  # current for second modulator
        
        if h == 0:
            IR1= 0
        else:        
            laser_wl= 800e-9/h
            K= np.sqrt(4*laser_wl*e_gamma**2/period-2);
            B = 2*np.pi*K*m_e*c/(e_charge*period);
            factor=mu0*windings/gap;
            IR1 =  1.365* B/factor;  # current for radiator
        

        ID1 = IM1/2
        ID2 = IM2/2
        ID3 = IM2/2
        ID4 = IR1/2
        IC1= c1     # power supply for chicane 1 ("Danfysik")
        IC2= c2     # power supply for chicane 2 ("Danfysik")
    
        # coil current
        curr=[ -ID1 , IM1 , -IM1 ,  IM1 , -IM1 , IM1 , -IM1 , IM1 , -IC1 , -IC1, -ID1, \
               IC1 , IC1 , IC1 , IC1 , -ID2 , -IC1 , -IC1 , \
               IM2 , -IM2 , IM2 , -IM2 , IM2 , -IM2 , IM2 , -IC2 , -ID3 , IC2 , IC2 , -ID4 , -IC2 , \
               IR1 , -IR1 , IR1 , -IR1 , IR1 , -IR1 , ID4 ]

        # fraction of full coil
  #       coil=[-0.50, 1.00,-1.00, 1.00,-1.00, 1.00,-1.00, 1.00,-1.00,-1.00,\
  #             -0.50, 1.00, 1.00, 1.00, 1.00,-0.50,-1.00,-1.00, 1.00,\
  #             -1.00, 1.00,-1.00, 1.00,-1.00, 1.00,-1.00,-0.50, 1.00,\
  #              1.00,-0.50,-1.00, 1.00,-1.00, 1.00,-1.00, 1.00,-1.00, 0.50]
        print("test")    
        factor=mu0*windings/gap
        b0= np.zeros(len(curr))
        for m in range(len(curr)):
            b0[m]=factor*curr[m]
            
        nm=len(curr) 
        magnet=period/2
        magnet2=magnet/2
        yoke2=yoke/2
        len_range=nm*magnet+2*drift
          
        nn=magnet/dl
        n2=nn/2
        nl=int(len_range/dl)
        nd=int(drift/dl)
        
        self.l= np.zeros(nl)
        for k in range(nl):
            self.l[k]=(k-0.5)*dl
        
        self.b=np.zeros(nl)
        
        for m in range(nm):
            n1=(m-1)*nn
            l1=drift+(m-1)*magnet  # magnet center
            for k in range(nl):
                self.b[k]=self.b[k]+b0[m]/(np.exp((l1+magnet2-yoke2-self.l[k])/edge)+1);
            
            for k in range(nl):
                self.b[k]=self.b[k]+b0[m]/(np.exp((self.l[k]-magnet2-yoke2-l1)/edge)+1);
        if plot:
            plt.plot(self.l,self.b)

        self.B_func= interpolate.interp1d(self.l,self.b)
        #return(B_func)


def calc_phasespace(bunch,e_E,plot=False):
    p=np.sqrt(np.sum(bunch[3:]**2,axis=0))
    E=np.sqrt(m_e**2*c**4+p**2*c**2)
    dEE=E/e_E-1
    z=np.copy(bunch[2,:]) 
    
    if plot:
        plt.figure()
        plt.plot((z-np.mean(z))*1e6,dEE,',')
        plt.xlabel('z ($\mu m$)')
        plt.ylabel('$\Delta E/E_0$')
        plt.tight_layout()
    return(z,dEE)    

def define_bunch(Test,N=1e4,slicelength=8e-6,E0=1.492e9*e_charge,dE=7e-4,R56_dE=0.0007):
    N_e=int(N) # number of electrons
    
    ##### electron parameter #####
    e_E=E0   # electron energy in J
    energyspread= dE 
       
    alphaX=8.811383e-01 #1.8348
    alphaY=8.972460e-01 #0.1999
    betaX=13.546
    betaY=13.401
    emitX= 1.6e-8
    emitY= 1.6e-9   
    Dx= 0.0894
    Dxprime= -4.3065e-9 
    
    if(Test):
        slicelength=40e-6 
        N_e=int(1e4)
        energyspread= 0e-4
        emitX= 0
        emitY= 0  
        Dx= 0
        Dxprime= 0
        
    CS_inv_x=np.abs(np.random.normal(loc=0,scale=emitX*np.sqrt(2*np.pi),size=N_e))
    CS_inv_y=np.abs(np.random.normal(loc=0,scale=emitY*np.sqrt(2*np.pi),size=N_e))
    phase_x=np.random.rand(N_e)*2*np.pi
    phase_y=np.random.rand(N_e)*2*np.pi
    
    # generate random electron parameters according to beam parameters
    elec0=np.zeros((6,N_e))
    elec0[4]=(np.random.rand(1,N_e)-0.5)*slicelength#/c   
    elec0[5]=np.random.normal(loc=0,scale=energyspread,size=N_e)#/e_m/c**2
    elec0[0]=(np.sqrt(CS_inv_x*betaX)*np.cos(phase_x))+elec0[5,:]*Dx
    elec0[1]=-(np.sqrt(CS_inv_x/betaX)*(alphaX*np.cos(phase_x)+np.sin(phase_x)))+elec0[5,:]*Dxprime
    elec0[2]=(np.sqrt(CS_inv_y*betaY)*np.cos(phase_y))
    elec0[3]=-(np.sqrt(CS_inv_y/betaY)*(alphaY*np.cos(phase_y)+np.sin(phase_y)))
    
    # Adding two particles with only energy difference to calculate the R56
    for i in range(6):
        if i==5:
            elec0[i][-2],elec0[i][-1] = 0, R56_dE
        else:
            elec0[i][-2],elec0[i][-1] = 0, 0.0
            
    #changing to parameter style: [x,y,z,px,py,pz] in laboratory frame
    elec=coord_change(elec0,e_E)
    np.save("e_dist.npy",elec)
    return(elec)

def coord_change(elec_dummy,e_E):
    elec=np.zeros((6,len(elec_dummy[0])))
    elec[0,:]=elec_dummy[0,:]
    elec[1,:]=elec_dummy[2,:]
    elec[2,:]=elec_dummy[4,:]
    p_elecs=np.sqrt(((1+elec_dummy[5,:])*e_E)**2-m_e**2*c**4)/c
    elec[5,:]=p_elecs/(np.sqrt(1/np.cos(elec_dummy[1,:])**2+np.tan(elec_dummy[3,:])**2))
    elec[4,:]=elec[5,:]*np.tan(elec_dummy[3,:])
    elec[3,:]=elec[5,:]*np.tan(elec_dummy[1,:])
    return(elec)


    
def lsrmod_track(Mod,Lsr,e_bunch,Lsr2=None,tstep=1e-12,zlim=None,plot_track=False,disp_Progress=True):
    N_e= len(e_bunch[0])
    bunch=np.copy(e_bunch)
    z_0=np.mean(bunch[2])
    #print(z_0)
    
    bunch[2]-=z_0
    z_mean=np.mean(bunch[2]) 
    #z_0=np.copy(z_mean)
    count=0
    progressrate=10
    progress=0 
       
    t= 0 #z_0/const.c
    starttime=time()
    EE=[]
    track_x=[[],[]]
    track_z=[[],[]]
    dZZ=[]
    ZZ=[]
    
    if zlim == None:
        zlim = Mod.len
    
    while z_mean<z_0+zlim:
        if disp_Progress:
            if progress<(z_mean)/zlim*progressrate:
                sys.stdout.write('\r Progress: '+str(progress)+'/'+str(progressrate))
                sys.stdout.flush()
                progress+=1
    
        z=np.copy(bunch[2])
        z_mean=np.mean(z)
        ZZ.append(z_mean)
        
        p_field=bunch[3:]
        p_vec=np.sqrt(np.sum(p_field**2,axis=0))
        gamma_vec=np.sqrt((p_vec/m_e/c)**2+1)
    
        Efield_x_vec=Lsr.E_field(bunch[0],bunch[1],bunch[2],t)#+l2_Efield(bunch[0,:]-mod2_path_offset,bunch[1,:],bunch[2,:],t)
        
        if Lsr2 != None: # and z_mean>z_0+2.25:
            Efield_x_vec=Lsr.E_field(bunch[0],bunch[1],bunch[2],t) + Lsr2.E_field(bunch[0],bunch[1],bunch[2],t)
            #Efield_x_vec=Lsr2.E_field(bunch[0],bunch[1],bunch[2],t)
        EE.append(Efield_x_vec[0])

        try:
            Bfield_y_vec=Mod.B_func(z)+Efield_x_vec/c
        except:
            Bfield_y_vec=Efield_x_vec/c
    
        dp_x_vec=(Efield_x_vec-p_field[2]*Bfield_y_vec/m_e/gamma_vec)*e_charge*tstep
        dp_y_vec=np.zeros(N_e)
        dp_z_vec=p_field[0]*Bfield_y_vec/m_e/gamma_vec*e_charge*tstep
    
    
        p_new=bunch[3:]+[dp_x_vec,dp_y_vec,dp_z_vec]
        p_vec_new=np.sqrt(np.sum(p_new**2,axis=0))
        gamma_vec_new=np.sqrt((p_vec_new/m_e/c)**2+1)                           
        spatial_new=bunch[0:3,:]+p_new/m_e/gamma_vec_new*tstep   
    
        bunch[0:3]=np.copy(spatial_new)
        bunch[3:]=np.copy(p_new)
        
        t+=tstep
        count+=1
        track_x[0].append(bunch[0][-2])
        track_z[0].append(bunch[2][-2])
        track_x[1].append(bunch[0][-1])
        track_z[1].append(bunch[2][-1])
        
        dz=(t*c-np.mean(bunch[2]))
        dZZ.append(dz)
  
    else:
        if disp_Progress:
            print('Progress: '+str(progress)+'/'+str(progressrate))
    
    # plt.figure()
    # plt.plot(ZZ,EE)
    # plt.title("track")
    # plt.xlabel('z')

    if plot_track == True:
        return(bunch,track_x,track_z)
    
    endtime=time()
    if disp_Progress:
        print("\nRuntime:  ",endtime-starttime)
    return(bunch)  
        
def chicane_track(bunch_in,R56,R51=0,R52=0,isr=False):
    RM=pd.read_csv("TM.txt", usecols=range(1,7))
    RR=np.array(RM)
    RR[4,0],RR[4,1]=R51,R52    
    RR[4,5]=R56
    
    pp=sum(bunch_in[3:]**2)**0.5
    dE=((pp**2*c**2)+(m_e**2*c**4))**0.5/e_charge-1492e6
    MM = np.asarray([[bunch_in[0]],[np.arctan(bunch_in[3]/bunch_in[5])],[bunch_in[1]],[np.arctan(bunch_in[4]/bunch_in[5])],[bunch_in[2]],[dE/1492e6]])
    p_mod= MM.transpose((2,0,1))
    p_end= np.matmul(RR,p_mod)
    elec_dummy=p_end.transpose((2,1,0))[0]   
    # convert to parameter style: [x,y,z,px,py,pz] in laboratory frame
    bunch_out=coord_change(elec_dummy)
    return(bunch_out)

def calc_R56(A11,A22,dE=7e-4,K=2,m=21,n=-1,wl=800e-9):
    A1,A2= A11/dE, A22/dE
    B2=(m+(0.81*m**(1/3)))/((K*m+n)*A2)
    R56_2= B2/(2*np.pi/wl)/dE
    rr2=R56_2                       #optimal R56(2)
    print("R56(2)= ",R56_2)
    R56_1= np.linspace(50e-6,2000e-6,1000)
    #R56_2= np.linspace(10e-6,85e-6,500)
    #RM=[]
    B2=R56_2*(2*np.pi/wl)*dE
    bn=[]
    for R in R56_1:
        B1=R*(2*np.pi/wl)*dE
        #B2=R56_2*(2*np.pi/800e-9)*0.0007
        bn.append(abs(special.jv(m,-(K*m+n)*A2*B2)*special.jv(n,(A1*(n*B1+((K*m+n)*B2))))*np.exp(-0.5*(n*B1+(K*m+n)*B2)**2)))
    
    bmax=max(bn)
    i=bn.index(bmax)
    r11=R56_1[i]
    bn[i]=0
    bmax=max(bn)
    i=bn.index(bmax)
    r12=R56_1[i]
    if (r12>r11):
        rr1=r12
    else:
        rr1=r11                     #optimal R56(1)
    print("R56(1)= ",rr1)
    return(rr1,rr2)
    
'''    
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
'''

