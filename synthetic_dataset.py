"""  
Created on Mon Mar 25 09:22:05 2024

@author: giovannidisarra
"""

"""
This code simulates the spiking activity of a grid cell module and run persistent cohomology analysis on 
the simulated data (as in Gardner et al. 2022). 
The trajectory of the mouse is imported by Gardner et al. 2022 data.
The rate map is generated on a hexagonal grid with gaussian bumps and the spike train 
is the poisson independent process sampled from the rate map.

The spike trains are fed to the persistent cohomology pipeline as in Gardner et al. 2022.

"""
import numpy as np
import matplotlib.pyplot as pl
import math 
import time
import os
from utils import *
from ripser import ripser
from Gamma_computation import make_idealized_barcode_strict
from Gamma_computation import norm_bdistance
from Gamma_computation import compute_Gamma_self
import copy
import opexebo
from scipy.signal import savgol_filter
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import time
import multiprocessing
import multi_sample_funcs as msf
from tqdm import tqdm
import random
import pickle

import warnings
warnings.filterwarnings("ignore")

seed_env = 10
seed = int(seed_env)+int(time.time())
np.random.seed(seed)

print(seed)
CASE = 3

rat_name = 'R'
day_name = 'day1'
mod_name = '1'
sess_name = 'OF'
title = '-'
tiling = 'hexagon'

oscillate = True

#put path to Gardner et al. data
folder_data = 'Toroidal_topology_grid_cell_data/'

Frac_Data_Length = 1/2
n_processes = 7

sigma = 0.12
scale = 1.5/(2*np.pi*sigma**2)
lambda0 = 0.05

folname = str(int(scale))+'_'+str(np.round(sigma,decimals = 2))#+'_'+str(int(seed_env))
if not os.path.exists(folname):
    os.makedirs(folname)

#folname = ''

Ncells = 20
plot = False
spacing_factor = 0.98
sp_threshold = 0.4  #in units of spacing
orientation = 0 

timebin = 10 #in ms
N_Oscillators = 200

if oscillate:
    oscil_const = 1.
    oscil_norm = 0.5884
else:
    oscil_const = 0
    oscil_norm = 1

theta_f = 8#0.0000000001
A_theta_0 = 1
A_dom_freq_1 = 0.5
A_dom_freq_2 = 0.8
A_f_0 = 0.25 #0.0

smoothing = 3
frequensies_min = 1
freqneucies_max = 15

grid_fields_jitter = [np.zeros(20), np.zeros(20)]

x_traj, y_traj, _, t, _ = load_pos(rat_name, sess_name, day_name, bSpeed = True, folder = folder_data, binning = timebin) #cm/s

folder = ''

L_max = max(max(x_traj), max(y_traj))
L_min = min(min(x_traj), min(y_traj))
L = max(L_max, np.absolute(L_min))

spacing = spacing_factor*L
sp_threshold = sp_threshold*spacing

prec = L/100
dx = np.arange(-L,L+prec,prec)
x,y = np.meshgrid(dx,dx)

T_samp = int(t.shape[0]*Frac_Data_Length)
x_traj = x_traj[:T_samp]
y_traj = y_traj[:T_samp]
t = t[:T_samp]

freqs = np.tile(np.logspace(0,1.7,N_Oscillators),(1,t.shape[0])).reshape(t.shape[0],N_Oscillators).T    

#freqs = np.tile(np.random.uniform(frequensies_min,freqneucies_max,N_Oscillators),(1,t.shape[0])).reshape(  t.shape[0],N_Oscillators).T    

A_f = A_f_0/(np.sqrt(freqs))
A_f[71,:] = A_dom_freq_1+A_f[71,:]
A_f[106,:] = A_dom_freq_2+A_f[106,:]

theta = (np.sin(2*np.pi*np.multiply(theta_f,t))) 

oscillations = np.multiply( np.sin(2*np.pi*np.multiply(freqs, t)), A_f)

trains = np.zeros( (t.shape[0], Ncells) ) 
trains_jittered = np.zeros( (t.shape[0], Ncells) ) 

range_x = 3
range_y = 3
offsets_x = np.random.uniform(-range_x+L, range_x-L, size = Ncells)
offsets_y = np.random.uniform(-range_y+L, range_y-L, size = Ncells)

pl.scatter(offsets_x, offsets_y)
pl.show()


#MAIN CODE

if __name__ == '__main__':
    spikes = {}
    paramslist = [ [] for _ in range(Ncells) ]
    for k, (offset_x,offset_y) in enumerate(zip(offsets_x,offsets_y)):
            paramslist[k] = [k,sp_threshold,offset_x,offset_y,tiling,L,spacing,orientation,grid_fields_jitter,scale,sigma,lambda0,x,y,dx,x_traj,y_traj,timebin,t,oscillations,theta,prec,oscil_const,oscil_norm,plot]
    
    print('starting muti-proc pool')
    print('sigma=',sigma,'scale=',scale,'lambda0=',lambda0,'seed=',seed)

    with multiprocessing.Pool(n_processes) as pool: 
         trains = pool.map(msf.sample_spikes,paramslist)
         pool.close()
         pool.join()
    #trains, rate_total = msf.sample_spikes(paramslist[0])
    #mu_x, mu_y = mfs.generate_grid(tiling, L, offset_x, offset_y, spacing, orientation, grid_fields_jitter)
    #mfs.spatial_selectivity(dx, mu_x, mu_y, sigma, scale, sp_threshold, plot = True)
    
    for k in range(Ncells):
        sp_times = msf.train_to_times(trains[k], t, timebin/1000)
        spikes[k] = np.array(sp_times)
        
        T_i = 2387001
        ratemap, x_r, y_r = msf.compute_ratemap(sp_times, x_traj, y_traj, smoothing, t, timebin/1000, folname, plot=False)
        power_normalization = (np.sum(trains[k][:T_i]))
        P_i, f_i = msf.psd(trains[k][:T_i], len(trains[k][:T_i]), timebin/1000, power_normalization)
        area_tot = np.trapz(np.real(P_i)[f_i>0.1])
        P_i = savgol_filter(P_i, 1000, 3)
        fig=pl.figure(figsize=(12,12))
        ax1=fig.add_subplot(111)
        ax1.plot(f_i[(f_i>1) & (f_i<20)], 10*np.log10(P_i[(f_i>1) & (f_i<20)]/area_tot), lw = 1)
        ax1.set_xlabel('frequency [Hz]', fontsize = 60)
        ax1.set_ylabel('PSD [dB]', fontsize = 60)
        #ax1.set_xscale('log')
        #ax1.set_yscale('log')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        #ax1.set_xlim(0.5,20)
        #ax1.set_ylim(4e3, 8e4)
        #ax1.set_ylim(2e2, 8e3)
        leg = ax1.legend(fontsize=65)
        for line in leg.get_lines():
            line.set_linewidth(6.)
        #ax1.set_ylim(0.01, 1e8)
        #ax1.set_xlim(1, 15)
        ax1.tick_params(axis='both',labelsize=60)
        #pl.xticks([1,5,10,15,20],fontsize=60)
        #pl.yticks([-60,-62,-64],fontsize=60)
        #ax1.set_ylim(-65, -58)
        pl.yticks(fontsize=72)
        pl.show()
    #sp_times = msf.train_to_times(trains, t, timebin/1000)
    #spikes = np.array(sp_times)
    #ratemap, x_r, y_r = msf.compute_ratemap(sp_times, x_traj, y_traj, smoothing, t, timebin/1000,plot=True)
    
# =============================================================================
#     power_normalization = (np.sum(trains))
#     P_i, f_i = msf.psd(trains, len(trains), timebin/1000, power_normalization)
#     P_i = savgol_filter(P_i, 600, 3)
#     fig=pl.figure(figsize=(12,12))
#     axis=fig.add_subplot(111)
#     axis.plot(f_i[(f_i>1) & (f_i<20)], 10*np.log10(P_i[(f_i>1) & (f_i<20)]), lw = 1)
#     pl.show()
# =============================================================================
    
    np.savez_compressed(folder+folname+'/case'+str(CASE)+'_spikes'+str(seed), spikes)
    
    #with open(folder+folname+'/params'+str(CASE)+'_'+str(seed)+'.pkl', 'wb') as f:
    #      pickle.dump(paramslist, f)
          
    
    
    Gammas = {}
    casename = folder+folname+'/case'+str(CASE)+'_spikes'+str(seed)+'.npz'
    
    ts = 0.
    sspikes_sim,__,__,__,__ = get_spikes(rat_name, mod_name, day_name, sess_name, bType = None, bSmooth = True, bSpeed = True, folder=folder_data, SIM=True, seed_clus = seed, case = casename, ts=ts )
    
    
    #run persistent cohomology pipeline on simulated data
    bRoll = False
    s = np.random.randint(0,100)
    s = 1
    dim = 6
    ph_classes = [0,1] # Decode the ith most persistent cohomology class
    num_circ = len(ph_classes)
    dec_tresh = 0.99
    metric = 'cosine'
    maxdim = 2
    coeff = 47
    active_times = 15000
    k = 1000
    num_times = 5
    n_points = 1200
    nbs = 800
    sigma0 = 1500
    num_neurons = len(sspikes_sim[0,:])
    
    if bRoll:
        np.random.seed(s)
        shift = np.zeros(num_neurons, dtype = int)
        for n in range(num_neurons):
            shifti = int(np.random.rand()*len(sspikes_sim[:,0]))
            sspikes_sim[:,n] = np.roll(sspikes_sim[:,n].copy(), shifti)
            shift[n] = shifti
            
    print(num_neurons)
    mode='gaussian'
    #print(sspikes)
    
    times_cube = np.arange(0,len(sspikes_sim[:,0]),num_times)
    movetimes = np.sort(np.argsort(np.sum(sspikes_sim[times_cube,:],1))[-active_times:])
    movetimes = times_cube[movetimes]
    
    dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes_sim[movetimes,:]), dim = dim)
    indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k,
                                                n_points, 1, metric)
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
    X = squareform(pdist(dim_red_spikes_move_scaled, metric))
    knn_indices = np.argsort(X)[:, :nbs]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    sigmas, rhos = smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    d = result.toarray()
    d = -np.log(d)
    np.fill_diagonal(d,0)
    
    
    persistence = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles= True, distance_matrix = True)
    
    ideal_torus = make_idealized_barcode_strict(persistence['dgms'])
        
    Gammas[ts] = 1 - norm_bdistance( persistence['dgms'], ideal_torus)
    print(Gammas)
    
    plot_barcode(persistence['dgms'])
    #pl.suptitle('Case '+str(CASE)+' seed '+str(seed)+'\n ts='+str(ts)+' Gamma1:'+str(np.round(Gammas[ts][0], decimals = 3) )+' Gamma2:'+str(np.round(Gammas[ts][1], decimals = 3) ), fontsize = 36)
    #pl.savefig(folder+folname+'/Case'+str(CASE)+'_seed_'+str(seed)+'_ts='+str(ts)+'.png')
    pl.show()
        
    #with open(folder+folname+ '/gammas'+str(CASE)+'_'+str(seed)+'.pkl', 'wb') as f:
    #    pickle.dump(Gammas, f)
