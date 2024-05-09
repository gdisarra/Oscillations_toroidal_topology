#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:19:36 2023

@author: giovannidisarra
"""

import matplotlib.pyplot as pl
import numpy as np 
from utils import *
import pickle
from scipy.signal import savgol_filter
import scipy

import opexebo

def psd(x, xlen, dt):
    powers = np.fft.fft(x) /np.sum(np.absolute(x))
    sp = 1/dt
    frequencies = np.fft.fftfreq(len(x))*sp
    powers = np.real(powers * np.conj(powers))
    return powers, frequencies

modules = [('S', '1', 'OF', ''),
           ('R', '1', 'OF', 'day1'),
           ('R', '1', 'OF', 'day2'),
           ('Q', '1', 'OF', ''),
           ('R', '2', 'OF', 'day1'),
           ('R', '2', 'OF', 'day2'),
           ('Q', '2', 'OF', ''),
           ('R', '3', 'OF', 'day1'),
           ('R', '3', 'OF', 'day2')
           ]
    

folder = 'Toroidal_topology_grid_cell_data/' # directory to data

ts = 0.

Ps = {}
fs = {}
eta_to_thetas ={}
#Aeta = {}
#Atheta = {}
#delta_to_thetas ={}
#delta_to_etas ={}
f_theta ={}
f_eta ={}
    

delta_band = {'R2_day1':[0.5,2.5], 'R3_day1':[0.5, 2.5], 'R1_day2':[0.5, 2.5], 'R2_day2':[0.5,2.5], 'Q1_':[0.5,2.9], 'Q2_':[0.5,2.7], 'S1_':[0.5,2.8], 'R1_day1':[0.5,2.5], 'R3_day2':[0.5,2.4]}
eta_band = {'R2_day1':[2.5,5.9], 'R3_day1':[2.5,5.9], 'R1_day2':[2.5,5.2], 'R2_day2':[2.5,5.2], 'Q1_':[2.9,6], 'Q2_':[2.7,6.5], 'S1_':[2.8,6], 'R1_day1':[2.5,6], 'R3_day2':[2.4,6]}
theta_band = {'R2_day1':[5.9,11],'R3_day1':[5.9,11], 'R1_day2':[5.2,11], 'R2_day2':[5.2,10.5], 'Q1_':[6,12.9], 'Q2_':[6.5,12.9], 'S1_':[6, 13], 'R1_day1':[6,11.5], 'R3_day2':[6, 10.5]}


T_i =[]

s=1
data = []

velocity = 'slow'
threshold = 100
timebin = 1 #ms

print('Power spectra for spike trains binned at '+str(timebin)+' ms, for rat running '+str(velocity)+'er than '+str(threshold)+' cm/s')
typ = None
for rat_name, mod_name, sess_name, day_name in modules:
    if sess_name in ('OF', 'WW'):
        spikes,__,__,__,__ = get_spikes(rat_name, mod_name, day_name, sess_name, bType = typ,
                                             bSmooth = False, bBinned = True, bSpeed = True, folder = folder, ts = ts, speed_thresh = threshold, vel= velocity, binning=timebin)
    T_i.append(spikes.T[1].shape[0])

T_i = min(T_i)


#%%

T_i = 2387001
modules = [#('S', '1', 'OF', ''),
           #('R', '1', 'OF', 'day1'),
           #('R', '3', 'OF', 'day2'),
           #('R', '1', 'OF', 'day2'),
           #('Q', '1', 'OF', ''),
           ('R', '2', 'OF', 'day1'),
           ('R', '2', 'OF', 'day2'),
           ('Q', '2', 'OF', ''),
           ('R', '3', 'OF', 'day1')
           ]

for rat_name, mod_name, sess_name, day_name in modules:
    f_theta[str(rat_name)+str(mod_name)+'_'+str(day_name)] = []
    f_eta[str(rat_name)+str(mod_name)+'_'+str(day_name)] = []
    fs[str(rat_name)+str(mod_name)+'_'+str(day_name)] = []
    eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] =[]
    #Aeta[str(rat_name)+str(mod_name)+'_'+str(day_name)] =[]
    #Atheta[str(rat_name)+str(mod_name)+'_'+str(day_name)] =[]
    #delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] =[]
    #delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)] =[]
    Ps[str(rat_name)+str(mod_name)+'_'+str(day_name)] = []

    
ts =0.
print('minimal common size:', T_i)


for rat_name, mod_name, sess_name, day_name in modules:
    
    
    if sess_name in ('OF', 'WW'):
        spikes,__,__,__,__ = get_spikes(rat_name, mod_name, day_name, sess_name, bType = typ,
                                             bSmooth = False, bBinned = True, bSpeed = True, folder = folder, ts = ts, speed_thresh = threshold, vel= velocity, binning=timebin)
    
    print(spikes.shape)
    spikes = spikes.T[:,:T_i]
    print(spikes[1].shape)
    
    for j,train in enumerate(spikes):
        print(str(j+1)+'/'+str(spikes.shape[0]))
        
        P_i, f_i = psd(train, len(train), timebin/1000)
        area_tot = np.trapz(np.real(P_i)[f_i>0.1])
        
               
        #delta_cond =  (f_i > delta_band[str(rat_name)+str(mod_name)+'_'+str(day_name)][0]) & (f_i < delta_band[str(rat_name)+str(mod_name)+'_'+str(day_name)][1])            
        eta_cond =  (f_i > eta_band[str(rat_name)+str(mod_name)+'_'+str(day_name)][0]) & (f_i < eta_band[str(rat_name)+str(mod_name)+'_'+str(day_name)][1])
        theta_cond = (f_i > theta_band[str(rat_name)+str(mod_name)+'_'+str(day_name)][0]) & (f_i < theta_band[str(rat_name)+str(mod_name)+'_'+str(day_name)][1])
        
        #compute integral under the curve at eta and theta bands
        #area_d = np.trapz(np.real(P_i)[delta_cond], f_i[delta_cond])
        area_e = np.trapz(np.real(P_i)[eta_cond], f_i[eta_cond])
        area_t = np.trapz(np.real(P_i)[theta_cond], f_i[theta_cond])
        
        #Aeta[str(rat_name)+str(mod_name)+'_'+str(day_name)].append( area_e)
        #Atheta[str(rat_name)+str(mod_name)+'_'+str(day_name)].append( area_t)
        eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)].append( area_e/area_t)
        #delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)].append( area_d/area_t)
        #delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)].append( area_d/area_e)
        
        P_i = np.real(P_i)/area_tot
        
        
# =============================================================================
#             P_hat = savgol_filter(P_i,300,3)
#             fig2, ax2 = pl.subplots(figsize=(16,16))
#             
#             #ax2.plot(f_i, np.real(P_i), lw =0.5)
#             ax2.plot(f_i[(f_i > 0.1) & (f_i < 20)], 10*np.log10(P_hat[(f_i > 0.1) & (f_i < 20)]))
#             ax2.set_title(str(rat_name)+str(mod_name)+'_'+str(day_name)+' cell '+str(j)+'\n'+r' $A_\eta$/$A_\theta$='+str(np.round(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)][-1],decimals=3))+'\n ', fontsize = 32)
#             #ax2.set_title(str(rat_name)+str(mod_name)+'_'+str(day_name)+' cell '+str(j), fontsize = 32)
#             ax2.set_xlabel('frequency [Hz]', fontsize = 45)
#             ax2.set_ylabel('dB', fontsize = 45)
#             #ax2.set_xscale('log')
#             #ax2.set_yscale('log')
#             ax2.spines['top'].set_visible(False)
#             ax2.spines['right'].set_visible(False)
#             #ax2.set_ylim(0.01, 1e7)
#             #ax2.set_xlim(0.01, 40)
#             ax2.tick_params(axis='both',  labelsize=40)
#             #pl.xlim(1,20)
#             #pl.savefig('PS_'+str(rat_name)+str(mod_name)+'_'+str(day_name)+'.png', dpi=300, bbox_inches='tight')
#             pl.show()
# =============================================================================
        
        Ps[str(rat_name)+str(mod_name)+'_'+str(day_name)].append(np.real(P_i))
        fs[str(rat_name)+str(mod_name)+'_'+str(day_name)].append(f_i)
        
# =============================================================================
#         P = savgol_filter(P_i, 4000, 3)
#         index_peak = scipy.signal.find_peaks( P[(f_i > 1) & (f_i < 20)], width = 2000 )
#         if len(index_peak[0])>1:
#             f_theta[str(rat_name)+str(mod_name)+'_'+str(day_name)].append(f_i[index_peak[0][0]])
#             f_eta[str(rat_name)+str(mod_name)+'_'+str(day_name)].append(f_i[index_peak[0][1]]) 
#         else:
#             f_theta[str(rat_name)+str(mod_name)+'_'+str(day_name)].append(np.nan)
#             f_eta[str(rat_name)+str(mod_name)+'_'+str(day_name)].append(np.nan)
# =============================================================================
        
    Ps[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.array(Ps[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    np.savez_compressed('ps/PS_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+'_'+str(velocity)+str(threshold), Ps[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    np.savez_compressed('ps/f_'+str(typ)+str(velocity)+str(threshold), f_i)

    
    #with open('ps/Aeta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'wb') as f:
    #    pickle.dump(Aeta, f)
    #with open('ps/Atheta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'wb') as f:
    #    pickle.dump(Atheta, f)
    with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'wb') as f:
        pickle.dump(eta_to_thetas, f)
        
with open('ps/ratio_eta_theta_'+str(typ)+str(velocity)+str(threshold)+'.pkl', 'wb') as f:
    pickle.dump(eta_to_thetas, f)
#with open('ps/ratio_delta_theta_'+str(velocity)+str(threshold)+'.pkl', 'wb') as f:
#    pickle.dump(delta_to_thetas, f)
#with open('ps/ratio_delta_eta_'+str(velocity)+str(threshold)+'.pkl', 'wb') as f:
#    pickle.dump(delta_to_etas, f)


#%%
with open('Toroidal_topology_grid_cell_data/spacings.pkl', 'rb') as f:
       spacing = pickle.load(f)
       
modules = [[#('R', '1', 'OF', 'day1', 'pink'),
           ('R', '2', 'OF', 'day1', 'lightcoral'),
           ('R', '3', 'OF', 'day1', 'maroon')
           ],
           [('R', '1', 'OF', 'day2', 'turquoise'),
           ('R', '2', 'OF', 'day2', 'darkblue'),
           #('R', '3', 'OF', 'day2', 'black')
           ],
           [('Q', '1', 'OF', '', 'lawngreen'),
           ('Q', '2', 'OF', '', 'darkgreen')],
           [('S', '1', 'OF', '', 'purple')]
           ]

modules = modules[0]

folder_data = 'ps/'
data = np.load(folder_data+'f_'+str(typ)+str(velocity)+str(threshold)+'.npz', allow_pickle=True)
f_i = data['arr_0']

fig1, ax1 = pl.subplots(figsize=(16,16))

for i, (rat_name, mod_name, sess_name, day_name, colors) in enumerate(modules):
    
    data = np.load(folder_data+'PS_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+'_'+str(velocity)+str(threshold)+'.npz',allow_pickle=True)
    Ps_dwn = data['arr_0']
    
    #Ps_dwn = Ps[str(rat_name)+str(mod_name)+'_'+str(day_name)]
    
    Phat = savgol_filter(np.nanmean(Ps_dwn, axis=0), 1600, 3)
    Phat_std = savgol_filter(np.nanstd(Ps_dwn,axis=0)/np.sqrt(Ps_dwn.shape[0]), 1600, 3)
    
    
    if i < 1:
        ax1.plot(f_i[(f_i>1) & (f_i<20)], 10*np.log10(Phat[(f_i>1) & (f_i<20)]), color=colors, lw = 6, ls ='dashed', label = r'$'+str(rat_name)+'^{'+str(mod_name)+'}_{'+str(int(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]),decimals =0)))+'}$')
    else:
        ax1.plot(f_i[(f_i>1) & (f_i<20)], 10*np.log10(Phat[(f_i>1) & (f_i<20)]), color=colors, lw = 6, label = r'$'+str(rat_name)+'_{'+str(int(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]),decimals =0)))+'}$')
    
    ax1.fill_between(f_i[(f_i>1) & (f_i<20)], y1= 10*np.log10(Phat[(f_i>1) & (f_i<20)]-Phat_std[(f_i>1) & (f_i<20)]), y2= 10*np.log10(Phat[(f_i>1) & (f_i<20)]+Phat_std[(f_i>1) & (f_i<20)]), color=colors, alpha= 0.4)    
ax1.set_xlabel('frequency [Hz]', fontsize = 60)
ax1.set_ylabel('PSD [dB]', fontsize = 60)
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
leg = ax1.legend(fontsize=65)
for line in leg.get_lines():
    line.set_linewidth(6.)
ax1.tick_params(axis='both',labelsize=60)
pl.xticks([1,5,10,15,20],fontsize=60)
pl.yticks([-56, -58, -60],fontsize=60)
ax1.set_ylim(-62, -55)
pl.yticks(fontsize=72)
pl.show()
 


with open('Toroidal_topology_grid_cell_data/spacings.pkl', 'rb') as f:
       spacing = pickle.load(f)
       
typ = None

modules = [('S', '1', 'OF', '', 'purple'),
           ('R', '1', 'OF', 'day1', 'pink'),
           ('R', '2', 'OF', 'day1', 'lightcoral'),
           ('R', '3', 'OF', 'day1', 'maroon'),
           ('Q', '1', 'OF', '', 'lawngreen'),
           ('Q', '2', 'OF', '', 'darkgreen'),
           ('R', '1', 'OF', 'day2', 'turquoise'),
           ('R', '2', 'OF', 'day2', 'darkblue'),
           ('R', '3', 'OF', 'day2', 'black')
           ]

eta_to_theta = []
#delta_to_theta = [] 
#delta_to_eta = []

spacings = []

# =============================================================================
# with open('ps/ratio_eta_theta_'+str(typ)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
#      eta_to_thetas = pickle.load( f)
# =============================================================================
    
for rat_name, mod_name, sess_name, day_name,_ in modules:
    
    with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
         eta_to_thetas = pickle.load( f)
         
    if (velocity == 'fast') & (threshold == 2.5) & (rat_name =='R') & (mod_name=='2') & (day_name =='day1'):
        eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.delete(np.array(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]),129)
        #delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.delete(np.array(delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]),129)
        #delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.delete(np.array(delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)]),129)
        #pw_f[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.delete(np.array(pw_f[str(rat_name)+str(mod_name)+'_'+str(day_name)]),129)
        spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]= np.delete(np.array(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]),129)
    else:
        eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.array(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)])
        #delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.array(delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)])
        #delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.array(delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)])
        #pw_f[str(rat_name)+str(mod_name)+'_'+str(day_name)] = np.array(pw_f[str(rat_name)+str(mod_name)+'_'+str(day_name)])
        spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]= np.array(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])

    eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)][~np.isnan(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])]
    #delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)][~np.isnan(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])]
    #delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)] = delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)][~np.isnan(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])]
    #pw_f[str(rat_name)+str(mod_name)+'_'+str(day_name)] = pw_f[str(rat_name)+str(mod_name)+'_'+str(day_name)][~np.isnan(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])]
    spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]=spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)][~np.isnan(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])]
    
    with open('ps/ratio_eta_theta_post_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'wb') as f:
        pickle.dump(eta_to_thetas, f)
    
    eta_to_theta.append(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    #delta_to_theta.append(delta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    #delta_to_eta.append(delta_to_etas[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    #pw_fs.append(pw_f[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    spacings.append(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    
    
eta_to_theta = np.concatenate(eta_to_theta).ravel()
#delta_to_theta = np.concatenate(delta_to_theta).ravel()
#delta_to_eta = np.concatenate(delta_to_eta).ravel()
spacings = np.concatenate(spacings).ravel()

from scipy.stats import pearsonr

corr_et = pearsonr(eta_to_theta, spacings)
#corr_dt = pearsonr(delta_to_theta, spacings)
#corr_de = pearsonr(delta_to_eta, spacings)

slope_et, res_et,_,_,_ = np.polyfit(spacings, eta_to_theta, deg=1, full=True)
#slope_dt, res_dt,_,_,_ = np.polyfit(spacings, delta_to_theta, deg=1, full=True)
#slope_de, res_de,_,_,_ = np.polyfit(spacings, delta_to_eta, deg=1, full=True)




fig1, ax1 = pl.subplots(figsize=(16,16))
for i,(rat_name, mod_name, sess_name, day_name, colors) in enumerate(modules):
    
    with open('ps/ratio_eta_theta_post_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
         eta_to_thetas = pickle.load( f)
    
    mark = 'h'
    m_size = 20
    if ((colors == 'black') or (colors == 'purple') or (colors == 'pink') ) & (typ!='pure'):
        mark = 'x'
        m_size = 40
    if ((colors == 'black') or (colors == 'purple') ) & (typ=='pure'):
        mark = 'x'
        m_size = 40
    
    
    ax1.scatter(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)], spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)], marker = mark, s=80, color=colors, alpha =0.4)
    if rat_name != 'S':
        ax1.errorbar(np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), np.nanmean( spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), yerr = np.nanstd( spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), xerr= np.nanstd(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), marker = mark, markersize=m_size, elinewidth = 5, color=colors, label = r'$'+str(rat_name)+'^{'+str(mod_name)+'}_{'+str(int(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals=0)))+'}$')
    else:
        ax1.errorbar(np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), np.nanmean( spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), yerr = np.nanstd( spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), xerr= np.nanstd(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), marker = mark, markersize=m_size, elinewidth = 5, color=colors, label = r'$'+str(rat_name)+'_{'+str(int(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals=0)))+'}$')
        
    #ax2.plot(flat_xtot, a[0]*np.array(flat_xtot)+ a[1], lw = 3, color = 'black')
    #ax2.plot(np.linspace(min(flat_xtot),max(flat_xtot),len(a)), np.multiply(a,np.arange(np.min(flat_xtot),np.max(flat_xtot),len(a))), lw =4, color ='black')
    #mean0 = np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])  
    #st_dev = np.nanstd(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    #mean = np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)])                    
    #ax2.vlines(mean, mean+mean0-st_dev , mean+mean0+st_dev, lw=8, color=colors[i])
    ax1.plot(spacings * slope_et[0] + slope_et[1] , spacings, lw = 3, color = 'black')
# =============================================================================
# if typ == 'pure':
#     ax1.plot([], [], ls = '', label =typ)
# else:
#     ax1.plot([], [], ls = '', label ='all')
# =============================================================================
    #ax2.plot([],[],  lw =0,label = 'corr='+str(np.round(corr_et.statistic, decimals = 2)))
#ax2.set_title(r'Grid spacing vs $A_\eta$/$A_\theta$'+'\n rat running '+str(velocity)+'er than '+str(threshold)+' cm/s', fontsize = 32) 
ax1.set_xlabel(r'$A_\eta$/$A_\theta$', fontsize = 60)
ax1.set_ylabel('grid spacing [cm]', fontsize = 60)
ax1.set_xlim(0.2,1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both',  labelsize=40)
pl.legend(fontsize = 36, markerscale = 1)
pl.show()

modules = [[#('R', '1', 'OF', 'day1', 'pink'),
           ('R', '2', 'OF', 'day1', 'lightcoral'),
           ('R', '3', 'OF', 'day1', 'maroon')
           ],
           [('R', '1', 'OF', 'day2', 'turquoise'),
           ('R', '2', 'OF', 'day2', 'darkblue'),
           #('R', '3', 'OF', 'day2', 'black')
           ],
           [('Q', '1', 'OF', '', 'lawngreen'),
           ('Q', '2', 'OF', '', 'darkgreen')],
           [('S', '1', 'OF', '', 'purple')]
           ]



modules = modules[0]
#steps = [0.025,0.025]
#steps = [0.022,0.022]
steps = [0.04,0.04]

fig2, ax2 = pl.subplots(figsize=(16,16))
for i,(rat_name, mod_name, sess_name, day_name, colors) in enumerate(modules):
    if typ == 'pure':
        with open('cps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
    else:
        with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
    
    binss = np.arange(np.nanmin(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), np.nanmax(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), 0.05)
    ax2.hist(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)], bins = binss, density = True, color=colors,label = r'$'+str(rat_name)+'_{'+str(int(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)])))+'}$', alpha =0.8)
ax2.set_xlabel(r'$A_\eta$/$A_\theta$', fontsize = 90)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both',  labelsize=50)
pl.xlim(0.1,1.)
pl.legend(fontsize = 62)
# =============================================================================
# if typ == 'pure':
#     pl.savefig('eta_theta_hist_'+str(typ)+str(rat_name)+'_'+str(day_name)+'.png', dpi=300, bbox_inches='tight')
# else:
#     pl.savefig('eta_theta_hist_'+str(rat_name)+'_'+str(day_name)+'.png', dpi=300, bbox_inches='tight')
# 
# =============================================================================
pl.show()


       
modules = [('S', '1', 'OF', '', 'purple'),
           ('R', '1', 'OF', 'day1', 'pink'),
           ('Q', '2', 'OF', '', 'darkgreen'),
           ('R', '2', 'OF', 'day1', 'lightcoral'),
           ('R', '2', 'OF', 'day2', 'darkblue'),
           ('Q', '1', 'OF', '', 'lawngreen'),
           ('R', '1', 'OF', 'day2', 'turquoise'),
           ('R', '3', 'OF', 'day2', 'black'),
           ('R', '3', 'OF', 'day1', 'maroon'),
           ]

D_pure={}
D_pure['R1_day1']=[0.75,0.71]
D_pure['R2_day1']=[0.77,0.81]
D_pure['R3_day1']=[0.77,0.86]
D_pure['R1_day2']=[0.78,0.83]
D_pure['R2_day2']=[0.74,0.73]
D_pure['R3_day2']=[0.45,0.56]
D_pure['Q1_']=[0.74,0.71]
D_pure['Q2_']=[0.62,0.64]
D_pure['S1_']=[0.6,0.34]

D_all={}
D_all['R1_day1']=[0.23,0.62]
D_all['R2_day1']=[0.77, 0.89]
D_all['R3_day1']=[0.78, 0.80]
D_all['R1_day2']=[0.64, 0.79]
D_all['R2_day2']=[0.85, 0.71]
D_all['R3_day2']=[0.54, 0.80]
D_all['Q1_']=[0.81, 0.74]
D_all['Q2_']=[0.70, 0.64]
D_all['S1_']=[0.18,0.29]

typ = None

fig1, ax1 = pl.subplots(figsize=(16,16))
for i, (rat_name, mod_name, sess_name, day_name, colors) in enumerate(modules):
    
    with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
             
    if typ==None:    
        if rat_name != 'S':
            ax1.scatter(np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), D_all[str(rat_name)+str(mod_name)+'_'+str(day_name)][0], s = 1000, marker = 'h', color = colors, label = r'$'+str(rat_name)+'^{'+str(mod_name)+'}_{'+str(int(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]),decimals =0)))+'}$')
        else:
            ax1.scatter(np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), D_all[str(rat_name)+str(mod_name)+'_'+str(day_name)][0], s = 1000, marker = 'h', color = colors, label = r'$'+str(rat_name)+'_{'+str(int(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]),decimals =0)))+'}$')

        ax1.scatter(np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), D_all[str(rat_name)+str(mod_name)+'_'+str(day_name)][1], s = 1000, marker = 'x', color = colors)
        
    elif typ=='pure':
        ax1.scatter(np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), D_pure[str(rat_name)+str(mod_name)+'_'+str(day_name)][0], s = 1000, marker = 'h', color = colors, label = r'$'+str(rat_name)+'^{'+str(mod_name)+'}_{'+str(int(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]),decimals =0)))+'}$')
        ax1.scatter(np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), D_pure[str(rat_name)+str(mod_name)+'_'+str(day_name)][1], s = 1000, marker = 'x', color = colors)
        
    ax1.set_xlabel(r'$A_{\eta}/A_{\theta}$', fontsize = 60)
    font2 = {'family':'Times New Roman','color':'black','size':15}
    ax1.set_ylabel(r'$\mathdefault{\Gamma_{d}}$',fontdict=font2, fontsize=130, rotation = 0, labelpad =80)
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    leg = ax1.legend(fontsize=45)
    for line in leg.get_lines():
        line.set_linewidth(6.)
    ax1.tick_params(axis='both',labelsize=60)
    pl.yticks(fontsize=72)
    pl.ylim(0.15, 0.95)
    pl.xlim(0.35,1.)
pl.show()

typ = None
modules = [#('S', '1', 'OF', '', 'purple'),
           #('R', '1', 'OF', 'day1', 'pink'),
           ('Q', '2', 'OF', '', 'darkgreen'),
           ('R', '2', 'OF', 'day1', 'lightcoral'),
           ('R', '2', 'OF', 'day2', 'darkblue'),
           ('Q', '1', 'OF', '', 'lawngreen'),
           ('R', '1', 'OF', 'day2', 'turquoise'),
           #('R', '3', 'OF', 'day2', 'black'),
           ('R', '3', 'OF', 'day1', 'maroon'),
           ]

for i, (rat_name, mod_name, sess_name, day_name,colors) in enumerate(modules):
    if typ == 'pure':
        with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
    else:
        with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
             

with open('/Users/giovannidisarra/Desktop/delta_tc1.pkl', 'rb') as f:
      delta_tc1 = pickle.load(f)
with open('/Users/giovannidisarra/Desktop/delta_tc2.pkl', 'rb') as f:
      delta_tc2 = pickle.load(f)
      
delta_tc = []
for i, (rat_name, mod_name, sess_name, day_name, colors) in enumerate(modules):
    delta_tc.append(min(delta_tc1[str(rat_name)+str(mod_name)+'_'+str(day_name)], delta_tc2[str(rat_name)+str(mod_name)+'_'+str(day_name)]))

fig1, ax1 = pl.subplots(figsize=(16,16))
for i, (rat_name, mod_name, sess_name, day_name, colors) in enumerate(modules):
    with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
    ax1.scatter(delta_tc[i], np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), s = 2000)
pl.show()
    

    
fig=plt.figure(figsize=(16,16))
axis=fig.add_subplot(111)
for i,(rat_name, mod_name, sess_name, day_name, color) in enumerate([#('R', '1', 'OF', 'day1'),
                                                ('R', '3', 'OF', 'day1', 'maroon'),
                                                ('Q', '2', 'OF', '', 'darkgreen'),
                                                ('R', '2', 'OF', 'day1', 'lightcoral'),
                                                ('R', '2', 'OF', 'day2', 'darkblue'),
                                                ('Q', '1', 'OF', '', 'lawngreen'),
                                                ('R', '1', 'OF', 'day2', 'turquoise'),
                                                #('R', '3', 'OF', 'day2'),
                                                #('S', '1', 'OF', ''),
                                                ]):
    with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
             
    #axis.scatter(delta_tc1[str(rat_name)+str(mod_name)+'_'+str(day_name)], np.nanmean(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)]), s = 2000)
    
    axis.scatter(delta_tc1[str(rat_name)+str(mod_name)+'_'+str(day_name)], D_all[str(rat_name)+str(mod_name)+'_'+str(day_name)][0], color=color, s = 2000)

font2 = {'family':'Times New Roman','color':'black','size':15}
axis.set_xlabel(r'$\Delta t_{c}^{(1)}$ [ms]',fontsize=60)
axis.set_ylabel(r'$\mathdefault{\Gamma}_1$',fontdict = font2, fontsize=70, rotation =0, labelpad =80)
for label in axis.xaxis.get_majorticklabels():
    label.set_fontsize(50)
for label in axis.yaxis.get_majorticklabels():
    label.set_fontsize(50)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
plt.locator_params(axis='both', nbins=5) 
#plt.legend(loc = 'lower right',fontsize = 52)
#plt.savefig('delta_t1_Gamma.png', dpi=300, bbox_inches='tight')
plt.show()

fig=plt.figure(figsize=(16,16))
axis=fig.add_subplot(111)
for i,(rat_name, mod_name, sess_name, day_name, color) in enumerate([#('R', '1', 'OF', 'day1'),
                                                ('R', '3', 'OF', 'day1', 'maroon'),
                                                ('Q', '2', 'OF', '', 'darkgreen'),
                                                ('R', '2', 'OF', 'day1', 'lightcoral'),
                                                ('R', '2', 'OF', 'day2', 'darkblue'),
                                                ('Q', '1', 'OF', '', 'lawngreen'),
                                                ('R', '1', 'OF', 'day2', 'turquoise'),
                                                #('R', '3', 'OF', 'day2'),
                                                #('S', '1', 'OF', ''),
                                                ]):
    with open('ps/ratio_eta_theta_'+str(typ)+str(rat_name)+str(mod_name)+'_'+str(day_name)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
             eta_to_thetas = pickle.load( f)
             
    axis.scatter(delta_tc2[str(rat_name)+str(mod_name)+'_'+str(day_name)], D_all[str(rat_name)+str(mod_name)+'_'+str(day_name)][1], color = color, s = 2000)

font2 = {'family':'Times New Roman','color':'black','size':15}
axis.set_xlabel(r'$\Delta t_{c}^{(2)}$ [ms]',fontsize=60)
axis.set_ylabel(r'$\mathdefault{\Gamma}_2$',fontdict = font2, fontsize=70, rotation =0, labelpad =80)
for label in axis.xaxis.get_majorticklabels():
    label.set_fontsize(50)
for label in axis.yaxis.get_majorticklabels():
    label.set_fontsize(50)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
plt.locator_params(axis='both', nbins=5) 
#plt.legend(loc = 'lower right',fontsize = 52)
#plt.savefig('delta_t2_Gamma.png', dpi=300, bbox_inches='tight')
plt.show()
    


#correlation between inflection point and eta/theta ratio

with open('ps/ratio_eta_theta_'+str(typ)+str(velocity)+str(threshold)+'.pkl', 'rb') as f:
     eta_to_thetas = pickle.load( f)

modules = [#('S', '1', 'OF', ''),
           #('R', '1', 'OF', 'day1'),
           ('R', '3', 'OF', 'day1'),
           ('Q', '2', 'OF', ''),
           ('R', '2', 'OF', 'day1'),
           ('R', '2', 'OF', 'day2'),
           ('Q', '1', 'OF', ''),
           ('R', '1', 'OF', 'day2'),
           #('R', '3', 'OF', 'day2')
           ]
eta_to_theta = []
x0s1 = []
x0s2 = []

for rat_name, mod_name, sess_name, day_name in modules:
    eta_to_theta.append(eta_to_thetas[str(rat_name)+str(mod_name)+'_'+str(day_name)])
    x0s1.append(np.tile(delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][0], len(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)])))
    x0s2.append(np.tile(delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)][0], len(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)])))
eta_to_theta = np.concatenate(eta_to_theta).ravel()
x0s1 = np.concatenate(x0s1).ravel()
x0s2 = np.concatenate(x0s2).ravel()

delta_tc = np.minimum(x0s1,x0s2)


from scipy.stats import pearsonr

corr_et1 = pearsonr(x0s1, eta_to_theta)
corr_et2 = pearsonr(x0s2,eta_to_theta)

corr_et = pearsonr(delta_tc, eta_to_theta)

slope_et, res_et,_,_,_ = np.polyfit(delta_tc, eta_to_theta, deg=1, full=True)
slope_et1, res_et1,_,_,_ = np.polyfit(x0s1, eta_to_theta, deg=1, full=True)
slope_et2, res_et2,_,_,_ = np.polyfit(x0s2, eta_to_theta, deg=1, full=True)

colors = [ 'maroon', 'darkgreen', 'lightcoral', 'darkblue', 'lawngreen', 'turquoise']

fig=plt.figure(figsize=(16,16))
axis=fig.add_subplot(111)
for i,(rat_name, mod_name, sess_name, day_name, j) in enumerate([#('R', '1', 'OF', 'day1'),
                                                ('R', '3', 'OF', 'day1',0),
                                                ('Q', '2', 'OF', '',1),
                                                ('R', '2', 'OF', 'day1', 2),
                                                ('R', '2', 'OF', 'day2',3),
                                                ('Q', '1', 'OF', '', 4),
                                                ('R', '1', 'OF', 'day2', 5),
                                                #('R', '3', 'OF', 'day2'),
                                                #('S', '1', 'OF', ''),
                                                ]):
    
        if day_name == 'day1':
            plt.errorbar(delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, markeredgewidth = 6, elinewidth=4, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals =0))+'}$')
        elif day_name == 'day2':
            axis.errorbar(delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, markeredgewidth = 6, elinewidth=4, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals =0))+'}$')
        elif day_name == '':
            axis.errorbar(delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, elinewidth=4, markeredgewidth = 6, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals=0))+'}$')

axis.plot( np.linspace(103, 630), slope_et1[0]*np.linspace(103, 630) +slope_et1[1],  lw = 2, alpha = 0.4, color = 'black')
#axis.plot( [], [], ls = '', label = 'corr '+str(np.round(corr_et2[0],decimals =2)))
axis.set_xlabel(r'$\Delta t_{c}^{(1)}$ [ms]',fontsize=80)
axis.set_ylabel(r'$A_{\eta}/A_{\theta}$',fontsize=70)
for label in axis.xaxis.get_majorticklabels():
    label.set_fontsize(70)
    #label.set_fontname('courier')
#for label in ax2.xaxis.get_majorticklabels():
 #   label.set_fontsize(40)
    #label.set_fontname('verdana')
for label in axis.yaxis.get_majorticklabels():
    label.set_fontsize(70)
    #label.set_fontname('courier')
#for label in ax2.yaxis.get_majorticklabels():
 #   label.set_fontsize(40)
    #label.set_fontname('verdana')
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
#axis.set_xlim(right=850)
plt.locator_params(axis='both', nbins=5) 
#plt.legend(loc = 'lower right',fontsize = 52)
#plt.savefig('H1_fit.png', dpi=300, bbox_inches='tight')
plt.show()

fig=plt.figure(figsize=(16,16))
axis=fig.add_subplot(111)
for i,(rat_name, mod_name, sess_name, day_name, j) in enumerate([#('R', '1', 'OF', 'day1'),
                                                ('R', '3', 'OF', 'day1',0),
                                                ('Q', '2', 'OF', '',1),
                                                ('R', '2', 'OF', 'day1', 2),
                                                ('R', '2', 'OF', 'day2',3),
                                                ('Q', '1', 'OF', '', 4),
                                                ('R', '1', 'OF', 'day2', 5),
                                                #('R', '3', 'OF', 'day2'),
                                                #('S', '1', 'OF', ''),
                                                ]):
    if day_name == 'day1':
        plt.errorbar(delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, markeredgewidth = 6, elinewidth=4, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals=0))+'}$')
    elif day_name == 'day2':
        axis.errorbar(delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, markeredgewidth = 6, elinewidth=4, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals =0))+'}$')
    elif day_name == '':
        axis.errorbar(delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, elinewidth=4, markeredgewidth = 6, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals =0))+'}$')
    
    
axis.plot( x0s2, slope_et2[0]*x0s2 +slope_et2[1],  lw = 2, alpha = 0.4,color = 'black')
#axis.plot( [], [], ls = '', label = 'corr '+str(np.round(corr_et1[0],decimals =2)))
axis.set_xlabel(r'$\Delta t_{c}^{(2)}$ [ms]',fontsize=80)
axis.set_ylabel(r'$A_{\eta}/A_{\theta}$',fontsize=70)
for label in axis.xaxis.get_majorticklabels():
    label.set_fontsize(70)
    #label.set_fontname('courier')
#for label in ax2.xaxis.get_majorticklabels():
 #   label.set_fontsize(40)
    #label.set_fontname('verdana')
for label in axis.yaxis.get_majorticklabels():
    label.set_fontsize(70)
    #label.set_fontname('courier')
#for label in ax2.yaxis.get_majorticklabels():
 #   label.set_fontsize(40)
    #label.set_fontname('verdana')
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
#axis.set_xlim(right=850)
plt.locator_params(axis='both', nbins=5) 
#plt.legend(loc = 'lower right',fontsize = 52)
#plt.savefig('H2_fit.png', dpi=300, bbox_inches='tight')
plt.show()


fig=plt.figure(figsize=(16,16))
axis=fig.add_subplot(111)
for i,(rat_name, mod_name, sess_name, day_name, j) in enumerate([#('R', '1', 'OF', 'day1'),
                                                ('R', '3', 'OF', 'day1',0),
                                                ('Q', '2', 'OF', '',1),
                                                ('R', '2', 'OF', 'day1', 2),
                                                ('R', '2', 'OF', 'day2',3),
                                                ('Q', '1', 'OF', '', 4),
                                                ('R', '1', 'OF', 'day2', 5),
                                                #('R', '3', 'OF', 'day2'),
                                                #('S', '1', 'OF', ''),
                                                ]):
    if day_name == 'day1':
        plt.errorbar(min(delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, markeredgewidth = 6, elinewidth=4, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals=0))+'}$')
    elif day_name == 'day2':
        axis.errorbar(min(delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, markeredgewidth = 6, elinewidth=4, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals =0))+'}$')
    elif day_name == '':
        axis.errorbar(min(delta_tc2[str(rat_name)+''+str(mod_name)+'_'+str(day_name)], delta_tc1[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), np.mean(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]),yerr=np.std(eta_to_thetas[str(rat_name)+''+str(mod_name)+'_'+str(day_name)]), color = colors[j], lw =0,  marker = 'h', markersize = 30, elinewidth=4, markeredgewidth = 6, capsize =10, capthick=4, label = r'$'+str(rat_name)+'_{'+str(np.round(np.nanmean(spacing[str(rat_name)+str(mod_name)+'_'+str(day_name)]), decimals=0))+'}$')
    
    
axis.plot( delta_tc, slope_et[0]*delta_tc +slope_et[1],  lw = 2, alpha = 0.4,color = 'black')
#axis.plot( [], [], ls = '', label = 'corr '+str(np.round(corr_et1[0],decimals =2)))
axis.set_xlabel(r'$\Delta t_{c}$ [ms]',fontsize=80)
axis.set_ylabel(r'$A_{\eta}/A_{\theta}$',fontsize=70)
for label in axis.xaxis.get_majorticklabels():
    label.set_fontsize(70)
    #label.set_fontname('courier')
#for label in ax2.xaxis.get_majorticklabels():
 #   label.set_fontsize(40)
    #label.set_fontname('verdana')
for label in axis.yaxis.get_majorticklabels():
    label.set_fontsize(70)
    #label.set_fontname('courier')
#for label in ax2.yaxis.get_majorticklabels():
 #   label.set_fontsize(40)
    #label.set_fontname('verdana')
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
#axis.set_xlim(right=850)
plt.locator_params(axis='both', nbins=5) 
#plt.legend(loc = 'lower right',fontsize = 52)
#plt.savefig('detatc_fit.png', dpi=300, bbox_inches='tight')
plt.show()

    

    
    

