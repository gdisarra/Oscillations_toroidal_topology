import numpy as np
import matplotlib.pyplot as pl
import math 
import time
import os
from utils import *
import copy
import opexebo
from scipy.signal import savgol_filter
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import time
import multiprocessing
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")


def psd(x, xlen, dt,power_normalization):
    #mean_x = np.sum(x)*dt/(len(x)*dt)
    powers = np.fft.fft(x)/power_normalization
    sp = 1/dt
    frequencies = np.fft.fftfreq(len(x))*sp
    powers = np.real(powers * np.conj(powers)) 
    return powers, frequencies


#def cos_bump(x,y,mu_x,mu_y,sigma,scale):
#    rsig=np.array(np.sqrt(((x-mu_x)**2+(y-mu_y)**2))/(sigma))
    
#    return scale*(1+np.cos(np.pi*rsig))*np.heaviside(1-rsig,0)/2
    

#def Bump(x,y,mu_x,mu_y,sigma,scale):
#    r2=np.array(((x-mu_x)**2+(y-mu_y)**2)/sigma**2)
    
#    return scale*np.exp(-1/(1-np.minimum(1*np.ones(r2.shape),r2)))
    
def gauss_2D(x,y,mu_x,mu_y,sigma,scale,sp_threshold):
    """
    Compute 2D Gaussian weight on a square grid
    :param x: x coordinate on the square grid
    :param y: y coordinate on the square grid
    :param mu_x: x coordinate of the gaussian mean  
    :param mu_y: y coordinate of the gaussian meanGay
    :param sigma: standard deviation of the gaussian distribution
    :param scale: scale of the gaussian distribution
    """
    r = np.sqrt((x-mu_x)**2+(y-mu_y)**2)
    out = scale*np.exp(-((x-mu_x)/sigma)**2/2-((y-mu_y)/sigma)**2/2)
    out = out*(1-np.heaviside(r-sp_threshold,0))
    
    return out


def generate_grid(tiling, L, offset_x, offset_y, radius, orientation, jitter):
    """
    Generate hexagonal grid center coordinates in a n_x x n_y box 
    :param L: size of the arena
    :param offset_x: x offset from the origin of the firing field
    :param offset_y: y offset from the origin of the firing field
    :param orientation: orientation of the grid pattern
    """
    #n_y = 30
    n_y = 20
    n_x = int(n_y)
    
    #if spacing=='s':
     #   radius = 0.77 * L
    #if spacing=='m':
    #    radius = 0.98 * L
    #elif spacing=='l': 
     #   radius = L * 1.55
        #radius = L * 2.65
        
   # print('radius:',radius)
    # Define the angles of the six vertices of the hexagon
    angles = np.linspace(0, 2*np.pi, 7)[:-1]

    # Define the x and y spacing of the hexagons in each row
    
    if tiling=='hexagon':
        x_spac = radius * np.cos(np.pi/6)
        y_spac = radius * np.sin(np.pi/6)
    elif tiling == 'square':
        x_spac = radius*0.7 #* np.cos(np.pi/2)
        y_spac = radius*0.7 #* np.sin(np.pi/2)
        orientation = 45.
        n_y = 20
        n_x = int(n_y)
    elif tiling =='minimal':
        radius = L/2 + 0.05
        x_spac = radius #* np.cos(np.pi/2)
        y_spac = radius*4.5 #* np.sin(np.pi/2)
        orientation = 90.
        n_y = 20
        n_x = int(n_y)
    

    mu_x=[]
    mu_y=[]
    for i in range(n_y):
        y = i * y_spac
        for j in range(n_x):
            x = j * (2 * x_spac)
            if i % 2 == 1:
                x += x_spac
            mu_x.append(x)
            mu_y.append(y)
            
    mu_x = np.array(mu_x)-np.max(mu_x)/2 
    mu_y = np.array(mu_y)-np.max(mu_y)/2 
    
    mu_x = np.array(mu_x) + offset_x
    mu_y = np.array(mu_y) + offset_y
    
    for i,(mx,my) in enumerate(zip(mu_x,mu_y)):
        mu_x[i], mu_y[i] = rotate_matrix(mx,my,orientation)
    
    
    mu_x = np.where(mu_x<L*1.5, mu_x, np.nan)
    mu_x = np.where(mu_x>-L*1.5, mu_x, np.nan)
    mu_x = np.where(mu_y<L*1.5, mu_x, np.nan)
    mu_x = np.where(mu_y>-L*1.5, mu_x, np.nan)
    
    mu_y = np.where(mu_y<L*1.5, mu_y, np.nan)
    mu_y = np.where(mu_y>-L*1.5, mu_y, np.nan)
    mu_y = np.where(mu_x<L*1.5, mu_y, np.nan)
    mu_y = np.where(mu_x>-L*1.5, mu_y, np.nan)
    
    mu_x = [x for x in mu_x if str(x) != 'nan']
    mu_y = [x for x in mu_y if str(x) != 'nan']
    
    mu_x = mu_x + jitter[0][:len(mu_x)]
    mu_y = mu_y + jitter[1][:len(mu_y)]
    
    return mu_x, mu_y
    

def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr


def spatial_selectivity(dx, mu_x, mu_y, sigma, scale, sp_threshold, plot = True):
    '''
    
    Parameters
    ----------
    dx : spatial resolution
    mu1 : hexagonal grid x coordinates 
    mu2 : hexagonal grid y coordinates
    sigma : spread of the grid field
    scale : height of the grid field

    Returns
    -------
    rate : rate map from spatial selectivity only

    '''
    x,y = np.meshgrid(dx,dx)
    rate = np.zeros(x.shape)
    
    print("generating rate map...",flush = True)
    for i,xt in enumerate(dx):
        for j,yt in enumerate(dx):
            for mu1,mu2 in zip(mu_x,mu_y):
                
                #rate[i,j] = rate[i,j] + cos_bump(xt,yt,mu1,mu2,sigma,scale)
                rate[i,j] = rate[i,j] + gauss_2D(xt,yt,mu1,mu2,sigma,scale,sp_threshold)
    
    if plot:
        z_max = np.abs(rate).max()
        fig, ax = pl.subplots( figsize=(16,16))
        ax.set(adjustable='box', aspect='equal')
        c = ax.pcolormesh(dx, dx, rate, cmap='viridis', vmin=0, vmax=z_max)
        cbar = fig.colorbar(c, fraction=0.046, pad=0.04,ax=ax)
        cbar.ax.tick_params(labelsize=32)
        ax.set_axis_off()
        pl.show()
        
        
        # Set up plot
        fig, ax = pl.subplots(subplot_kw=dict(projection='3d'))
        ls = LightSource(270, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        rgb = ls.shade(rate, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, rate, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)

        pl.show()
        
    return rate


def poisson_sampling(m_rate, dt):
    '''

    Parameters
    ----------
    m_rate : lambda of the poisson process
    dt : resolution of poisson sampling

    Returns
    -------
    draw : 0 or 1 poisson draw

    '''
    if dt ==0.01:
        draw = np.random.poisson(lam=m_rate*dt)
        
    elif dt ==0.001:
        r = np.random.uniform(0,1)
        if r < m_rate*dt:
            draw = True
        else:
            draw = False
            
    return draw


def generate_train(rate, lambda0, oscillations, theta, x_traj, y_traj, x, y, prec,t,oscil_const,oscil_norm,dt,plot=True):
    '''

    Parameters
    ----------
    rate : rate map from spatial selectivity only (lambda)
    oscillations : oscillatory modulation of lambda
    x_traj : 
    y_traj : 
    x : x spatial resolution
    y : y spatial resolution
    dt : poisson sampling resolution
    oscillate : condtion on oscillatory behavior

    Returns
    -------
    train : poisson spike train binned at 1 ms
    
    '''
    
    print("generating spike train...",flush = True)
    train = np.zeros(x_traj.shape[0])
    
    spatial_rate = []
    rate_tot = []
    for i, (xt,yt) in enumerate( zip(x_traj,y_traj) ):
        #find the mean firing rate for the location of the mouse
        condition = ((x > xt-prec) & (x < xt)) & ((y > yt-prec) & (y < yt))
        spatial_rate.append(rate[np.asarray(condition).nonzero()][0] + lambda0)
        
        m_rate = spatial_rate[-1]*(oscil_const*(np.sum(oscillations[:,i]))+(1-oscil_const))/oscil_norm
        
        m_rate = max(0, m_rate)
        
        rate_tot.append(m_rate)
        
        train[i] = poisson_sampling(m_rate, dt)
    
        
    
    #print(t_p)
    if plot:
        cut_t=10000

        cond2 = ( train > 0 ) & ( t<t[cut_t] )
        t_p = t[np.asarray(cond2).nonzero()]
        fig1, ax1 = pl.subplots(figsize=(16,16))
        ax1.plot(t[:cut_t], rate_tot[:cut_t], color = 'r', label = 'total')
        ax1.vlines(t_p, 0, np.max(5.),lw=0.3, color = 'k', label = 'spikes' )
        ax1.plot(t[:cut_t], spatial_rate[:cut_t], color='g', label = 'spatially selective rate')
        ax1.set_title('Theta rhythm, spike train and firing rate', fontsize=42)
        ax1.set_xlabel('t [s]', fontsize = 32)
        ax1.set_ylabel('rate [Hz]', fontsize = 32)
        ax1.tick_params(axis='both',  labelsize=25)
        ax1.legend(fontsize=42)
        #pl.savefig(str(mode)+'Theta rhythm, spike train and firing rate, x offset='+str(np.round(offset_x,decimals=3))+', y offset='+str(np.round(offset_y,decimals=3))+'.png')
        pl.show()
        '''
        P_i, f_i = psd(train,len(train),timebin/1000)
        fig=plt.figure(figsize=(12,12))
        axis=fig.add_subplot(111)
        P_i = savgol_filter(P_i, 300, 3)
       # pl.plot(f_i,P_i)
        axis.plot(f_i[(f_i>0.1) & (f_i<20)], 10*np.log10(P_i[(f_i>0.1) & (f_i<20)]), lw = 1)
        #pl.xscale('log')
        #pl.yscale('log')
        pl.show() 
        '''
    return train, rate_tot

def train_to_times(train, t, dt):
    '''
    Parameters
    ----------
    train : binned spike train at 1 ms
    t : temporal resolution

    Returns
    -------
    sp_times : spike times with resolution of 1 ms

    '''
    print('creating spike times...')
    
    t0 = t[0]
    sp_times = []
    
    for ti, nbin in enumerate(train):
        if nbin != 0: 
            times_in_bin = np.random.uniform(0,dt*1000,int(nbin))
            times_in_bin  = times_in_bin /1000
            for n in range(int(nbin)):          
                sp_times.append(t0+ti/(1/dt)+times_in_bin[n])
            
    sp_times= np.array(sp_times)
    return sp_times


def compute_ratemap(sp_times, x, y, smooth, t, dt, folname, plot = True):

    '''
    Parameters
    ----------
    sp_times : spike times of a given neuron
    x : trajectory x coordinate
    y : trajectory y coordinate
    smooth : spatial smoothing parameter
    t : time
    
    Returns
    -------
    ratemap : ratemap from opexebo

    '''
    print('creating rate map...')            
    # Calculate an occupancy map.
    binn = np.int16(100)
    bin_edges = [np.linspace(min(x)-0.015, max(y)+0.015, binn),
                    np.linspace(min(y)-0.015, max(x)+0.015, binn)]
    
    arena_s = (150,150)
    masked_map, coverage, bin_edges = opexebo.analysis.spatial_occupancy(t,
                                                                        np.array([x, y]),
                                                                        arena_size = arena_s,
                                                                        bin_edges = bin_edges,
                                                                        )
                
    dx, dy = bin_edges
    dx,dy = np.meshgrid(dx,dy) 
    
    #prec_x = np.diff(dx[0])[0]
    #prec_y = np.diff(dy[:,0])[0]
    if dt ==0.01:
        sp_times = np.round(sp_times, decimals = 2)
        tt = np.round(t, decimals = 2)
       # print('tt', tt)
       # print('sp_times', sp_times)
    elif dt ==0.001:
        sp_times = np.round(sp_times, decimals = 3)
        tt = np.round(t, decimals = 3)
       # print('tt', tt)
       # print('sp_times', sp_times)
        
    
    
    spike_t, spikes_ind, _  = np.intersect1d( tt, sp_times, return_indices=True)

    
    spike_x = x[spikes_ind]
    spike_y = y[spikes_ind]
    
    # Example of a rate map with some smoothing.

    ratemap = opexebo.analysis.rate_map(masked_map,
                                 np.array([spike_t, spike_x, spike_y]),
                                 arena_size = arena_s,
                                 bin_edges = bin_edges)
    
    ratemap = opexebo.general.smooth(ratemap, smooth)
    
    if True:
# =============================================================================
#         z_max = np.abs(ratemap).max()
#         fig, ax = pl.subplots(ncols=2, sharex=True, sharey=True, figsize=(16,16))
#         for a in ax:    
#             a.set(adjustable='box', aspect='equal')
#             a.set_axis_off()
#         c = ax[0].pcolormesh(dx, dy, ratemap, cmap='viridis', vmin=0, vmax=z_max)
#         ax[1].plot(x,y,lw=0.1,color='blue')
#         cbar = fig.colorbar(c, fraction=0.046, pad=0.04,ax=ax[0])
#         cbar.ax.tick_params(labelsize=32)
#         ax[1].scatter(spike_x, spike_y, marker='.',color='red')
#         pl.title(str('-')+'\n # of spikes:'+str(sp_times.shape[0]), fontsize= 24)        
#         pl.savefig(folname+'/Firing map '+str(sp_times[12])+'.png')
#         pl.show()
# =============================================================================
        
        z_max = np.abs(ratemap).max()
        fig, ax = pl.subplots(figsize=(16,16))
        ax.set_axis_off()
        ax.pcolormesh(dx, dy, ratemap, cmap='viridis', vmin=0, vmax=z_max)
        pl.savefig(folname+'/Firing map '+str(sp_times[12])+'.png')
        pl.show()
        
        # Set up plot
        fig, ax = pl.subplots(subplot_kw=dict(projection='3d'))
        ls = LightSource(270, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        rgb = ls.shade(ratemap, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(dx[:-1,:-1], dy[:-1,:-1], ratemap, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)

        pl.show()
        
    return ratemap, dx, dy
 
def jitter_spike_times(sp_times, ts):
    
    noise = np.random.normal(0., ts, size=sp_times.shape)
    jit_sp_times = sp_times + noise
    
    return jit_sp_times

def sample_spikes(params):    
    
    cellnum, sp_threshold,offset_x,offset_y,tiling,L,spacing,orientation,grid_fields_jitter,scale,sigma,lambda0,x,y,dx,x_traj,y_traj,timebin,t,oscillations,theta,prec,oscil_const,oscil_norm,plot= params
    print("(x0,y0) = ("+str(np.round(offset_x,decimals=3))+","+str(np.round(offset_y,decimals=3))+")",flush = True)
   
    print(cellnum,flush = True)
    
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time,flush = True)
    #generate hexagonal grid
    mu_x, mu_y = generate_grid(tiling, L, offset_x, offset_y, spacing, orientation, grid_fields_jitter)
    
    #create firing rate matrix with 2D gaussian bumps 
    rate_space = spatial_selectivity(dx, mu_x, mu_y, sigma, scale,sp_threshold, plot=plot)

    #create spike train from the trajectory of the rat
    trains, rate_total = generate_train(rate_space, lambda0, oscillations, theta, x_traj, y_traj, x, y, prec,t, oscil_const, oscil_norm,dt = timebin/1000,plot=plot)
     
    return trains
