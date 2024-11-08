#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:47:31 2023

@author: giovannidisarra

Collection of functions that quantify the degree of toroidal topology from barcodes

"""

import numpy as np
import persim
from itertools import combinations
import copy

def make_idealized_barcode_strict(barcode_in):
    '''

    Parameters
    ----------
    barcode : list of holes birth and death times in n-dimensions

    Returns
    -------
    final_barcode : list of holes birth and death times in n-dimensions where the typical toroidal bars
    are kept, while the others are put equal to the minimal non-typical bar.

    '''
    
    barcode = copy.deepcopy(barcode_in)
    
    final_barcode = []
    
    lifetimes = [np.subtract( barcode[h][:,1], barcode[h][:,0]) for h in range(len(barcode))]
    
    for h in range(len(barcode)):
        
        if h==1:
            longliveds = sorted(lifetimes[h])[-2:]
            longlived_min = np.min(longliveds)
            longlived_max = np.max(longliveds)
            min_life = np.nanmin(lifetimes[h])
            barcode[h][:,1] = np.where( lifetimes[h] >= longlived_min, barcode[h][:,0] + longlived_max, min_life)
           
        else:
            longlived = np.max(lifetimes[h])
            min_life = np.nanmin(lifetimes[h])
            barcode[h][:,1] = np.where( lifetimes[h] < longlived, barcode[h][:,0]+min_life, barcode[h][:,1])
        
        final_barcode.append(barcode[h])
    

    return final_barcode



def make_idealized_barcode(barcode_in):
    '''

    Parameters
    ----------
    barcode : list of holes birth and death times in n-dimensions

    Returns
    -------
    final_barcode : list of holes birth and death times in n-dimensions where the typical toroidal bars
    are kept, while the others are put equal to the minimal non-typical bar.

    '''
    
    barcode = copy.deepcopy(barcode_in)
    
    final_barcode = []
    
    lifetimes = [np.subtract( barcode[h][:,1], barcode[h][:,0]) for h in range(len(barcode))]
    
    for h in range(len(barcode)):
        
        if h==1:
            longlived = np.min(sorted(lifetimes[h])[-2:])
        else:
            longlived = np.max(lifetimes[h])
            
        lifetimes[h] = np.where( lifetimes[h] < longlived, lifetimes[h], np.nan)
        #avg_life = np.nanmean(lifetimes[h])
        avg_life = np.nanmin(lifetimes[h])
        #avg_life = 0.
        
        barcode[h][:,1] = np.where( lifetimes[h] < longlived, barcode[h][:,0] + avg_life, barcode[h][:,1])
        
        final_barcode.append(barcode[h])
    

    return final_barcode




def norm_bdistance(barcode1, barcode2):
    '''
    
    Parameters
    ----------
    barcode1 : first list of holes birth and death times in n-dimensions
    barcode2 : second list of holes birth and death times in n-dimensions

    Returns
    -------
    Gamma : normalized bottleneck distance between the two lists in n-dimensions.
    *Increase floating point precision

    '''
    
    barcodes = [copy.deepcopy(barcode1), copy.deepcopy(barcode2)]
    
    
    #barcodes = [ barcode1, barcode2 ]
    
# =============================================================================
#     plot_barcode(barcodes[0])
#     plot_barcode(barcodes[1])
#     plt.show()
# =============================================================================
    
    
    dims = len(barcodes[0])
    print(dims)
    diams = np.zeros((len(barcodes), dims))
    
    #compute diameters of the barcodes in n dimensions (dims)
    for i,barcode in enumerate(barcodes):
        for j in range(1,dims):
            
            diams[i][j] = np.max(np.array([ persim.bottleneck([bd[0]],[bd[1]]) for bd in list(combinations(barcode[j],2))]))
            
    #or simply...
    #for i,barcode in enumerate(barcodes):
    #    for j in range(1,dims):
    #        bar_max = barcode[j][np.where(np.diff(barcode[j], axis=1)==np.min(np.diff(barcode[j], axis=1)))[0][0]]
    #        bar_min = barcode[j][np.where(np.diff(barcode[j], axis=1)==np.max(np.diff(barcode[j], axis=1)))[0][0]]
     #       diams[i][j] =  persim.bottleneck([bar_max], [bar_min])
    
    
    
    for i,barcode in enumerate(barcodes):
        for j,bars in enumerate(barcode):
            if j==0:
                continue
            else:
                barcodes[i][j] = bars / diams[i][j]
    
    
# =============================================================================
#     plot_barcode(barcodes[0])
#     plot_barcode(barcodes[1])
#     plt.show()
# =============================================================================
    
    Gamma = np.array([ persim.bottleneck(barcodes[0][j], barcodes[1][j] ) for j in range(1,dims)])
    #D = np.array([ persim.bottleneck(barcodes[0][j]/diams[0][j], barcodes[1][j]/diams[1][j] ) for j in range(1,dims)])
    
    return Gamma


def compute_Gamma_self(barcode):
    '''
    
    Parameters
    ----------
    barcode : list of holes birth and death times in n dimensions

    Returns
    -------
    Gamma : degree of toroidal topology in n dimensions for barcode

    '''
    
    ideal_barcode = make_idealized_barcode_strict(barcode)
    
    
    Gamma = 1 - norm_bdistance(ideal_barcode, barcode)
    
    return Gamma




