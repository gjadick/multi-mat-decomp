#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:38:22 2023

@author: giavanna
"""

import numpy as np

i_bmis = 2  # 1 for tiss/bone, 2 for H2O/Al
doses = [0.1]#np.arange(0.1, 2.1, 0.1) 
#doses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2.0]

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from mmd import plot_tessel, mmd, make_vmi
import xcompy as xc
from time import time
from datetime import timedelta

from matplotlib import patches

# lol
import warnings
warnings.filterwarnings("ignore")

uwd = {
    '80kV':      0.24212932586669922  ,
    '140kV':     0.2016972303390503  ,
    }
uad = {
    '80kV':      0.002364289714023471  ,
    '140kV':     0.0014648198848590255 ,
    }
def to_mu(HU, kVp):
    ua = uad[f'{kVp}kV']
    uw = uwd[f'{kVp}kV']
    return (HU/1000)*(uw-ua) + uw

def to_HU(mu, E0):
    # approx air as 0
    ua = 0
    # read the data file -- must be called `vmi_data.txt` and in the `input` folder! 
    data = []
    f = open('input/vmi_data.txt', 'r')
    for line in f.readlines()[1:]:  # skip 1st line, header
        data.append(np.array(line.split(), dtype=np.float64))
    f.close()
    data = np.array(data).T
    E, mac1_E, mac2_E, mac_water_E = data
    uw = np.interp(E0, E, mac_water_E)
    return 1000*(mu-uw)/(uw-ua)

def bf(string):  
    '''make text boldface'''
    return "\\textbf{"+string+"}"

def label_panels(ax, c='k', loc='outside', dx=-0.06, dy=0.09, fontsize=None,
                 label_type='lowercase', label_format='({})', i0=0):
    '''
    Function to label panels of multiple subplots in a single figure.

    Parameters
    ----------
    ax : matplotlib AxesSubplot
    c : (str) color of text. The default is 'k'.
    loc : (str), location of label, 'inside' or 'outside'. 
    dx : (float) x location relative to upper left corner. The default is 0.07.
    dy : (float) y location relative to upper left corner. The default is 0.07.
    fontsize : (number), font size of label. The default is None.
    label_type : (str), style of labels. The default is 'lowercase'.
    label_format : (str) format string for label. The default is '({})'.

    '''
    if 'upper' in label_type:
        labels = list(map(chr, range(65,91)))
    elif 'lower' in label_type:
        labels = list(map(chr, range(97, 123)))
    else: # default to numbers
        labels = np.arange(1,27).astype(str)
    labels = [ label_format.format(x) for x in labels ]

    # get location of text
    if loc == 'outside':
        xp, yp = -dx, 1+dy
    else:
        xp, yp = dx, 1-dy
        
    for i, axi in enumerate(ax.ravel()):
        xmin, xmax = axi.get_xlim()
        ymin, ymax = axi.get_ylim()
        xloc = xmin + (xmax-xmin)*xp
        yloc = ymin + (ymax-ymin)*yp
        
        label = labels[i0+i]
        axi.text(xloc, yloc, bf(label), color=c, fontsize=fontsize,
          va='center', ha='center')
        
    return None

## compute figures of merit!!
def calc_rmse_cyl(ind, bmi, obj):
    ''' 
    Compute RMSE over just the water cylinder (w/ inserts)
    ind = index of basis mat
    obj = original object of IDs
    bmi = basis mat image
    '''
    if ind==0:
        return 2 # no air
    mat_mask = np.where(obj==ind)     # mask of true basis material pixels
    gt = np.zeros(bmi.shape)          # initialize ground truth
    gt[mat_mask] = 1. 
    cyl_mask = np.where(obj!=0)         # mask the cylinder/ not air
    rmse = np.sqrt(np.mean((bmi[cyl_mask]-gt[cyl_mask])**2))
    return rmse


def calc_cnr_cyl(ind, bmi, obj):
    ''' 
    Compute CNR of insert relative to just the water cylinder (no other inserts)
    ind = index of basis mat
    obj = original object of IDs
    bmi = basis mat image
    '''
    if ind==0:
        return 2 # no air
    sig = bmi[np.where(obj==ind)]  # material insert signal
    bg = bmi[np.where(obj==1)]   # water cylinder background
    cnr = (np.mean(sig)-np.mean(bg))/(np.sqrt(np.var(sig)+np.var(bg)))
    return cnr

plt.rcParams.update({
    # figure
    "figure.dpi": 300,   # higher quality image
    # text
    "font.size":10,
    "font.family": "serif",                  # uncomment for tex style
    "font.serif": ['Computer Modern Roman'], # uncomment for tex style
    "text.usetex": True,                     # uncomment for tex style
    # axes
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "axes.linewidth": 1,
    # ticks
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    # grid
    "axes.grid" : False,
    "axes.grid.which" : "major",
     "grid.color": "lightgray",
     "grid.linestyle": ":",
     # legend
     "legend.fontsize":8,
    "legend.facecolor":'white',
    "legend.framealpha":1.0 ,  
     })


#%%

## ground truth phantom
phantom_id = 'cylinder_5mats'
N_phantom = 512
fname_phantom = f'input/sim_{phantom_id}/raw/{phantom_id}_{N_phantom}x{N_phantom}_uint8.bin'
phantom = np.fromfile(fname_phantom, dtype=np.uint8).reshape([N_phantom, N_phantom]).T
matcomp_dict = {
 0: ['air',             0.001205, 'C(0.0124)N(75.5268)O(23.1781)Ar(1.2827)'],
 1: ['water',           1.0,      'H(11.2)O(88.8)'],
 2: ['tissue_icru',     1.06,     'H(10.2)C(14.3)N(3.4)O(70.8)Na(0.2)P(0.3)S(0.3)Cl(0.2)K(0.3)'],
 3: ['fat',             0.95,     'H(11.4000)C(59.8000)N(0.7000)O(27.8000)Na(0.1000)S(0.1000)Cl(0.1000)'],
 4: ['calcium',         1.55,     'Ca(1.0)'],
 5: ['omni300_in_blood',1.06,     'H(8.7983)C(14.3582)N(3.6635)O(63.1072)Na(0.0800)P(0.0800)S(0.1600)Cl(0.2400)K(0.1600)Fe(0.0800)I(9.2728)']
 }

matnames = [matcomp_dict[m][0] for m in matcomp_dict]
print('materials:', matnames)

## input basis material images
kVp1, kVp2 = 80, 140
N_matrix = 512 # matrix size
D = 0.5  # dose
E1, E2 = 70, 140  # vmi energies

def get_fname(i_mat, dose, i_run=1):
    rootdir = f'./input/sim_{phantom_id}/two_mat_decomp/matdecomp{i_run}_{kVp1}kV_{kVp2}kV/'
    return rootdir + f'{int(dose*1000):04}uGy_{int(dose*1000):04}uGy/mat{i_mat}_recon_512_50cm_100ramp.bin'
        

### EXAMPLE IMAGES 
example=False
if example:
    img_mat1 = np.fromfile(get_fname(1, D), dtype=np.float32).reshape([N_matrix, N_matrix])
    img_mat2 = np.fromfile(get_fname(2, D), dtype=np.float32).reshape([N_matrix, N_matrix])

    vmi1 = make_vmi(E1, img_mat1, img_mat2)
    vmi2 = make_vmi(E2, img_mat1, img_mat2)

    fig,ax=plt.subplots(1,2,figsize=[7,3])
    m = ax[0].imshow(vmi1, cmap='gray',vmin=0.18,vmax=0.21)
    fig.colorbar(m, ax=ax[0])
    m = ax[1].imshow(vmi2, cmap='gray', vmin=0.14,vmax=0.19)
    fig.colorbar(m, ax=ax[1])
    for axi in ax:
        axi.axis('off')
    fig.tight_layout()
    plt.show()




#### now, actually test things
#%%  run the multi mat decomp



outdir = f'output/sim_{phantom_id}/'
os.makedirs(outdir, exist_ok=True)

## multi-mat decomp parameters 
matlib = [matcomp_dict[m] for m in matcomp_dict.keys()]

# init some figures
noise1 = []
noise2 = []

t0 = time()
i_calc = 1 
N_calc = len(doses)

# doses
for D in doses: #np.arange(0.1,0.2,0.1):#1.2,0.1):#np.arange(0.5,3,0.5):#[0.5]:
    outdirD = outdir + f'matdecomp{i_bmis}_{int(1000*D)}uGy/'
    os.makedirs(outdirD, exist_ok=True)

    ## 1. load VMIs
    img_mat1 = np.fromfile(get_fname(1, D, i_run=i_bmis), dtype=np.float32).reshape([N_matrix, N_matrix])
    img_mat2 = np.fromfile(get_fname(2, D, i_run=i_bmis), dtype=np.float32).reshape([N_matrix, N_matrix])
    
    vmi1 = make_vmi(E1, img_mat1, img_mat2)
    vmi2 = make_vmi(E2, img_mat1, img_mat2)
    
    vmi1_HU = to_HU(vmi1, E1)
    vmi2_HU = to_HU(vmi2, E2)
   

    ## 2. measure reference noise (HU)
    dx, dy = 512//8, 512//8
    x0, y0 = (512-dx)//2, (512-dy)//2

    std1 = np.std(vmi1_HU[y0:y0+dy, x0:x0+dx]) 
    std2 = np.std(vmi2_HU[y0:y0+dy, x0:x0+dx]) 
    noise1.append(std1)
    noise2.append(std2)
    fig,ax=plt.subplots(1,2,figsize=[6,3.1])
    ax[0].set_title(f'{E1} keV')
    ax[1].set_title(f'{E2} keV')
    m = ax[0].imshow(vmi1_HU, cmap='gray',vmin=-50, vmax=50)
    m = ax[1].imshow(vmi2_HU, cmap='gray', vmin=-50, vmax=50)
    cbaxes = fig.add_axes([1.008, 0.05, 0.03, 0.88])
    cb = plt.colorbar(m, cax=cbaxes, label='HU')
    for axi in ax:
        axi.axis('off')
        rect = patches.Rectangle((x0, y0), dx, dy, linewidth=.5, edgecolor='red', facecolor='none')
        axi.add_patch(rect)
    label_panels(ax, c='w', loc='inside', dx=0.07, dy= 0.05)
    fig.tight_layout()
    #plt.show()
    plt.savefig(outdirD+f'vmis_roi.png', bbox_inches="tight")
    plt.close()
   

    ## 3. create tesselations
    fig, ax = plot_tessel(matlib, E1, E2, imgs=[vmi1, vmi2], show=False)
    ax.set_xlim(0,0.85)
    ax.set_ylim(0,0.31)
    fig.suptitle(f'{D:.1f} mGy')
    #plt.show()
    plt.savefig(outdirD+'tessel.png', bbox_inches="tight")
    plt.close()


    ## 4. compute basis mat images + figures of merit
    basis_img_dict = mmd(E1, E2, vmi1, vmi2, matlib)
    cnrs = []
    rmses = []
    for i_mat, matname in enumerate(matnames):#list(basis_img_dict.keys())):  # this order must match the phantom inds!!!
        bmi = basis_img_dict[matname] 
        bmi.astype(np.float32).tofile(outdirD+f'{matname}.bin')

        # 4a. RMSE
        rmse_cyl = calc_rmse_cyl(i_mat, bmi, phantom)
        rmses.append(rmse_cyl)

        # 4b. CNR
        cnr_cyl = calc_cnr_cyl(i_mat, bmi, phantom)
        cnrs.append(cnr_cyl)

    # save figures for that dose level
    np.array(rmses).astype(np.float32).tofile(outdirD+'rmse.bin')
    np.array(cnrs).astype(np.float32).tofile(outdirD+'cnr.bin')

    # print timing 
    print(f'{i_calc}/{N_calc} - {D:.1f} mGy - {timedelta(seconds=int(time()-t0))}')
    i_calc+=1

# save noise measurements for all dose levels/the two VMIs
all_noise = np.array([doses, noise1, noise2]).astype(np.float32)
all_noise.tofile(outdir+f'matdecomp{i_bmis}_noise_{E1}_{E2}.bin')



