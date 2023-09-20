#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:55:29 2023

@author: giavanna
"""

import numpy as np
import xcompy as xc
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay



def md3_area(mu, triple):
    '''
    input:
        mu - vec of measured mu at E1, E2
        triple - triangle with true basis material mu(E1, E2) vertices
    output:
        alphas - volume fractions
    '''
    A1 = tri_area(mu, triple[1], triple[2])
    A2 = tri_area(mu, triple[0], triple[2])
    A3 = tri_area(mu, triple[0], triple[1])
    A = np.array([A1, A2, A3])

    alphas = np.array([Ai/np.sum(A) for Ai in A])
    return alphas


def tri_area(p1, p2, p3):
    area = 0.5 * (p1[0] * (p2[1] - p3[1]) 
            + p2[0] * (p3[1] - p1[1]) 
            + p3[0] * (p1[1] - p2[1]))
    return np.abs(area)


# choose optimal triangle
def is_inside(mu, triple, EPS=1e-8):
    '''
    a point P is inside a triangle ABC if 
    the sum of areas of PAB, PBC, PAC = area of ABC
    '''
    A = tri_area(triple[0], triple[1], triple[2])
    A1 = tri_area(mu, triple[0], triple[1])
    A2 = tri_area(mu, triple[1], triple[2])
    A3 = tri_area(mu, triple[0], triple[2])
    diff = np.abs(A-(A1+A2+A3))
    return diff < EPS

        
# in case the point is not inside the tesselation...
def d_hausdorff(mu, triple):
    '''
    mu - vec of measured mu at E1, E2
    triple - triangle of truth mu vector coordinates
    '''
    # make sure all numpy arrays
    mu = np.array(mu)
    triple = np.array(triple)

    # calc distance
    distances = np.zeros(3)
    for i in range(3):
        distances[i] = np.linalg.norm(triple[i] - mu)

    return np.min(distances)


def d_point_line(P0, P1, P2):
    '''
    calc distance between point P0 (x0,y0)
    and the line between P1 and P2
    '''
    # unpack coords
    x0, y0 = P0
    x1, y1 = P1
    x2, y2 = P2
    
    # line P1-P2
    m = (y2-y1)/(x2-x1)
    b = y1-m*x1

    # perpendicular line from P0 
    m0 = -1/m
    b0 = y0-m0*x0

    # intersection point
    xi = (b0-b)/(m-m0)
    yi = m*xi + b

    # distance Pi to P0
    d = np.sqrt((x0-xi)**2 + (y0-yi)**2)
    return d
    

def get_alphas(mu_test, mat_mus, mat_names, tri):

    # init
    alphas = None
    min_hausdorff = 1e8  # something large
    
    for tri_inds in tri.simplices:
        triplet_names = [mat_names[i] for i in tri_inds]
        mu_triplet = np.array([mat_mus[i] for i in tri_inds])   
    
        # first check triplets
        if is_inside(mu_test, mu_triplet):
            #alphas = md3(mu_test, mu_triplet)
            alphas = md3_area(mu_test, mu_triplet)
            break
    
        # if not in triplets, prepare minimum hausdorff dist
        d = d_hausdorff(mu_test, mu_triplet) 
        if d < min_hausdorff:
            min_hausdorff = d
            
    
    if alphas is None: # not inside tesselation, go back through
        for tri_inds in tri.simplices:
            triplet_names = [mat_names[i] for i in tri_inds]
            mu_triplet = np.array([mat_mus[i] for i in tri_inds])   
        
            d = d_hausdorff(mu_test, mu_triplet) 
            if d==min_hausdorff:
                #alphas = md3(mu_test, mu_triplet)
                alphas = md3_area(mu_test, mu_triplet)
                break
    return triplet_names, alphas




def mmd(E1, E2, M1, M2, mats):
    
    # `points` - mu(E1, E2) coordinates for each basis material
    points = []
    names = []
    for name, density, matcomp in mats:
        mu_vals = density*xc.mixatten(matcomp, np.array([E1, E2], dtype=np.float64))
        points.append(mu_vals)
        names.append(name)
    points = np.array(points)
    
    # `tri` - create material triplet library
    tri = Delaunay(points)
    
    # initialize basis images
    basis_img_dict = {}
    for name in names:
        basis_img_dict[name] = np.zeros(M1.shape, dtype=np.float32)
    
    # compute images
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            mu_px = np.array([M1[i,j], M2[i,j]])
            triplet_names, alphas = get_alphas(mu_px, points, names, tri)
            for k, name in enumerate(triplet_names):
                basis_img_dict[name][i,j] = alphas[k]

    return basis_img_dict



def plot_tessel(mats, E1, E2, xy=None, imgs=None, show=True):
    E = np.array([E1, E2], dtype=np.float64)

    # get the linear atten coeff's ! (mu_vals)
    points = []
    names = []
    for name, density, matcomp in mats:
        mu_vals = density*xc.mixatten(matcomp, E)
        points.append(mu_vals)
        names.append(name.split('_')[0])
    points = np.array(points)
    tri = Delaunay(points)

    # plot the tesselation! 
    fig, ax = plt.subplots(1,1, figsize=[6,4])
    if imgs is not None: 
        M1, M2 = imgs
        ax.plot(M1.ravel(), M2.ravel(), 'b.', markersize=.1, label='image data')
    
    fig.suptitle(f'$E_1, E_2 =$ {E1}, {E2} keV')
    ax.triplot(points[:,0], points[:,1], tri.simplices, 'k-', lw=.5, alpha=0.7)
    for i in range(len(mats)):
        x, y = points[i]
        name = names[i]
        ax.plot(x,y, marker='.', ls='', markerfacecolor='None', label=name)#, color='k')
    if xy is not None:
        ax.plot(xy[0], xy[1], 'k+', label='data')
    ax.set_xlabel('$\mu$('+str(E1)+' keV) [cm$^{-1}$]')
    ax.set_ylabel('$\mu$('+str(E2)+' keV) [cm$^{-1}$]')
    ax.set_xlim(0,2.1)
    ax.set_ylim(0,.9)
    ax.legend()#bbox_to_anchor=[1.1,.9])
    if show:
        plt.show()
    else:
        return fig, ax


def make_vmi(E0, img1, img2, matcomp1, matcomp2, HU=False):
    """
    Function to compute virtual monoenergetic image (VMI) 
    from two input basis material images (img1, img2).
    The images img1 and img2 should have pixels in units
    of g/cm^3. The energy at which to evaluate the VMI (E0) 
    should be in units keV. The material compositions 
    of the basis material images is given in matcomp1, matcomp2.    
    The scaling factors (mass attenuation coefficients) are 
    then computed using xcompy.
    
    Optional argument "HU" (default True) changes the units
    of the pixels in the output image. If True, images are
    in Hounsfield Units. If False, units are the linear
    attenuation coefficients.
    """
    mac1 = xc.mixatten(matcomp1, np.array([E0]).astype(np.float64))
    mac2 = xc.mixatten(matcomp2, np.array([E0]).astype(np.float64))
    vmi = mac1*img1 + mac2*img2
    if HU:
        u_w =  1.0 * xc.mixatten('H(11.2)O(88.8)',  np.array([E0]).astype(np.float64))
        vmi = 1000*(vmi-u_w)/u_w
    return vmi.astype(np.float32)
    


### combining Omni + blood
percent_omni = 0.2
d_blood =  {1: 0.102000,
                     6: 0.110000,
                     7: 0.033000,
                     8: 0.745000,
                     11: 0.001000,
                     15: 0.001000,
                     16: 0.002000,
                     17: 0.003000,
                     19: 0.002000,
                     26: 0.001000}
d_omni = {1: 0.03191478751101292, 
                         6: 0.2779110718133381, 
                         7: 0.05117301559050043,
                         8: 0.1753598375717305,
                         53: 0.46364128751341815}

d_omni_in_blood = {}
for key in d_blood:
    d_omni_in_blood[key] = (1-percent_omni)*d_blood[key]
for key in d_omni:
    if key in d_omni_in_blood:
        d_omni_in_blood[key]+= percent_omni*d_omni[key]
    else:
        d_omni_in_blood[key] = percent_omni*d_omni[key]








