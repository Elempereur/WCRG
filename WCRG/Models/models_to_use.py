"""Ansatz that are ready to be used to learn conditional/direct energies """

import numpy as np
import matplotlib.pyplot as plt
import torch

from .bands import decompose_concat, reconstruct
from .models import  SigmoidWindows, GaussianNonCondi, GaussianPotential,GaussianPotential_NoSym
from .reco_deco import *

from Models_abstract import Condi_Union

def ANSATZ_NoCondi(L,centers,sigma,shifts,shifts_sym = False):
    """Non conditionnal ansatz for direct estimation of energy, a scalar potential (sigmoids) + a quadratic potential
    
    Parameters:
    L (int): system size = L*L 
    centers (tensor): position of the centers of the sigmoids 
    sigma (tensor) : width of the sigmoids
    shifts (list of tuples) : spatial shifts for the quadratic potential, carefull (0,0) is already taken into account, do not add here  
    shifts_sym (Bool) : if True, the shifts are not symetrized
    
    
    Returns:
        ansatz_union (Condi_Union) : Ready to be Trained Ansatz
    
    """

    reco , deco = reconstruct_no_condi(), decompose_no_condi()
    
    #Quadratic potential
    if shifts_sym==False:
        ansatz_gauss = GaussianNonCondi(num_varying_channels=1,
                                        num_conditioning_channels=0,
                                        shifts = shifts)
    else:
      raise NotImplementError
    
    #Scalar Potential
    ansatz_scalar=SigmoidWindows(reco, centers, sigma, device='cuda')
    #Union
    ansatz_union = Condi_Union([ansatz_scalar,ansatz_gauss], condi_index=[0,1],decompose=deco, reconstruct = reco, device='cuda')
    
    """Dirty"""
    ansatz_union.L = L 
    
    return(ansatz_union)

def ANSATZ_Wavelet(W,L,centers,sigma,mode,shifts,shifts_sym = False):
    """Conditionnal ansatz for conditonal Energy \bar E(\bar x_j\vert x{j}) estimation with a wavelet transform,with a scalar potential (sigmoids) + a quadratic potential
    
    Parameters:
    W (Wavelet) : Wavelet to perfom fast wavelet transform
    L (int): system size ( of x_{j-1}) = L*L 
    centers (tensor): position of the centers of the sigmoids 
    sigma (tensor) : width of the sigmoids
    shifts (list of tuples) : spatial shifts for the quadratic potential, carefull (0,0) is already taken into account, do not add here 
    shifts_sym (Bool) : if True, the shifts are not symetrized
    
    
    Returns:
        ansatz_union (Condi_Union) : Ready to be Trained Ansatz
    
    """


  
    reco , deco = reconstruct_wav(W), decompose_wav(W)
    
    if shifts_sym == False :
        ansatz_gauss = GaussianPotential(mode=mode, num_varying_channels=3,
                                        num_conditioning_channels=1,
                                        shifts = shifts)
    elif shifts_sym == True:
        ansatz_gauss = GaussianPotential_NoSym(mode=mode, num_varying_channels=3,
                                        num_conditioning_channels=1,
                                        shifts = shifts)
    
    else:
        print('shifts_sym  should be True or False')
    
    #Scalar potential
    ansatz_scalar=SigmoidWindows(reco, centers, sigma, device='cuda')
    #Union
    ansatz_union = Condi_Union([ansatz_scalar,ansatz_gauss], condi_index=[0,1], decompose=deco, reconstruct = reco, device='cuda')
    
    """Dirty"""
    ansatz_union.L = L
    
    return(ansatz_union)


def ANSATZ_Packet(W,L,N,Width,tree,centers,sigma,mode,shifts,shifts_sym = False,inter_plus=False):
    """Conditionnal ansatz for conditonal Energy \bar E(\bar x_j\vert x{j}) estimation with a wavelet packet transform,with a scalar potential (sigmoids) + a quadratic potential
    
    Parameters:
    W (Wavelet) : Wavelet to perfom fast wavelet transform
    L (int): system size ( of x_{j-1}) = L*L 
    N (int) : position of the bandwidth to use as \bar x_j
    Width (int) : width of the band, for an image L*L, the bandwith is between [N:N+width]
    tree (Tree) : the tree for wavelet packet
    centers (tensor): position of the centers of the sigmoids 
    sigma (tensor) : width of the sigmoids
    mode (string): Frequencies to interact in quadratic potential, 'All' for all frequency combinaisons, 'Next_Neighbors'  for close frequencies only  
    shifts (list of tuples) : spatial shifts for the quadratic potential, carefull (0,0) is already taken into account, do not add here  
    shifts_sym (Bool) : if True, the shifts are not symetrized
    inter_plus (Bool) : if True, adds \bar x_{j+2} for quadratic interactions
    
    
    Returns:
        ansatz_union (Condi_Union) : Ready to be Trained Ansatz
    
    """
    
    #Number of sub-maps of the wavelet packet in the band is 2*n_band+1
    n_band = N//Width
    #Energy Model
    
    reco = reconstruct_wav_packet(W,tree,L)
    
    #reco and deco decompose x_{j-1} into x_{j},\bar x_{j} and \bar x_{j+1}
    if inter_plus == False:
        deco = decompose_wav_packet(W,tree,L,Width,N)
    else:
        #if inter_plus == True, then we add \bar x_{j+2} for quadratic interactions
        deco = decompose_inter_plus(W,tree,L,Width,N)
    
    
    #Quadratic potential
    if shifts_sym == False :
        ansatz_gauss = GaussianPotential(mode=mode, num_varying_channels=2 * n_band + 1,
                                        num_conditioning_channels=2 * n_band - 1,
                                        shifts = shifts)
    elif shifts_sym == True:
        ansatz_gauss = GaussianPotential_NoSym(mode=mode, num_varying_channels=2 * n_band + 1,
                                        num_conditioning_channels=2 * n_band - 1,
                                        shifts = shifts)
    
    else:
        print('shifts_sym  should be True or False')
    
    #Scalar potential
    ansatz_scalar=SigmoidWindows(reco, centers, sigma, device='cuda')
    #Union
    ansatz_union = Condi_Union([ansatz_scalar,ansatz_gauss], condi_index=[0,1], decompose=deco, reconstruct = reco, device='cuda')
    
    """Dirty"""
    ansatz_union.L = L
    
    return(ansatz_union)

"""This works only for wavelet transform, as x_j is on the coarse grid"""
def FREE_ANSATZ(W,L,centers,sigma,shifts,shifts_sym = False):
    """Ansatz for free energy, a scalar potential (sigmoids) + a quadratic potential
    
    Parameters:
    W (Wavelet) : Wavelet to perfom fast wavelet transform
    L (int): system size = L*L 
    centers (tensor): position of the centers of the sigmoids 
    sigma (tensor) : width of the sigmoids
    shifts (list of tuples) : spatial shifts for the quadratic potential, carefull (0,0) is already taken into account, do not add here  
    shifts_sym (Bool) : if True, the shifts are not symetrized
    
    
    Returns:
        ansatz_union (Condi_Union) : Ready to be Trained Ansatz, takes x_{j-1} in entrance but computes free energy for x_j
    
    """
    
    #Theses functions just change shape, in order to use Condi_Union with a non conditionnal purpose
    reco , deco = reconstruct_free(), decompose_free(W)
    
    
    #Quadratic potential
    if shifts_sym==False:
        ansatz_gauss = GaussianNonCondi(num_varying_channels=1,
                                        num_conditioning_channels=0,
                                        shifts = shifts)
    else:
      raise NotImplementError
    
    #Scalar potential
    ansatz_scalar=ansatz_scalar=SigmoidWindows(reco, centers, sigma, device='cuda')
    
    #Union
    ansatz_union = Condi_Union([ansatz_scalar,ansatz_gauss], condi_index=[0,0], decompose=deco, reconstruct = reco, device='cuda')
    
    """Dirty"""
    ansatz_union.L = L
    
    return(ansatz_union)