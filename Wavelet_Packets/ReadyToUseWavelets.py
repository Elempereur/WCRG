import numpy as np
import torch
from .Wavelet import Wavelet

def DefineWavelet(Family,m=None,device='cpu',mode='Periodic'):
    """Create Wavelet Object

    Parameters:
    Family (string): 'Db','BM','Haar'
    m (int): Which Wavelet in the family
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions 
    
    Returns:
        Wavelet

    """
    if Family == 'Haar':
        if m is not None:
            print('There is only one Haar Wavelet')
    elif Family == 'Db':
        return(Db_wavelets(m,device,mode))
    elif Family == 'BM':
        if mode == 'Periodic':
            print('mode switched to Symmetric')
        return(BM_wavelets(m,device='cpu'))
    
    else:
        print('Family is not implemented yet')

"""Haar_Wavelet"""
def Haar_wavelets(device='cpu',mode = 'Periodic'):
    """Haar Wavelet

    Parameters:
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions
    
    Returns:
        Haar Wavelet

    """
    Haar = torch.tensor(np.array([1/np.sqrt(2),1/np.sqrt(2)]),dtype=torch.float32)[None,None]
    m1,m2 =0,1
    
    #Define the Wavelet
    if device=='cuda':
        Haar=Haar.cuda()
    W=Wavelet(Haar,m1,m2,mode = mode)
    return(W)

def Db_wavelets(m=2,device='cpu',mode = 'Periodic'):
    """Debauchies Wavelets

    Parameters:
    m (int) : Which Wavelet we would like
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions
    
    Returns:
        Dbm Wavelet

    """
    
    if m == 2: 
      Db = torch.tensor(np.array([0.482962913145,0.836516303738,0.224143868042,-0.129409522551]),dtype=torch.float32)[None,None]
      m1,m2 =0,3
    elif m == 3 :
      Db = torch.tensor(np.array([0.332670552950,0.806891509311, 0.459877502118, -0.135011020010, -0.085441273882, 0.035226291882]),dtype=torch.float32)[None,None]
      m1,m2 =0,5
    elif m ==4 :
      Db = torch.tensor(np.array([0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417, -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785]),dtype=torch.float32)[None,None]
      m1,m2 =0,7
    elif m ==10:
        Db = torch.tensor(np.array([0.026670057901, 0.188176800078,0.527201188932,0.688459039454, 0.281172343661, -0.249846424327, -0.195946274377, 0.127369340336, 0.093057364604, -0.071394147166, -0.029457536822,0.033212674059, 0.003606553567, -0.010733175483, 0.001395351747, 0.001992405295,-0.000685856695, -0.000116466855, 0.000093588670,  0.000013264203]),dtype=torch.float32)[None,None].cuda()
        m1,m2 =0,19
    else:
      print('m is not implemented yet')    
     
    
    
    #Define the Wavelet
    if device=='cuda':
        Db=Db.cuda()
    W=Wavelet(Db,m1,m2,mode = mode)
    return(W)

def BM_wavelets(m,device='cpu'):
    """Symmetric BattleLemarié Wavelets (padding is necessarly symmetric)

    Parameters:
    m (int) : Which Wavelet we would like
    device (string): 'cpu' or 'cuda'
    
    Returns:
        BMm Wavelet

    """
    
    if m==1:
        BMT=torch.tensor(np.array([-0.000122686, -0.000224296, 0.000511636,
                        0.000923371, -0.002201945, -0.003883261, 0.009990599,
                        0.016974805, -0.051945337, -0.06910102, 0.39729643,
                        0.817645956, 0.39729643, -0.06910102, -0.051945337,
                        0.016974805, 0.009990599, -0.003883261, -0.002201945,
                        0.000923371, 0.000511636, -0.000224296, -0.000122686]
        ),dtype=torch.float32)[None,None]
        m1,m2 =11,11
    elif m==3:
        BMT=torch.tensor(np.array([ 0.000146098,-0.000232304, -0.000285414,
                        0.000462093, 0.000559952, -0.000927187, -0.001103748,
                        0.00188212, 0.002186714, -0.003882426, -0.00435384,
                        0.008201477, 0.008685294, -0.017982291, -0.017176331,
                        0.042068328, 0.032080869, -0.110036987, -0.050201753,
                        0.433923147, 0.766130398, 0.433923147, -0.050201753,
                        -0.110036987, 0.032080869, 0.042068328, -0.017176331,
                        -0.017982291, 0.008685294, 0.008201477, -0.00435384,
                        -0.003882426, 0.002186714, 0.00188212, -0.001103748,
                        -0.000927187, 0.000559952, 0.000462093, -0.000285414,
                        -0.000232304, 0.000146098]),dtype=torch.float32)[None,None]
        
        m1,m2 =20,20
    else:
        print('m = '+str(m)+' is not implemented yet')
    #Define the Wavelet
    if device=='cuda':
        BMT=BMT.cuda()
    W=Wavelet(BMT,m1,m2,mode = 'Symmetric')
    return(W)
