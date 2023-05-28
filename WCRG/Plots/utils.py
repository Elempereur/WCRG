import torch
import numpy as np

def azimuthalAverage(image, center=None, Fourier=True):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    '''added modification a and b'''
    a, b = image.shape

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], (y - center[1]) * a / b)
    if Fourier == False:
        r = np.hypot(x - center[0], (y - center[1]))

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

   
def Square_laplacian(ansatz_union,Free=False,index_quad=1,device='cuda'):
        """Compute the Trace of the Square potential.
    
        Parameters:
        ansatz_union (Condi_ansatz object)
        Free : Whether ansatz_union is a Free Energy (True) or not (False)
        index_quad (int): position of the quad potential in ansatz_union.ansatze
        
        Returns:
            Trace / number of variables
    
        """
        L = ansatz_union.L #input shape of what the model takes,
        x = torch.zeros((1,L,L),device=device)
        wav_condi, wav_hf = ansatz_union.decompose(x) #(n,N,N), (n,n_condi,Width,Width), (n,n_pred,Width,Width)
        #The Trace is the same everywhere, we compute it in 0
        
        #Variables
        n_variables = len(wav_hf[0].reshape((-1)))
        
        #Num_potentials
        n_pot = 0
        for i in range(0,index_quad):
            n_pot+=ansatz_union.ansatze[i].num_potentials
        n_quad = ansatz_union.ansatze[index_quad].num_potentials
            
        #Laplacian
        if Free ==False:
          hess = ansatz_union.ansatze[1].laplacian(wav_hf,wav_condi[1],ansatz_union.theta()[n_pot:n_pot+n_quad]) #(n,1)
          
        else:
          hess = ansatz_union.ansatze[1].laplacian(wav_hf,wav_condi[0],ansatz_union.theta()[n_pot:n_pot+n_quad]) #(n,1)
        
        #For Langevin
        return(hess.mean()/n_variables)