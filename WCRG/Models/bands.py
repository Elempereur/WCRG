import torch


def decompose_bands(Matrix, N, Width):
    """Give him a Matrix, it will return the frequency bands asked for, ie the square blocks between N and N+widhts in both axis
    
    Parameters:
    Matrix (tensor): (n_batch,L,L)
    N (int): position of the band
    Width (int); width of the band
    
    Returns:
        (tensor) : (n_batch,2*N/Width+1,Width,Width) 
    
    """
    
    if N % Width == 0:
        n = N // Width
    else:
        print('Width is asked to divide N')
    
    

    Mat_1 = torch.concat([Matrix[:, None, N:N + Width, j * Width:(j + 1) * Width] for j in range(n)], axis=-3)
    Mat_2 = torch.concat([Matrix[:, None, j * Width:(j + 1) * Width, N:N + Width] for j in range(n)], axis=-3)
    Mat_3 = Matrix[:, None, N:N + Width, N:N + Width]
    
    return (torch.concat([Mat_1, Mat_2, Mat_3], axis=1))


    


# This is for the case when we are not trying the first band!, ie N>L/2
def decompose(x, N, Width,L,W,tree):  # (*, dim_tot) to (*,N,N), (*,n_condi,Width,Width), (*,n_pred,Width,Width)
    """decomposes x_{j-1} into (x_j,\bar x_{j+1},\bar x_j)
    
    Parameters:
    x (tensor): (n_batch,L**2)
    N (int) : position of the bandwidth to use as \bar x_j
    Width (int) : width of the band, for an image L*L, the bandwith is between [N:N+width]
    L (int): system size ( of x_{j-1}) = L*L 
    W (Wavelet) : Wavelet to perfom fast wavelet transform
    tree (Tree) : the tree for wavelet packet
    
    Returns:
    very_low (tensor): low frequencies x_j (this is a concatenation of mid frequencies bar x and a low x )(n_batch,N,N)
    wav_lf (tensor): mid frequency \bar x_{j+1} (expect if N=L/2, then it's the concatenation of \bar x_j+1 and \bar x_j+2) (n_batch,2*n_bands-1,Width,Width)
    wav_hf (tensor): high frequency \bar x_j (n_batch,2*n_bands+1,Width,Width)
        
    
    """
    
    
    # x ~ (...,L**d)
    batch_shape = x.shape[:1]

    x = x.reshape(batch_shape + (L,) * 2)  # (*,L,L)
    wav = W.Packet_2d(x, tree)  # ~ (*,L,L)

    # hf we are trying to predict
    wav_hf = decompose_bands(wav, N, Width)  # ~ (*,n_pred,Width,Width)
    # lf we will use for conditionning in the gaussian potential
    wav_lf = decompose_bands(wav, N - Width, Width)  # ~ (*,n_condi,Width,Width)

    # we also add the lower freqs we will use for reconstruction later
    very_low = wav[:, :N, :N]
    return very_low, wav_lf, wav_hf

def decompose_concat(x, N, Width,L,W,tree):
    """decomposes x_{j-1} into ((x_j,\bar x_{j+1}),\bar x_j)
    
    Parameters:
    x (tensor): (n_batch,L**2)
    N (int) : position of the bandwidth to use as \bar x_j
    Width (int) : width of the band, for an image L*L, the bandwith is between [N:N+width]
    L (int): system size ( of x_{j-1}) = L*L 
    W (Wavelet) : Wavelet to perfom fast wavelet transform
    tree (Tree) : the tree for wavelet packet
    
    Returns:
    wav_condi (tuple of tensors): conditioning variable for ansatz (x_j,\bar x_{j+1})
    wav_hf (tensor) :  variable for ansatz \bar x_j
        
    
    """
    
    # (*, dim_tot) to ((,N,N), (*,n_condi,Width,Width),) (*,n_pred,Width,Width)
    very_low, wav_lf, wav_hf = decompose(x, N, Width,L,W,tree)
    wav_condi = (very_low, wav_lf)
    return(wav_condi,wav_hf)


def reconstruct(very_low, hf, L,W,tree):
    """reconstruct x_{j-1} from x_j and \bar x_j into a finer grid, setting possible higher frequencies to 0
    
    Parameters:
    very_low (tensor): low frequencies x_j (this is a concatenation of mid frequencies bar x and a low x )(n_batch,N,N)
    hf (tensor): high frequency \bar x_j (n_batch,2*n_bands+1,Width,Width)
    L (int): dyadic grid size for x_{j-1}) = L*L 
    W (Wavelet) : Wavelet to perfom inverse fast wavelet transform
    tree (Tree) : the tree for wavelet packet
    
    Returns:
    (tensor): Reconstructed map x_{j-1} (n_batch,L,L)
        
    
    """
    # (*,N,N),  (*,n_pred,Width,Width) to (*,L,L)
    # L is the size of what we expect to reconstruct

    batch_shape = very_low.shape[:1]
    Nh = very_low.shape[2]
    Width = hf.shape[3]
    nh = Nh // Width
    
    #avoid -1
    #Number of submaps 
    n_max = len(hf[0,:,0,0])-1
    
    #low freqs
    wav = very_low #(*,Nh,Nh)
    
    #high freqs
    unfold1 = torch.nn.Unfold(kernel_size=(1,Width), dilation=1, padding=0, stride=1)
    wav_1 = unfold1(hf[:, :nh, :, :]).transpose(1,2) #(*,Width, Nh)
    
    wav = torch.cat([wav,wav_1],axis=1 ) # #(*,Nh+Width, Nh)
    
    unfold2 = torch.nn.Unfold(kernel_size=(Width,1), dilation=1, padding=0, stride=1)
    wav_2 = unfold2(hf[:, nh:2*nh, :, :]) #(*,Nh, Width)
    
    wav_3 = torch.cat([wav_2,hf[:, n_max, :, :]],axis=1) #(*,Nh+Width, Width)
    
    wav = torch.cat([wav,wav_3],axis =2) #(*,Nh+Width, N_h+Width)
    
    #fill with zeros the higher frequencies
    
    wav = torch.cat([wav,torch.zeros(batch_shape + (L-Nh-Width,Nh+Width),device=wav.device)],axis=1) #(*,L, N_h+Width)
    wav = torch.cat([wav,torch.zeros(batch_shape + (L,L-Nh-Width),device=wav.device)],axis=2) #(*,L,L)
    
    
    
    #inverse wavelet transform
    return W.Inv_Packet_2d(wav, tree)





