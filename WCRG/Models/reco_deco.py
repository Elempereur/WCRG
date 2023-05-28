"""Different modules that use wavelet transform and dispatch variable and conditionning variable"""

import torch
import torch.nn as nn

from .bands import decompose_concat, reconstruct


################################################# For Non-Conditionnial ansatz #################################################
class reconstruct_no_condi(nn.Module):
      def __init__(self):
        super().__init__()
      def forward(self,x_condi,x):
        """x (B,1,L,L) to (B,L,L) """
        return(x[:,0])

class decompose_no_condi(nn.Module):
      def __init__(self):
        super().__init__()
      def forward(self, x):
        """x (B,L,L) """
        """The first coordinate is used as conditionning ((B,L,L),(B,1,L,L)), the second as variable (B,1,L,L)"""
        return((torch.zeros_like(x),torch.zeros_like(x)[:,None]),x[:,None])

################################################# For Wavelet Transform #################################################
class reconstruct_wav(nn.Module):
    def __init__(self,W):
        super().__init__()
        self.W = W #Wavelet object
    def forward(self,very_low, hf):
        """ Reconstructs x_{j-1} from x_j and \bar x_j
    
        Parameters:
        very low (tensor): (B,L/2,L/2) low frequencies x_j
        hf (tensor): (B,3,L/2,L/2) high frequencies \bar x_j
        
        
        Returns:
            (tensor) :(B,L,L) reconstructed x_{j-1} with inverse wavelet transform
        
        """
        return(self.W.Inv_2d(very_low,hf[:,0],hf[:,1],hf[:,2]))
    
class decompose_wav(nn.Module):
    def __init__(self,W):
        super().__init__()
        self.W = W #Wavelet object
    def forward(self,x):
        """ decomposes x_{j-1} into (x_j,x_j[:,None]) and \bar x_j
    
        Parameters:
        x (tensor): (B,L,L) x_{j-1}
        
        Returns:
            (tuple of tensor): (B,L/2,L/2),(B,1,L/2,L/2) x_j
            (tensor): (B,3,L/2,L/2) \bar x_j
        
        """
        dec0,dec1,dec2,dec3 =self.W.Wav_2d(x)
        return((dec0,dec0[:,None]),torch.concat([dec1[:,None],dec2[:,None],dec3[:,None]],axis=1))
        
################################################# For wavelet packets #################################################
class reconstruct_wav_packet(nn.Module):
    def __init__(self,W,tree,L):
        super().__init__()
        self.W = W #Wavelet object
        self.tree = tree
        self.L = L
    def forward(self,very_low, hf):
        """ Reconstructs x_{j-1} from x_j and \bar x_j
    
        Parameters:
        very low (tensor): (B,N,N) low frequencies x_j
        hf (tensor): (B,C,N',N') high frequencies \bar x_j
        
        
        Returns:
            (tensor) :(B,L,L) reconstructed x_{j-1} with inverse wavelet transform
        
        """
        return(reconstruct(very_low, hf, self.L, self.W, self.tree))

class decompose_wav_packet(nn.Module):
    def __init__(self,W,tree,L,Width,N):
        super().__init__()
        self.W = W #Wavelet object
        self.tree = tree
        self.L = L
        self.Width = Width
        self.N = N
    def forward(self,x):
        """ decomposes x_{j-1} into (x_j,\bar x_{j+1}) and \bar x_j
    
        Parameters:
        x (tensor): (B,L,L) x_{j-1}
        
        Returns:
            (tuple of tensor): (B,N,N),(B,C,Width,Width) x_j \bar x_{j+1}
            (tensor): (B,C",Width,Width) \bar x_j
        
        """
        return(decompose_concat(x, self.N,self.Width, self.L, self.W, self.tree))

#\bar x_{j+1} and \bar x_{j+2} for wavelet packets
class decompose_inter_plus(nn.Module):
    def __init__(self,W,tree,L,Width,N):
        super().__init__()
        self.W = W #Wavelet object
        self.tree = tree
        self.L = L
        self.Width = Width
        self.N = N
    def forward(self,x):
        """ decomposes x_{j-1} into (x_j,\bar x_{j+1,2}) and \bar x_j
    
        Parameters:
        x (tensor): (B,L,L) x_{j-1}
        
        Returns:
            (tuple of tensor): (B,N,N),(B,C,Width,Width) x_j \bar x_{j+1,2}
            (tensor): (B,C",Width,Width) \bar x_j
        
        """
        x_condi1,x1 = decompose_concat(x, self.N,self.Width, self.L,self.W, self.tree)
        x_condi2,x2 = decompose_concat(x, self.N-self.Width,self.Width, self.L,self.W, self.tree)
        x_condi = (x_condi1[0], torch.concat([x_condi1[1],x_condi2[1]],axis=1))
        return((x_condi,x1))

################################################# Free Energy #################################################
class reconstruct_free(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x_condi, x):
        """ Reconstructs x_{j-1} from x_j and \bar x_j
    
        Parameters:
        x (tensor): (B,1,L/2,L/2) low frequencies x_j
        
        Returns:
            (tensor) :(B,1,L,L) low frequencies x_j
        
        """
        return(x[:,0])
    
class decompose_free(nn.Module):
    def __init__(self,W):
        super().__init__()
        self.W = W #Wavelet object
    def forward(self,x):
        """ decomposes x_{j-1} into (x_j,x_j[:,None]) and \bar x_j
    
        Parameters:
        x (tensor): (B,L,L) x_{j-1}
        
        Returns:
            (tuple of tensor): (B,L/2,L/2) Nothing usefull
            (tensor): (B,3,L/2,L/2) \bar x_j
        
        """
        x_wav = self.W.Wav_2d(x)[0][:,None] # x_{j} (B,1,L/2,L/2)
        return ((x_wav,),x_wav) #We trick by giving _wav also as x_condi, but we never use it!
        