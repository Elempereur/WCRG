"""The Ansatz are defined here"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod
from functorch import vmap, jacrev,jacfwd, hessian,jvp

from Models_abstract import Condi_Ansatz

      

    
# SCALAR POTENTIAL
class SigmoidWindows(Condi_Ansatz):
    """ Scalar potentials given by translated Gaussian windows. """

    def __init__(self,reconstruct, centers,sigma, device='cpu'):
        
        "Will reconstruct x at the finer scale, without high frequency, and compute the scalar potential of mid freqs conditionaly to low freqs"
        num_potentials = len(centers)
        super().__init__(num_potentials)
        self.device = device #'cpu' or 'cuda'
        self.sigma = sigma #width of the sigmoids
        self.centers = centers #centers of the sigmoids
        self.reconstruct = reconstruct #Reconstruction at finer scale, takes x has mid freqs, x_condi as low freqq and sets high freqs to 0, takes (x_condi, x) (((n_batch,n_pred,Width,Width)),((n_batch,N,N)) returns (n_batch,L,L)

    def potential(self, x, x_condi):
        """Potential for single realisation 
    
        Parameters:
        x (tensor): High Frequency \bar x_j (n_pred,Width,Width)
        x_condi(tensor): Low Frequency x_j (N,N)
        
        Returns:
            (tensor) : sigmoids applied and sumed on each pixel of x_{j-1} (n_potentials,)

        """
        phi = self.reconstruct(x_condi[None], x[None])[0]  # (L,L)
        phi = phi.reshape((-1,))  # (L**2)

        return(torch.sigmoid(-(phi[None, :] - self.centers[:, None]) / (self.sigma[:,None] )).sum(1)) #(n_potentials,)

    def potential_batch(self, x, x_condi):
        """Potential for batched realisation 
    
        Parameters:
        x (tensor): High Frequency \bar x_j (n_batch,n_pred,Width,Width)
        x_condi(tensor): Low Frequency x_j (n_batch,N,N)
        
        Returns:
            (tensor) : sigmoids applied and sumed on each pixel of x_{j-1} (n_batch,n_potentials,)

        """
        # (n_batch,n_pred,W,W) and (n_batch,N,N) -> (n_batch,M,)
        phi = self.reconstruct(x_condi, x)  # (batch,L,L)
        phi = phi.reshape(phi.shape[:-2] + (-1,))  # (batch,L**2)
        SIG =torch.sigmoid(-(phi[:, None, :] - self.centers[None, :, None]) / (self.sigma[:,None] ))
        return SIG.sum(
            2)  # (n_batch,n_potentials,L**2) to (n_batch,n_potentials,)


# GAUSSIAN POTENTIAL
class GaussianPotential(Condi_Ansatz):
    """ Stationary Gaussian (covariance) potentials. We consider the quadratic interaction of x given x_condi.
    x is composed of num_varying_channels channels, x_condi is composed of num_conditioning_channels channels.
    Quadratic interactions are computed in a stationary manner, and are restricted to the given shifts. """
    
    def __init__(self,mode = 'All', num_varying_channels=1, num_conditioning_channels=0, shifts=((0, 1), (1, 0))):
    
        self.mode = mode #'All' or 'Next_Neighbors'
            
        self.num_varying_channels = num_varying_channels  # denoted as V in shapes
        self.num_conditioning_channels = num_conditioning_channels  # denotes as K in shapes
        self.num_channels = self.num_varying_channels + self.num_conditioning_channels  # denoted as C in shapes
    
        pos_shifts = torch.tensor(shifts)
        all_shifts = torch.cat((torch.zeros((1, 2)), pos_shifts, -pos_shifts))
    
        def pos_to_neg(i):
            #Convert an index of a shift to the index of its opposite. 
            if i == 0:
                return 0
            elif i <= len(shifts):
                return i + len(shifts)
            else:
                return i - len(shifts)
                
            #return(i)
    
        # We compute moments of the form sum_i x[c, i] x[d, i - s] / 2 for several channels c, d and shifts s.
        # We now build various lists of indices to have fast batched implementations of the covariances.
    
        
        if mode == 'All':
    
            indices = []  # (c, d, s_pos, s_neg, is_quad) indices
    
            for c in range(self.num_channels):
                # Do not consider covariances between conditioning channels.
                min_channel = self.num_conditioning_channels if c < self.num_conditioning_channels else c
                for d in range(min_channel, self.num_channels):
                    # Positive shifts only when c = d, add negative shifts when c != d.
                    num_shifts = 1 + len(pos_shifts) if c == d else len(all_shifts)
                    for s_pos in range(num_shifts):
                        s_neg = pos_to_neg(s_pos)
                        is_quad = c == d and s_pos == 0
                        indices.append((c, d, s_pos, s_neg, is_quad))
        
        elif mode == 'Next_Neighbors':
            indices = []  # (c, d, s_pos, s_neg, is_quad) indices
            
            M=(self.num_conditioning_channels+1)//2
            #2M-1 conditionning_channels and 2M+1 varying ones
            #Conditionning * Varying
            num_shifts =  len(all_shifts)
            for c in range(M-1):
                for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((c, c+2*M-1, s_pos, s_neg, is_quad))
                    indices.append((c+M-1, c+3*M-1, s_pos, s_neg, is_quad))
            
            for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((2*M-2, 4*M-1, s_pos, s_neg, is_quad))
                    indices.append((2*M-2, 3*M-2, s_pos, s_neg, is_quad))   
                    indices.append((2*M-2, 4*M-2, s_pos, s_neg, is_quad)) 
            
            #Varying * Varying Non_Quadratics
            for c in range(2*M-1,3*M-2):
                for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((c,c+1 , s_pos, s_neg, is_quad))
                    indices.append((c+M, c+1, s_pos, s_neg, is_quad))
            
            for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((2*M-1, 4*M-1, s_pos, s_neg, is_quad))
                    indices.append((3*M-1, 4*M-1, s_pos, s_neg, is_quad))
            
            #Quadratics
            num_shifts = 1 + len(pos_shifts)
            for c in range(2*M-1,4*M):
                for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = True and s_pos == 0
                    indices.append((c,c , s_pos, s_neg, is_quad))
                 
        else:
            raise 'Mode not Implemented'
            
        super().__init__(num_potentials=len(indices))
    
        self.register_buffer("shifts", all_shifts)  # (S, 2)
    
        indices = torch.tensor(indices)  # (M, 5), tensor for convenient indexing
        self.register_buffer("first_channel_indices", indices[:, 0])  # (M,) with values in [0:C]
        self.register_buffer("second_channel_indices", indices[:, 1])  # (M,) with values in [K:C]
        self.register_buffer("pos_shift_indices", indices[:, 2])  # (M,) with values in [0:S]
        self.register_buffer("neg_shift_indices", indices[:, 3])  # (M,) with values in [0:S]
        self.register_buffer("quad_indices", indices[:, 4].nonzero()[:, 0])  # (M',) with values in [0:M]
            
           
            
            
        
        

    def get_rolls(self, x):
        """ (*, C, L, L) to (*, S, C, L, L). """
        # torch.roll(x, s)[i] = x[i - s]
        return torch.stack([torch.roll(x, (shift[0].int(), shift[1].int()), (-2, -1)) for shift in self.shifts], dim=-4)

    def potential(self, x, x_condi):
        """Potential for single realisation 
    
        Parameters:
        x (tensor): High Frequency \bar x_j (n_pred,Width,Width)
        x_condi(tensor): Mid Frequency x_{j+1} (n_condi,Width,Width)
        
        Returns:
            (tensor) : (n_potentials) the quadratics potentials sumed on pixels

        """
        
        # The m-th moment is defined as sum_i x[c[m], i] x[d[m], i - s[m]] / 2.
        # NOTE: slightly inefficient when we have a single channel, as computing negative shifts is superfluous.
        x = torch.cat((x_condi[None],x[None]), dim=-3)  # (*, n_pred + n_condi, Width, Width)
        x_roll = self.get_rolls(x[..., self.num_conditioning_channels:, :, :])  # (*, S, n_condi, Width,Width)
        x_prod = x[..., self.first_channel_indices, :, :] * x_roll[..., self.pos_shift_indices,
                                                            self.second_channel_indices - self.num_conditioning_channels ,:, :]  # (*, n_potentials, Width, Width)
                                              
        return (x_prod.sum((-1, -2)) / 2)[0]  # (n_potentials,)

    def potential_batch(self, x, x_condi):
        """Potential for batched realisations 
    
        Parameters:
        x (tensor): High Frequency \bar x_j (n_batch,n_pred,Width,Width)
        x_condi(tensor): Mid Frequency x_{j+1} (n_batch,n_condi,Width,Width)
        
        Returns:
            (tensor) : (n_potentials) the quadratics potentials sumed on pixels

        """
        # The m-th moment is defined as sum_i x[c[m], i] x[d[m], i - s[m]] / 2.
        # NOTE: slightly inefficient when we have a single channel, as computing negative shifts is superfluous.
       
      
        x = torch.cat((x_condi,x ), dim=-3)  # (*, n_pred + n_condi, Width, Width)
        x_roll = self.get_rolls(x[..., self.num_conditioning_channels:, :, :])  # (*, S, n_condi, Width,Width)
        x_prod = x[..., self.first_channel_indices, :, :] * x_roll[..., self.pos_shift_indices, self.second_channel_indices - self.num_conditioning_channels, :, :]  # (*, n_potentials, Width, Width)
        return x_prod.sum((-1, -2)) / 2  # (*, num_potentials)
        
        
    def gradient(self, x, x_condi,theta=None):
        """ (N, C, L, L) to (N, M, V, L, L). """
        x = torch.cat((x_condi,x ), dim=-3)  # (*, C + V, L, L)
        x_roll = self.get_rolls(x) / 2  # (N, S, C, L, L)
        potentials = torch.arange(self.num_potentials, device=x.device)
        subset = self.first_channel_indices >= self.num_conditioning_channels

        # The gradient of the m-th moment with respect to x[k, j] is: (k is only a varying channel)
        # ret[m, k, j] = (delta[k, c[m]] x[d[m], j - s[m]] + delta[k, d[m]] x[c[m], j + s[m]]) / 2.
        ret = torch.zeros((x.shape[0], self.num_potentials, self.num_varying_channels) + x.shape[2:], device=x.device)  # (N, M, V, L, L)
        ret[:, potentials[subset], self.first_channel_indices[subset] - self.num_conditioning_channels] += x_roll[:, self.pos_shift_indices[subset], self.second_channel_indices[subset]]  # (N, M,V, L, L)
        ret[:, potentials, self.second_channel_indices - self.num_conditioning_channels] += x_roll[:, self.neg_shift_indices, self.first_channel_indices]  # (N, M,V, L, L)
        if theta is not None:
            
            ret = (ret*theta[:,None,None,None]).sum(1)[:,None] # (N, 1,V, L, L)
        
        return ret

    def laplacian(self, x,x_condi,theta=None):
        """ (N, C, L, L) to (N, M). """
        x = torch.cat((x_condi,x ), dim=-3)  # (*, C + V, L, L)
        # The laplacian of the m-th moment is L^2 for quadratic moments, and zero for others.
        ret = torch.zeros(x.shape[0], self.num_potentials, device=x.device)
        ret[:, self.quad_indices] = np.prod(x.shape[2:])
        if theta is not None:
            ret = (ret*theta[:]).sum(1)[:,None] # (N, 1)
        return ret
    
    def laplacian_Hutchinson(self,x,x_condi,z,theta=None):
        return(self.laplacian(x,x_condi,theta)) # (N, 1)

class GaussianPotential_NoSym(Condi_Ansatz):
    """ Stationary Gaussian (covariance) potentials. We consider the quadratic interaction of x given x_condi.
    x is composed of num_varying_channels channels, x_condi is composed of num_conditioning_channels channels.
    Quadratic interactions are computed in a stationary manner, and are restricted to the given shifts. Similar to GaussianPotential, except the shifts are not symetrized.
    """
    
    def __init__(self,mode = 'All', num_varying_channels=1, num_conditioning_channels=0, shifts=()):
    
        self.mode = mode #'All' or 'Next_Neighbors'
            
        self.num_varying_channels = num_varying_channels  # denoted as V in shapes
        self.num_conditioning_channels = num_conditioning_channels  # denotes as K in shapes
        self.num_channels = self.num_varying_channels + self.num_conditioning_channels  # denoted as C in shapes
    
        pos_shifts = torch.tensor(shifts)
        all_shifts = torch.cat((torch.zeros((1, 2)), pos_shifts))
    
        def pos_to_neg(i):
            #Convert an index of a shift to the index of its opposite. 
            #if i == 0:
            #    return 0
            #elif i <= len(shifts):
            #    return i + len(shifts)
            #else:
            #    return i - len(shifts)
                
            return(i)
    
        # We compute moments of the form sum_i x[c, i] x[d, i - s] / 2 for several channels c, d and shifts s.
        # We now build various lists of indices to have fast batched implementations of the covariances.
    
        
        if mode == 'All':
    
            indices = []  # (c, d, s_pos, s_neg, is_quad) indices
    
            for c in range(self.num_channels):
                # Do not consider covariances between conditioning channels.
                min_channel = self.num_conditioning_channels if c < self.num_conditioning_channels else c
                for d in range(min_channel, self.num_channels):
                    # Positive shifts only when c = d, add negative shifts when c != d.
                    num_shifts = 1 + len(pos_shifts) if c == d else len(all_shifts)
                    for s_pos in range(num_shifts):
                        s_neg = pos_to_neg(s_pos)
                        is_quad = c == d and s_pos == 0
                        indices.append((c, d, s_pos, s_neg, is_quad))
        
        elif mode == 'Next_Neighbors':
            indices = []  # (c, d, s_pos, s_neg, is_quad) indices
            
            M=(self.num_conditioning_channels+1)//2
            #2M-1 conditionning_channels and 2M+1 varying ones
            #Conditionning * Varying
            num_shifts =  len(all_shifts)
            for c in range(M-1):
                for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((c, c+2*M-1, s_pos, s_neg, is_quad))
                    indices.append((c+M-1, c+3*M-1, s_pos, s_neg, is_quad))
            
            for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((2*M-2, 4*M-1, s_pos, s_neg, is_quad))
                    indices.append((2*M-2, 3*M-2, s_pos, s_neg, is_quad))   
                    indices.append((2*M-2, 4*M-2, s_pos, s_neg, is_quad)) 
            
            #Varying * Varying Non_Quadratics
            for c in range(2*M-1,3*M-2):
                for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((c,c+1 , s_pos, s_neg, is_quad))
                    indices.append((c+M, c+1, s_pos, s_neg, is_quad))
            
            for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = False and s_pos == 0
                    indices.append((2*M-1, 4*M-1, s_pos, s_neg, is_quad))
                    indices.append((3*M-1, 4*M-1, s_pos, s_neg, is_quad))
            
            #Quadratics
            num_shifts = 1 + len(pos_shifts)
            for c in range(2*M-1,4*M):
                for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = True and s_pos == 0
                    indices.append((c,c , s_pos, s_neg, is_quad))
                 
        else:
            raise 'Mode not Implemented'
            
        super().__init__(num_potentials=len(indices))
    
        self.register_buffer("shifts", all_shifts)  # (S, 2)
    
        indices = torch.tensor(indices)  # (M, 5), tensor for convenient indexing
        self.register_buffer("first_channel_indices", indices[:, 0])  # (M,) with values in [0:C]
        self.register_buffer("second_channel_indices", indices[:, 1])  # (M,) with values in [K:C]
        self.register_buffer("pos_shift_indices", indices[:, 2])  # (M,) with values in [0:S]
        self.register_buffer("neg_shift_indices", indices[:, 3])  # (M,) with values in [0:S]
        self.register_buffer("quad_indices", indices[:, 4].nonzero()[:, 0])  # (M',) with values in [0:M]
            
           
            
            
        
        

    def get_rolls(self, x):
        """ (*, C, L, L) to (*, S, C, L, L). """
        # torch.roll(x, s)[i] = x[i - s]
        return torch.stack([torch.roll(x, (shift[0].int(), shift[1].int()), (-2, -1)) for shift in self.shifts], dim=-4)

    def potential(self, x, x_condi):
        """ (C, L, L) and (V, L, L) to (M). """
        # The m-th moment is defined as sum_i x[c[m], i] x[d[m], i - s[m]] / 2.
        # NOTE: slightly inefficient when we have a single channel, as computing negative shifts is superfluous.
        x = torch.cat((x_condi[None],x[None]), dim=-3)  # (*, C + V, L, L)
        x_roll = self.get_rolls(x[..., self.num_conditioning_channels:, :, :])  # (*, S, V, L, L)
        x_prod = x[..., self.first_channel_indices, :, :] * x_roll[..., self.pos_shift_indices,
                                                            self.second_channel_indices - self.num_conditioning_channels ,:, :]  # (*, M, L, L)
        return (x_prod.sum((-1, -2)) / 2)[0]  # (*, M)

    def potential_batch(self, x, x_condi):
        """ (*,C, L, L) and (*,V, L, L) to (*, M). """
        # The m-th moment is defined as sum_i x[c[m], i] x[d[m], i - s[m]] / 2.
        # NOTE: slightly inefficient when we have a single channel, as computing negative shifts is superfluous.
       
      
        x = torch.cat((x_condi,x ), dim=-3)  # (*, C + V, L, L)
        x_roll = self.get_rolls(x[..., self.num_conditioning_channels:, :, :])  # (*, S, V, L, L)
        x_prod = x[..., self.first_channel_indices, :, :] * x_roll[..., self.pos_shift_indices, self.second_channel_indices - self.num_conditioning_channels, :, :]  # (*, M, L, L)
        return x_prod.sum((-1, -2)) / 2  # (*, M)
        
    def gradient(self, x, x_condi,theta=None):
        """ (N, C, L, L) to (N, M, V, L, L). """
        x = torch.cat((x_condi,x ), dim=-3)  # (*, C + V, L, L)
        x_roll = self.get_rolls(x) / 2  # (N, S, C, L, L)
        potentials = torch.arange(self.num_potentials, device=x.device)
        subset = self.first_channel_indices >= self.num_conditioning_channels

        # The gradient of the m-th moment with respect to x[k, j] is: (k is only a varying channel)
        # ret[m, k, j] = (delta[k, c[m]] x[d[m], j - s[m]] + delta[k, d[m]] x[c[m], j + s[m]]) / 2.
        ret = torch.zeros((x.shape[0], self.num_potentials, self.num_varying_channels) + x.shape[2:], device=x.device)  # (N, M, V, L, L)
        ret[:, potentials[subset], self.first_channel_indices[subset] - self.num_conditioning_channels] += x_roll[:, self.pos_shift_indices[subset], self.second_channel_indices[subset]]  # (N, M,V, L, L)
        ret[:, potentials, self.second_channel_indices - self.num_conditioning_channels] += x_roll[:, self.neg_shift_indices, self.first_channel_indices]  # (N, M,V, L, L)
        if theta is not None:
            
            ret = (ret*theta[:,None,None,None]).sum(1)[:,None] # (N, 1,V, L, L)
        
        return ret

    def laplacian(self, x,x_condi,theta=None):
        """ (N, C, L, L) to (N, M). """
        x = torch.cat((x_condi,x ), dim=-3)  # (*, C + V, L, L)
        # The laplacian of the m-th moment is L^2 for quadratic moments, and zero for others.
        ret = torch.zeros(x.shape[0], self.num_potentials, device=x.device)
        ret[:, self.quad_indices] = np.prod(x.shape[2:])
        if theta is not None:
            ret = (ret*theta[:]).sum(1)[:,None] # (N, 1)
        return ret
    
    def laplacian_Hutchinson(self,x,x_condi,z,theta=None):
        return(self.laplacian(x,x_condi,theta)) # (N, 1)
        

########################################################################################################################
#GAUSSIAN NON CONDITIONNAL
########################################################################################################################
class GaussianNonCondi(Condi_Ansatz):
    """ Stationary Gaussian (covariance) potentials. We consider the quadratic interaction of x 
    x is composed of num_varying_channels channels. Quadratic interactions are computed in a stationary manner, and are restricted to the given shifts. """
    
    def __init__(self, num_varying_channels=1, num_conditioning_channels=0, shifts=((0, 1), (1, 0))):
    
        self.num_varying_channels = num_varying_channels  # denoted as V in shapes
        self.num_conditioning_channels = num_conditioning_channels  # denotes as K in shapes
        self.num_channels = self.num_varying_channels + self.num_conditioning_channels  # denoted as C in shapes
    
        pos_shifts = torch.tensor(shifts)
        all_shifts = torch.cat((torch.zeros((1, 2)), pos_shifts, -pos_shifts))
    
        def pos_to_neg(i):
            #Convert an index of a shift to the index of its opposite. 
            if i == 0:
                return 0
            elif i <= len(shifts):
                return i + len(shifts)
            else:
                return i - len(shifts)
                
            #return(i)
    
        # We compute moments of the form sum_i x[c, i] x[d, i - s] / 2 for several channels c, d and shifts s.
        # We now build various lists of indices to have fast batched implementations of the covariances.
    
        
        
    
        indices = []  # (c, d, s_pos, s_neg, is_quad) indices
    
        for c in range(self.num_channels):
            # Do not consider covariances between conditioning channels.
            min_channel = self.num_conditioning_channels if c < self.num_conditioning_channels else c
            for d in range(min_channel, self.num_channels):
                # Positive shifts only when c = d, add negative shifts when c != d.
                num_shifts = 1 + len(pos_shifts) if c == d else len(all_shifts)
                for s_pos in range(num_shifts):
                    s_neg = pos_to_neg(s_pos)
                    is_quad = c == d and s_pos == 0
                    indices.append((c, d, s_pos, s_neg, is_quad))
        
        
            
        super().__init__(num_potentials=len(indices))
    
        self.register_buffer("shifts", all_shifts)  # (S, 2)
    
        indices = torch.tensor(indices)  # (M, 5), tensor for convenient indexing
        self.register_buffer("first_channel_indices", indices[:, 0])  # (M,) with values in [0:C]
        self.register_buffer("second_channel_indices", indices[:, 1])  # (M,) with values in [K:C]
        self.register_buffer("pos_shift_indices", indices[:, 2])  # (M,) with values in [0:S]
        self.register_buffer("neg_shift_indices", indices[:, 3])  # (M,) with values in [0:S]
        self.register_buffer("quad_indices", indices[:, 4].nonzero()[:, 0])  # (M',) with values in [0:M]
            
    def get_rolls(self, x):
        """ (*, C, L, L) to (*, S, C, L, L). """
        # torch.roll(x, s)[i] = x[i - s]
        return torch.stack([torch.roll(x, (shift[0].int(), shift[1].int()), (-2, -1)) for shift in self.shifts], dim=-4)

    def potential(self, x, x_condi):
        """ (C, L, L) and (V, L, L) to (M). """
        # The m-th moment is defined as sum_i x[c[m], i] x[d[m], i - s[m]] / 2.
        # NOTE: slightly inefficient when we have a single channel, as computing negative shifts is superfluous.
        x_roll = self.get_rolls(x[..., self.num_conditioning_channels:, :, :])  # (*, S, V, L, L)
        x_prod = x[..., self.first_channel_indices, :, :] * x_roll[..., self.pos_shift_indices,
                                                            self.second_channel_indices - self.num_conditioning_channels ,:, :]  # (*, M, L, L)
                                              
        return (x_prod.sum((-1, -2)) / 2)[0]  # (*, M)

    def potential_batch(self, x, x_condi):
        """ (*,C, L, L) and (*,V, L, L) to (*, M). """
        # The m-th moment is defined as sum_i x[c[m], i] x[d[m], i - s[m]] / 2.
        # NOTE: slightly inefficient when we have a single channel, as computing negative shifts is superfluous.
       
        x_roll = self.get_rolls(x[..., self.num_conditioning_channels:, :, :])  # (*, S, V, L, L)
        x_prod = x[..., self.first_channel_indices, :, :] * x_roll[..., self.pos_shift_indices, self.second_channel_indices - self.num_conditioning_channels, :, :]  # (*, M, L, L)
        return x_prod.sum((-1, -2)) / 2  # (*, M)
        
    def gradient(self, x, x_condi,theta=None):
        """ (N, C, L, L) to (N, M, V, L, L). """
        x_roll = self.get_rolls(x) / 2  # (N, S, C, L, L)
        potentials = torch.arange(self.num_potentials, device=x.device)
        subset = self.first_channel_indices >= self.num_conditioning_channels

        # The gradient of the m-th moment with respect to x[k, j] is: (k is only a varying channel)
        # ret[m, k, j] = (delta[k, c[m]] x[d[m], j - s[m]] + delta[k, d[m]] x[c[m], j + s[m]]) / 2.
        ret = torch.zeros((x.shape[0], self.num_potentials, self.num_varying_channels) + x.shape[2:], device=x.device)  # (N, M, V, L, L)
        ret[:, potentials[subset], self.first_channel_indices[subset] - self.num_conditioning_channels] += x_roll[:, self.pos_shift_indices[subset], self.second_channel_indices[subset]]  # (N, M,V, L, L)
        ret[:, potentials, self.second_channel_indices - self.num_conditioning_channels] += x_roll[:, self.neg_shift_indices, self.first_channel_indices]  # (N, M,V, L, L)
        if theta is not None:
            
            ret = (ret*theta[:,None,None,None]).sum(1)[:,None] # (N, 1,V, L, L)
        
        return ret
        
    
    def laplacian(self, x,x_condi,theta=None):
        """ (N, C, L, L) to (N, M). """
        # The laplacian of the m-th moment is L^2 for quadratic moments, and zero for others.
        ret = torch.zeros(x.shape[0], self.num_potentials, device=x.device)
        ret[:, self.quad_indices] = np.prod(x.shape[2:])
        if theta is not None:
            ret = (ret*theta[:]).sum(1)[:,None] # (N, 1)
        return ret
    
    def laplacian_Hutchinson(self,x,x_condi,z,theta=None):
        return(self.laplacian(x,x_condi,theta)) # (N, 1)