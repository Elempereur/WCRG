"""Langevin Sampling"""

import numpy as np
import torch
import scipy.fftpack as sfft


#LANGEVIN
def Langevin(ansatz_union,x_0,x_condi,n_steps,step_size,window_min,window_max) :
    """Conditionnal Windowed Langevin Dynamic
    
    Parameters:
    ansatz_union (Condi_ansatz object) : The ansatz giving the gradient for Langevin Dynamic
    x_0 (tensor) : (n_batch,n_chanels,N,N) Seed from which we start Langevin Dynamic
    x_condi (tensor) : (n_batch,n_condi,N',N') each batch element i is conditionned from x_condi[i]
    n_steps (int) : number of steps 
    step_size (float) : step size  
    window_min (float) : keep the update only if the reconstruction has no pixel < window_min
    window_max (float) : keep the update only if the reconstruction has no pixel > window_max

    
    
    Returns:
        x (tensor) :Result of Langevin Dynamic (n_batch,n_chanels,N,N)
    
    """
    
  
    theta = ansatz_union.theta().detach() # (n_potentials,)
    
    
    x=torch.clone(x_0)
    
    for _ in range(n_steps):
      # x (*,n_1,n_2,n_3)
      """Won't work if  has another shape"""
      # x_condi tuple of (*,n_1,..)
      #Gradient computation
      gradient = ansatz_union.gradient(x,x_condi,theta) # (*,1,n_1,n_2,n_3)
      gradient = gradient[:,0] # (*,n_1,n_2,n_3)
      #Noise
      noise = np.sqrt(2*step_size)*torch.randn_like(x) # (*,n_1,n_2,n_3)
      x_new = x - step_size * gradient + noise # (*,n_1,n_2,n_3)
      #Reconstruction from x_condi and x_new
      x_rec = ansatz_union.reconstruct(x_condi[0],x_new).abs()
      
      #Windowed Langevin, do not update outside of the window
      ind_max = torch.where(torch.max(x_rec.reshape((x_rec.shape[0],-1)).abs(),1)[0]>window_max)
      ind_min = torch.where(torch.min(x_rec.reshape((x_rec.shape[0],-1)).abs(),1)[0]<window_min)
      x_new[ind_max] = x[ind_max]
      x_new[ind_min] = x[ind_min]
      #Update
      x = x_new
      
    
    return(x)
  
# MALA
def Mala(ansatz_union,x_0,x_condi,n_steps,step_size,window_min,window_max) :
    """Conditionnal Windowed MALA Dynamic
    
    Parameters:
    ansatz_union (Condi_ansatz object) : The ansatz giving the gradient for MALA Dynamic
    x_0 (tensor) : (n_batch,n_chanels,N,N) Seed from which we start MALA dynamic
    x_condi (tensor) : (n_batch,n_condi,N',N') each batch element i is conditionned from x_condi[i]
    n_steps (int) : number of steps 
    step_size (float) : step size  
    window_min (float) : keep the update only if the reconstruction has no pixel < window_min
    window_max (float) : keep the update only if the reconstruction has no pixel > window_max

    
    
    Returns:
        x (tensor) :Result of MALA Dynamic (n_batch,n_chanels,N,N)
    
    """
    
    theta = ansatz_union.theta().detach() # (n_potentials,)
    
    x=torch.clone(x_0)
    n_batch = len(x)
    for _ in range(n_steps):
      # x (*,n_1,n_2,n_3)
      """Won't work if  has another shape"""
      # x_condi tuple of (*,n_1,..)
      #COMPUTE GRADIENT IN X_t
      gradient = ansatz_union.gradient(x,x_condi,theta) # (*,1,n_1,n_2,n_3)
      gradient = gradient[:,0] # (*,n_1,n_2,n_3)
      #COMPUTE NOISE
      noise = np.sqrt(2*step_size)*torch.randn_like(x) # (*,n_1,n_2,n_3)

      x_new = x - step_size * gradient + noise # (*,n_1,n_2,n_3)


      #METROPOLIS Rule
      gradient_new = ansatz_union.gradient(x_new,x_condi,theta) # (*,1,n_1,n_2,n_3)
      gradient_new = gradient_new[:,0]# (*,n_1,n_2,n_3)
    
      log_qx,log_qx_new  = log_Q(x_new.reshape((x.shape[0],-1)), x.reshape((x.shape[0],-1)),gradient.reshape((x.shape[0],-1)),gradient_new.reshape((x.shape[0],-1)), step_size) #(n_batch)
      
      
      log_pix = - ansatz_union.potential_batch(x,x_condi,theta).sum(1) #(*,)
      log_pix_new = - ansatz_union.potential_batch(x_new,x_condi,theta).sum(1) #(*,)
    
      log_ratio = log_pix_new-log_pix+log_qx - log_qx_new
      
      
    
      
      #Windowed Langevin, do not update outside of the window
      x_rec = ansatz_union.reconstruct(x_condi[0],x_new)
      #Both conditions
      ind_max =  torch.where(torch.max(x_rec.reshape((x_rec.shape[0],-1)),1)[0]>window_max)[0]
      ind_min =  torch.where(torch.max(-x_rec.reshape((x_rec.shape[0],-1)),1)[0]>-window_min)[0]
      RANDOM = torch.rand(log_ratio.shape,device = log_ratio.device )
      ind_mala = torch.where((RANDOM-torch.exp(log_ratio))>0)[0]
      if _%10 ==0:
        print('Acceptance_rate ='+str(1-len(ind_mala)/n_batch))
      x_new[ind_min] = x[ind_min]
      x_new[ind_max] = x[ind_max]
      x_new[ind_mala] = x[ind_mala]
      x = x_new
      
       
    
    return(x)

def log_Q(x_prime, x,grad_x,grad_x_prime, step_size):
    """MALA transition proba
    
    Parameters:
    x_prime (tensor): x_n+1}
    x (tensor): x_{n}
    grad_x (tensor): \nabla{log p}x_{n}
    grad_x_prime (tensor): \nabla{log p}x_{n+1}
    step_size (float)

    
    
    Returns:
        log_qx (tensor) :log q(x{x_n}) with q = MALA transition proba
        log_qx_prime (tensor) :log q(x{x_{n+1}}) with q = MALA transition proba
    
    """
    log_qx_prime = -(torch.norm(x_prime - x + step_size * grad_x, p=2, dim=1) ** 2) / (4 * step_size)
    log_qx = -(torch.norm(x - x_prime + step_size * grad_x_prime, p=2, dim=1) ** 2) / (4 * step_size)
    return log_qx,log_qx_prime


def LANGEVINMALA(ansatz_union,
             x_0,
             window_min,
             window_max,
             n_steps,
             step_size,
             n_batch,
             n_repeat, Show_langevin = None,x_compare=None):
                 
    """Conditionnal Windowed MALA Dynamic
    
    Parameters:
    ansatz_union (Condi_ansatz object) : The ansatz giving the gradient for MALA Dynamic
    x_0 (tensor) : (n_batch,L,L), dynamic is conditionned by x_0 low frequencies
    window_min (float) : keep the update only if the reconstruction has no pixel < window_min
    window_max (float) : keep the update only if the reconstruction has no pixel > window_max
    n_steps (int) : number of steps 
    step_size (float) : step size  
    n_batch (int) : MALA is applied to batches of size of size n_batch ( from x_0[n_batch*k:n_batch*(k+1)] low frequencies )
    n_repeat (int) : number of batches (k in range(n_repeat) )
    Show_langevin (func) : if specified, a plot function Show_langevin(x_0,x_reco) is runed
    x_compare (tensor) : if specified, a plot function Show_langevin(x_compare,x_reco) is runed
    

    
    
    Returns:
        x_reco (tensor) :Result of MALA Dynamic reconstructed using low frequencies from which it has been sampled (n_batch*n_repeat,L,L)
    
    """

    x_reco=[]
    for k in range(n_repeat):  
      
      x = x_0[n_batch*k:n_batch*(k+1)].cuda()
      x_condi,x_hf = ansatz_union.decompose(x)
      x_zeros = torch.zeros(x_hf.shape,device=x_hf.device)
      #x_zeros = torch.randn_like(x_hf)*(window_max-window_min)/2
      x_langevin = Mala(ansatz_union,x_zeros,x_condi,n_steps,step_size,window_min,window_max)
      x_reco.append(ansatz_union.reconstruct(torch.repeat_interleave(x_condi[0], 1, dim=0),x_langevin).cpu())

    #Concatenate the sampled images
    x_reco = torch.concat(x_reco)
    if x_compare is None :
      x_compare = x_0
    #Compare to x_0 where we cut higher frequencies, if some where still there! 
    x_condi,x_hf = ansatz_union.decompose(x_compare)
    x = ansatz_union.reconstruct(x_condi[0],x_hf)

    #show
    if Show_langevin is not None:
        Show_langevin(x,x_reco)
    return(x_reco)
  