"""Handles everything plot related"""

import numpy as np
import torch
import scipy.fftpack as sfft
import matplotlib.pyplot as plt
from .utils import azimuthalAverage, Square_laplacian

#PLOTS
def Compare_Spectrum(DATA,X_fake,log=False):
    """Compares Radially Averaged Fourier Spectrum of original DATA and synthesis data X_fake

    Parameters:
    DATA (tensor): (n_batch,L,L)
    X_fake (tensor): (n_batch,L,L)
    log (Bool): if True log scale

    """
    DATA = DATA.detach().cpu().numpy()
    X_fake = X_fake.detach().cpu().numpy()
    true=azimuthalAverage(sfft.fftshift((np.abs(np.fft.fft2(DATA))**2).mean(axis=0)))
    langevin=azimuthalAverage(sfft.fftshift((np.abs(np.fft.fft2(X_fake))**2).mean(axis=0)))
    plt.plot(true, label='Original')
    plt.plot(langevin, label='Synthesis')
    plt.legend()
    if log == True:
        plt.yscale('log')
        plt.xscale('log')
    plt.xlabel(r'$\vert\omega\vert$')
    plt.title('Fourier Spectrum')
    plt.show()


    
def Show_langevin(x,x_reco,log=False):
  """Compares original DATA and synthesis data X_fake

  Parameters:
  DATA (tensor): (n_batch,L,L)
  X_fake (tensor): (n_batch,L,L)
  log (Bool): if True log scale

  """
  Compare_Spectrum(x,x_reco)

  plt.hist(x_reco.detach().cpu().reshape((-1,)),label='Langevin',density=True, cumulative=True,bins=50)
  plt.hist(x.detach().cpu().reshape((-1,)),alpha=0.5,label='True',density=True, cumulative=True,bins=50)
  plt.legend()
  plt.show()

  plt.hist(x_reco.detach().cpu().reshape((-1,)),label='Langevin',density=True, cumulative=False,bins=50)
  plt.hist(x.detach().cpu().reshape((-1,)),alpha=0.5,label='True',density=True, cumulative=False,bins=50)
  if log == True:
      plt.yscale('log')
  plt.legend()
  plt.show()

  print('mean_true = '+str(x.detach().cpu().reshape((-1,)).mean()))
  print('std_true = '+str(x.detach().cpu().reshape((-1,)).reshape((-1,)).std()))
  print('mean_langevin = '+str((x_reco).detach().cpu().reshape((-1,)).mean()))
  print('std_langevin = '+str((x_reco).detach().cpu().reshape((-1,)).std()))

  #Sample example
  plt.imshow(x[0].detach().cpu())
  plt.title('True')
  plt.show()
  plt.imshow(x_reco[0].detach().cpu())
  plt.title('Langevin')
  plt.show()
  

  #Covariances (Stationnary)
  cov_x = sfft.fftshift(torch.fft.ifft2(torch.fft.fft2(x).abs()**2).mean(0).detach().cpu().real)
  cov_langevin = sfft.fftshift(torch.fft.ifft2(torch.fft.fft2(x_reco).abs()**2).mean(0).detach().cpu().real)
  plt.imshow(cov_x)
  plt.title('True')
  plt.colorbar()
  plt.show()
  plt.imshow(cov_langevin)
  plt.title('Langevin')
  plt.colorbar()
  plt.show()


#Shows Spatial extent of Sigmoids 
def Plot_Sigmoid(centers,sigma):
    """Show Spatial extent of Sigmoids

    Parameters: 
    centers (tensor): position of the centers of the sigmoids, sorted in increasing order
    sigma (tensor) : width of the sigmoids
    
    
    """
    centers,sigma = centers.cpu(),sigma.cpu()
    window_min = centers[0]
    window_max = centers[-1]
    #POTENTIAL
    X=torch.linspace(window_min,window_max,1000)
    U=torch.sigmoid(-(X[None, :] - centers[:, None]) / (sigma[:,None] )) #(n_potentials,n_X)
    plt.plot(X,U.transpose(0,1))
    plt.title('Scalar Potentials')
    plt.ylabel(r'$\rho(t)$')
    plt.xlabel(r'$t$')
    plt.show()

#Show Sigmoid Learned Potentials
def Show_Sigmoid(ansatz_union,add_Trace=True,Free=False,index_scalar=0,index_quad=1) :
  """Shows the learned scalar Potential

  Parameters: 
  ansatz_union (Condi Ansatz): The ansatz 
  add_trace(Bool): if True, we add the trace of the quadratic form
  Free : Whether ansatz_union is a Free Energy (True) or not (False)
  index_scalar (int): position of the quad potential in ansatz_union.ansatze
  index_quad (int): position of the quad potential in ansatz_union.ansatze


  """
  
  Sc = ansatz_union.ansatze[index_scalar] 
  window_min =Sc.centers[0]
  window_max =Sc.centers[-1]
  #POTENTIAL
  X=torch.linspace(window_min,window_max,1000).cuda()
  U=torch.sigmoid(-(X[None, :] - Sc.centers[:, None]) / (Sc.sigma[:,None] )) # (M, D) to (M,)
  
  #Num_potentials
  n_pot = 0
  for i in range(0,index_scalar):
    n_pot+=ansatz_union.ansatze[i].num_potentials
  n_scalar = ansatz_union.ansatze[index_scalar].num_potentials
  #theta learned
  theta =ansatz_union.theta()[n_pot:n_pot+n_scalar]
  
  #Compute Trace
  if add_Trace == True:
    Sq = Square_laplacian(ansatz_union,Free=Free,index_quad=index_quad)
  else:
    Sq =0
  plt.plot(X.cpu().detach(),(theta@U+(Sq/2)*X**2).cpu().detach())
  plt.title('potential')
  plt.show()


def compare_hist(x_reco,phi_s,log=True,bins=200):
  plt.hist(phi_s.cpu().reshape(-1),density=True,bins=bins,label='Original')
  plt.hist(x_reco.cpu().reshape(-1),density=True,bins=bins,alpha=0.6,label='Synthesis')
  plt.legend()
  if log == True:
    plt.yscale('log')
  plt.show()


def show_kernel(ansatz,n_shifts_x,n_shifts_y):

  Gauss=ansatz.ansatze[1]
  n_offset = ansatz.num_potentials-Gauss.num_potentials
  theta = ansatz.theta()[n_offset:].cpu()

  Kernel = torch.zeros((4,4,2*n_shifts_x+1,2*n_shifts_y+1))
  shifts = Gauss.shifts #(n_shifts)
  Channel_1 = Gauss.first_channel_indices #(n_theta,)
  Channel_2 = Gauss.second_channel_indices#(n_theta,)
  pos_shifts = Gauss.pos_shift_indices#(n_theta,)

  for i in range(len(Channel_1)):
    c1,c2=Channel_1[i].numpy(),Channel_2[i].numpy()
    index = pos_shifts[i].numpy()
    s1,s2= int(shifts[index,0].numpy()+n_shifts_x),int(shifts[index,1].numpy()+n_shifts_y)
    Kernel[c1,c2,s1,s2] = theta[i]
  Kernel = Kernel.detach()
  index=[r'$x$',r'$\bar{x}^1$',r'$\bar{x}^2$',r'$\bar{x}^3$']
  for i in range(4):
    for j in range(i,4):
      if i ==0 and j==0:
        pass
      else:
        plt.imshow(Kernel[i,j])
        plt.title(r'$K($'+index[i]+r'$,$'+index[j]+r'$)$',fontsize=15)
        plt.colorbar()
        plt.show()
  
  return(Kernel)