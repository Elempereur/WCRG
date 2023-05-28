import numpy as np
import torch
import matplotlib.pyplot as plt
from Wavelet_Packets import Tree_Depth

#Will load synthetised images of size L*L and upsample them to size 2L*2L
def load(L,KEY,W,direct,device='cuda'): 
  x_reco = torch.load(direct+KEY+str(L)+'_synth.pt').to(device)
  print(x_reco.shape)
  x_reco = W.Inv_2d(x_reco, torch.zeros_like(x_reco), torch.zeros_like(x_reco), torch.zeros_like(x_reco))
  print(x_reco.shape)
  return(x_reco)
 
#Save synthetised image 
def save(x_reco,L,KEY,direct):
  torch.save(x_reco,direct+KEY+str(L)+'_synth.pt')

#Save Model
def save_ansatz(ansatz,L,KEY,direct):  
    torch.save(ansatz,direct+KEY+str(L)+'_ansatz.pt')
#Load Model
def load_ansatz(L,KEY,direct):
    return torch.load(direct+KEY+str(L)+'_ansatz.pt')

#Will wavelet decompose depth time and return low freqs 
def load_data(W,Data,depth,J,show_hist=True,n_bins=1000):
  if depth == 0:
      
    phi_s = Data
  else :
    tree_util = Tree_Depth(depth)
    # DataShape
    L = 2**(J-depth)
    
    n_batch = len(Data)
    phi_s = torch.cat([W.Packet_2d(Data[n_bins *k:n_bins*(k+1)],tree_util)[:,:L,:L] for k in range(n_batch//n_bins+1)])#(N,L,L)

  if show_hist == True:
      plt.hist(phi_s.cpu().reshape((-1)),bins=100)
      plt.yscale('log')
      plt.show()
  return(phi_s)