"""Predifined possible centers/sigma possibilities for scalar potentials and shifts"""

import torch

def linspace_centers(window_min,window_max,num_potentials,extent =0.5*0.47,device='cuda'):
    """regularly spaced potentials for scalar potential
    
    Parameters:
    window_min (float): position of the lowest potential
    window_max (float): position of the highest potential
    num_potentials (int): number of potentials
    extent (float): spatial amplitude of the potentials (*spacing)
    device (torch.device) : 
    
    
    Returns:
        centers (tensor): position of the centers of the potentials
        sigma (tensor) : width of the potentials
    
    """
   
    centers = torch.linspace(window_min,window_max,num_potentials).to(device)
    sigma = torch.ones((num_potentials,),device=device)*(centers[1]-centers[0])*extent
    
    return (centers,sigma)
    
def quantile_centers(data,num_potentials,quantile_min=0,quantile_max=1,extent =0.65*0.5,device='cuda'):
    """Potentials localised at marginals quantiles for scalar potential
    
    Parameters:
    data (tensor): quantiles are taken according to data marginal
    num_potentials (int): number of potentials
    quantile_min (float): smallest quantile to take into account
    quantile_max (float): largest quantile to take into account
    extent (float): spatial amplitude of the potentials (*spacing)
    device (torch.device) : 
    
    
    Returns:
        centers (tensor): position of the centers of the potentials 
        sigma (tensor) : width of the potentials
    
    """
   
    centers = torch.quantile(data,torch.linspace(quantile_min,quantile_max,num_potentials).to(device))
    sigma =  torch.cat([(centers[1:-1]-centers[:-2])*extent,extent*(centers[-2]-centers[-3])[None],extent*(centers[-1]-centers[-2])[None]])
    
    return (centers,sigma)

def multi_quantile_centers(data,num_potentials,quantiles,extent =0.65*0.5,device='cuda'):
    """Potentials localised at marginals quantiles for scalar potential
    
    Parameters:
    data (tensor): quantiles are taken according to data marginal
    num_potentials (list of int): number of potentials for each quantile window
    quantiles (list of float): quantile windows are [quantiles[i],quantiles[i+1]]
    extent (float): spatial amplitude of the potentials (*spacing)
    device (torch.device) : 
    
    
    Returns:
        centers (tensor): position of the centers of the potentials 
        sigma (tensor) : width of the potentials
    
    """
    centers = torch.cat([torch.quantile(data,torch.linspace(quantiles[i],quantiles[i+1],num_potentials[i]).to(device))[::-2] for i in range(len(num_potentials)-1)]+[torch.quantile(data,torch.linspace(quantiles[-2],quantiles[-1],num_potentials[-1]).to(device))])
   
    centers = torch.quantile(data,torch.linspace(quantile_min,quantile_max,num_potentials).to(device))
    sigma =  torch.cat([(centers[1:-1]-centers[:-2])*extent,extent*(centers[-2]-centers[-3])[None],extent*(centers[-1]-centers[-2])[None]])
    
    return (centers,sigma)

def shifts_quad(n_x,n_y):
  """Positive shifts for quadratic potential, for a centered kernel of size (2*n_x+1,2*n_y+1)
    
  Parameters:
    n_x (int) : kernel extent on x axis
    n_y (int): kernel extent on y axis
    
  Returns:
        shifts (list of tuples):
    
  """
  shifts = []
  for i in range(n_x+1):
    for j in range(n_y+1):
      if i ==0:
        if j == 0:
          pass
        else:
          shifts.append((i,j))
      else: 
        if j ==0:
          shifts.append((i,j))
        else:
          shifts.append((i,j))
          shifts.append((i,-j))
  return(shifts)

def shift_quad_Sym(n_x,n_y):
  """Positive and negative shifts for quadratic potential, for a centered kernel of size (2*n_x+1,2*n_y+1)
    
  Parameters:
    n_x (int) : kernel extent on x axis
    n_y (int): kernel extent on y axis
    
  Returns:
        shifts (list of tuples) :
    
  """
  shifts = []
  for i in range(-n_x+1,n_x+1):
    for j in range(-n_y+1,n_y+1):
      if i==0 and j==0 :
          pass
      else:
          shifts.append((i,j))
  return(shifts)
    
