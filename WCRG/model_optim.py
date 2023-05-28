"""Ansatz Optimisation"""


import numpy as np
import torch
import matplotlib.pyplot as plt


def Loss_score(x,ansatz_union):
    # x is the images (n,L,L)
    theta = ansatz_union.theta() #(n_potentials,)
    # Potential Gradients
    Grad_mean, Laplace_mean =  ansatz_union.compute_grad(x,theta) #(1,1) and (1,)
    #Compute Loss
    Quad = Grad_mean.sum()
    Linear =  Laplace_mean.sum()
    Loss=Quad-2*Linear
    
    return(Loss)
    
    

#Optim
def optim(ansatz_union,dataloader,num_epochs,lr,momentum=0,weight_decay=0):
    """Ansatz optimisation with SGD
    
    Parameters:
    ansatz_union (Condi_union) : ansatz to optimise
    dataloader (torch Dataloader):  with x_{j-1} maps for conditional models
    num_epochs (int) :number of steps
    lr (float) : learning rate
    momentum : momentum in SGD
    weight_decay : l2 renormalisation in SGD
    
    """

    optimizer = torch.optim.SGD([ansatz_union.theta_no_rescale], lr=lr, momentum=momentum, weight_decay=weight_decay)
    for _ in range(num_epochs):
      for x in dataloader:
          l=Loss_score(x,ansatz_union)
          optimizer.zero_grad()
          l.backward()
          optimizer.step()
          print( '[{}/{}] loss: {:.8}'.format(_, num_epochs, l.cpu().detach().numpy() ) )


def direct_estimate(dataloader,Sc,mean_mode=True):
    """Ansatz optimisation with direct matrix inversion
    
    Parameters:
    dataloader (torch Dataloader):  with x_{j-1} maps for conditional models
    Sc (Condi_union) : ansatz to optimise
    mean_mode (Bool) : if True quantities are averaged at each batch, meaning all batch should be of equal size
    
    """
    
    #E[grad U grad U^T] ~ (n_potentials,n_potentials) and E[ΔU] ~ (n_potentials,)
    Grad = []
    for x in dataloader :
      u,v = Sc.compute_grad(x,None,mean_mode=mean_mode)
      if mean_mode == True:
        u,v = u.cpu()[None],v.cpu()[None]
      else:
        u,v = u.cpu(),v.cpu()
      Grad.append((u,v))

    Grad_mean = torch.concat([elem[0] for elem in Grad],axis=0)
    Laplace_mean = torch.concat([elem[1] for elem in Grad],axis=0)

    Grad_mean=Grad_mean.mean(0).cuda()
    Laplace_mean=Laplace_mean.mean(0).cuda()
    #We will store theses matrices in case we need them
    Sc.Grad_mean = Grad_mean
    Sc.Laplace_mean = Laplace_mean
    #Conditionning of learning hessian
    H_nonorm = Grad_mean
    D = torch.diag(H_nonorm)  # (S,)
    H_norm = H_nonorm / torch.sqrt(D[None, :] * D[:, None])  # (S, S)
    Hs = torch.stack((H_nonorm, H_norm))  # (2, S, S)
    Hs = Hs / Hs.abs().amax((-1, -2), keepdim=True)
    eigvals, eigvectors = torch.linalg.eigh(Hs)  
    #Plot eigenvalues for learning hessian
    plt.plot(eigvals[0].cpu(),label='No_Normalisation')
    plt.plot(eigvals[1].cpu(),label='Diagonal_Renormalisation')
    plt.legend()
    plt.title('Hessian_EigenValues')
    plt.show()
    print('No_Normalisation_Conditioning = '+str(eigvals[0].cpu().max()/eigvals[0].cpu().min()))
    print('Diagonal_Renormalisation_Conditioning = '+str(eigvals[1].cpu().max()/eigvals[1].cpu().min()))

    
    #INVERTING
    D = torch.diag(Grad_mean) #(n_potential,)
    D_sqrtinv = torch.diag(1/D**0.5) #(n_potential,n_potential)
    
    A = (D_sqrtinv@Grad_mean)@D_sqrtinv #(n_potential,n_potential)
    b = D_sqrtinv@Laplace_mean #(n_potential,)
    
    
    
    #Store the diagonal rescaling applied to learning hessian and theta normalized by the rescaling
    #theta_no_rescale = torch.linalg.solve(A, b)
    theta_no_rescale = torch.linalg.inv(A)@b
    
    Sc.theta_no_rescale = torch.nn.Parameter(theta_no_rescale) #(n_potential,)
    Sc.rescale_theta = torch.diag(D_sqrtinv) #(n_potential,)
    
    #Compute Loss
    Quad = ((Sc.theta()[None]@Grad_mean)@Sc.theta()).sum()
    Linear =  (Laplace_mean*Sc.theta()).sum()
    Loss=Quad-2*Linear
    
    print('LOSS ='+str(Loss))
   

def Free_estimate(dataloader,Free_ansatz,Condi_ansatz):
    """Free energy optimisation with direct matrix inversion
    
    Parameters:
    dataloader (torch Dataloader):  with x_{j-1} maps for conditional models 
    Free_ansatz(Condi_union) : Free energy ansatz to optimise F(x_j)
    Condi_ansatz(Condi_union) : Conditional Energy E(\bar x_j\vert x_j)
    mean_mode (Bool) : if True quantities are averaged at each batch, meaning all batch should be of equal size
    
    """
    
    #compute argmin E_xbarx[|theta_tilde nabla x U_tilde - theta nabla_x U |**2]
    #theta U is Condi_ansazt
    #tilde is for free energy
    Grad_store = []
    theta = Condi_ansatz.theta()
    
    n_iter = 0

    Grad_ansatz = 0
    Grad_Free_ansatz = 0
    
    for x in dataloader :
      n_iter+=1
      #We compute nabla_x U and nabla x U_tilde

      condi_condi, condi_hf = Condi_ansatz.decompose(x) #(n,N,N), (n,n_condi,Width,Width), (n,n_pred,Width,Width)
      free_condi, free_hf = Free_ansatz.decompose(x) #useless and (n,N,N)
      
      #Potential

      #Grad_U
      Grad_Condi_sc,Grad_Condi_g = Condi_ansatz.gradient_condi(condi_hf,condi_condi,theta) # probably a tuple, (n,1,L//2,L//2) and (n, 1,1,L//2,L//2)
     
      Grad_Condi_g = Grad_Condi_g[:,0,0] #(n, L/2,L/2)
      Grad_Condi_sc = Grad_Condi_sc[:,0]#(n, L/2,L/2)

      Grad = Grad_Condi_g+Grad_Condi_sc #(n, L/2,L/2)
      Grad = Grad.reshape(Grad.shape[:1]+(-1,)) #(*,n_variables)

      #Grad U_tilde
      
      Grad_Free = Free_ansatz.gradient(free_hf,free_condi) #(n,n_free_potentials,1,N,N)

      Grad_Free = Grad_Free[:,:,0]

      Grad_Free =  Grad_Free.reshape(Grad_Free.shape[:2]+(-1,)) #(n,n_free_potentials,n_variables)

      Grad_Free_2 = Grad_Free @ Grad_Free.transpose(-1,-2) #(n,n_potential,n_potential) 

      #GradUGradUtilde
      Grad = (Grad_Free*Grad[:,None]).sum(-1) #(*,n_potentials)

      Grad_ansatz += Grad.mean(0)
      Grad_Free_ansatz += Grad_Free_2.mean(0)
      

      
    Grad=Grad_ansatz/n_iter
    Grad_Free_2= Grad_Free_ansatz/n_iter
    
    #We will store theses matrices in case we need them
    Free_ansatz.Grad = Grad
    Free_ansatz.Grad_Free_2 = Grad_Free_2
    #Conditionning of matrix to invert
    H_nonorm = Grad_Free_2
    D = torch.diag(H_nonorm)  # (S,)
    H_norm = H_nonorm / torch.sqrt(D[None, :] * D[:, None])  # (S, S)
    Hs = torch.stack((H_nonorm, H_norm))  # (2, S, S)
    Hs = Hs / Hs.abs().amax((-1, -2), keepdim=True)
    eigvals, eigvectors = torch.linalg.eigh(Hs)  
    #Plot eigenvalues
    plt.plot(eigvals[0].cpu(),label='No_Normalisation')
    plt.plot(eigvals[1].cpu(),label='Diagonal_Renormalisation')
    plt.legend()
    plt.title('Hessian_EigenValues')
    plt.show()
    print('No_Normalisation_Conditioning = '+str(eigvals[0].cpu().max()/eigvals[0].cpu().min()))
    print('Diagonal_Renormalisation_Conditioning = '+str(eigvals[1].cpu().max()/eigvals[1].cpu().min()))

    
    #INVERTING
    D = torch.diag(Grad_Free_2) #(n_potential,)
    D_sqrtinv = torch.diag(1/D**0.5) #(n_potential,n_potential)
    
    A = (D_sqrtinv@Grad_Free_2)@D_sqrtinv #(n_potential,n_potential)
    b = D_sqrtinv@Grad #(n_potential,)
    
    
    
    #Store the diagonal rescaling applied to learning hessian and theta normalized by the rescaling
    #theta_no_rescale = torch.linalg.solve(A, b)
    theta_no_rescale = torch.linalg.inv(A)@b
    
    Free_ansatz.theta_no_rescale = torch.nn.Parameter(theta_no_rescale) #(n_potential,)
    Free_ansatz.rescale_theta = torch.diag(D_sqrtinv) #(n_potential,)

def Diagonal_Renormalisation(dataloader,Sc,mean_mode=True):
    """Will rescale ansatz in such a way that the hessian of the score matching loss function has diagonal 1
    
    Parameters:
    dataloader (torch Dataloader):  with x_{j-1} maps for conditional models
    Sc (Condi_union) : ansatz to optimise
    mean_mode (Bool) : if True quantities are averaged at each batch, meaning all batch should be of equal size
    
    """
    
    #E[grad U grad U^T] ~ (n_potentials,n_potentials) and E[ΔU] ~ (n_potentials,)
    Grad = []
    for x in dataloader :
      u,v = Sc.compute_grad(x,None,mean_mode=mean_mode)
      if mean_mode == True:
        u,v = u.cpu()[None],v.cpu()[None]
      else:
        u,v = u.cpu(),v.cpu()
      Grad.append((u,v))

    Grad_mean = torch.concat([elem[0] for elem in Grad],axis=0)

    Grad_mean=Grad_mean.mean(0).cuda()
   
    #INVERTING
    D = torch.diag(Grad_mean) #(n_potential,)
    D_sqrtinv = torch.diag(1/D**0.5) #(n_potential,n_potential)
    #Store
    Sc.rescale_theta = torch.diag(D_sqrtinv) #(n_potential,)