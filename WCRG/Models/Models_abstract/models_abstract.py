"""The Ansatz are defined here"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod
from functorch import vmap, jacrev,jacfwd, hessian,jvp
from .utils import partial_reshape

#Abstract Conditional Potential
class Condi_Ansatz(torch.nn.Module, ABC):
    """ Class which provides defaults for gradients, Hessians and Laplacians. """
    def __init__(self, num_potentials):
        super().__init__()
        self.num_potentials = num_potentials
    
    def potential(self, x,x_condi):
        """ (D,) to (M,) where M is the number of potentials. """
        raise NotImplementedError
        
    def get_energy(self, theta=None):
        """ Returns a function that maps (D,) to (M,) or (), depending on whether theta was given. """
        def f(x,x_condi):
            p = self.potential(x,x_condi)  # (M,)
            if theta is not None:
                p = (p @ theta )[None]  # (1,) This way it will work with the other functions
            return p
        return f
    
    def compute(self, transform, x,x_condi, theta=None):
        """ Computes transform(energy)(x) batched over all batch axes. (*, in_s) to (*, out_s). """
   
        """vmap won't work unfortunately"""
        #y = vmap(transform(self.get_energy(theta)))(x,x_condi)  # (B, out
        """I will rather loop even tho it's longer"""
        # Changed to a single concat call (linear complexity instead of quadratic)
        #y = torch.concat([transform(self.get_energy(theta))(x[i], x_condi[i])[None] for i in range(x.shape[0])], axis=0)
        y = vmap(transform(self.get_energy(theta)))(x,x_condi)
        #jacrev will compute gradient according to the first argument, which is what we expect
    
        return y 
    
    def energy(self, x,x_condi, theta=None):
        """ (*, D) to (*, [M]). """
        return self.compute(lambda f: f, x,x_condi, theta)
    
    def gradient(self, x,x_condi, theta=None):
        """ (*,n_1,...n_k) to (*,n_potentials, n_1,....n_k). """
        #jacrev will compute gradient according to the first argument, which is what we expect
        return self.compute(jacrev, x,x_condi, theta)
    
    def gradient_condi(self,x,x_condi,theta = None):
        return self.compute(lambda f: jacrev(f,argnums=1), x,x_condi, theta)
    
    def hessian(self, x,x_condi, theta=None):
        hess=torch.concat([self.hessian_nobatch(x[i], x_condi[i])[None] for i in range(x.shape[0])], axis=0) #(n_variables,n_potentials,n_tot,n_tot)
        return hess
    
    def hessian_nobatch(self,x,x_condi,theta=None):
        #x_condi is given with no batch here!
        #x (n_1,n_2,...n_k)
        k=len(x.shape)
        hess=hessian(self.get_energy(theta))(x, x_condi)
        #hess=jacfwd(jacrev(self.get_energy(None)))(x, x_condi) #(n_potentials,n_1,...n_k,n_1,...,n_k)
        hess=partial_reshape(hess, (-1,), start=1, stop=k+1) #(n_potentials,n_tot,n_1,...,n_k)
        hess=partial_reshape(hess, (-1,), start=2, stop=k+2) #(n_potentials,n_tot,n_tot)

        return(hess)
      
    def laplacian_nobatch(self,x,x_condi,theta=None):
        #x,x_condi is given with no batch here!
        #x (n_1,n_2,...n_k)
        hess=self.hessian_nobatch(x,x_condi,theta) #(n_potentials,n_tot,n_tot)
        laplacian=torch.diagonal(hess, offset=0, dim1=-2, dim2=-1)  #(n_potentials,n_tot)
        laplacian=laplacian.sum(axis=-1) #(n_potentials,)
        return(laplacian)

    
    def laplacian(self, x,x_condi,theta=None):
        #x (batch,n_1,n_2,...n_k)
   
        #y=vmap(self.laplacian_nobatch)(x,x_condi) #(n_batch,n_potentials)
        #y = torch.concat([self.laplacian_nobatch(x[i], x_condi[i])[None] for i in range(x.shape[0])], axis=0) #(n_batch,n_potentials)
        y = vmap(lambda x,y : self.laplacian_nobatch(x,y,theta))(x,x_condi)

        return y
        
    def laplacian_Hutchinson_nobatch(self,x,x_condi,z,theta=None):
        #x,x_condi is given with no batch here!
        #x (n_1,n_2,...n_k)
        #z (n_1,n_2,...n_k)
        def Grad(x):
            Grad_U = self.gradient(x[None],x_condi[None],theta)[0] # (n_potentials,n_pred,Width,Width)
            Grad_U = Grad_U.reshape(Grad_U.shape[:1]+(-1,)) # (n_potentials,n_variables)
            return(Grad_U)
        
        LaplaZ = jvp(Grad,(x,),(z,))[1] # (n_potentials,n_variables)
        LaplaZ = LaplaZ @ z.reshape((-1,)) # (n_potentials,)
        
        return(LaplaZ)
    
    def laplacian_Hutchinson(self,x,x_condi,z,theta=None):
        #x (n,n_1,n_2,...n_k)
        #z (n,n_1,n_2,...n_k)
        #y = torch.concat([self.laplacian_Hutchinson_nobatch(x[i], x_condi[i],z[i],theta)[None] for i in range(x.shape[0])], axis=0) #(n_batch,n_potentials)
        
        y = vmap(lambda x,y,z : self.laplacian_Hutchinson_nobatch(x,y,z,theta))(x,x_condi,z)
        return y
    
        
#Union Potential
class Condi_Union(Condi_Ansatz):
    """ Union of different Ansätze. """
    
    
    def __init__(self, ansatze,condi_index,decompose,device = 'cuda',reconstruct = None):
        
        self.slices = []  # List of (i, j) pairs for indexing parameters. +(,k) which is the x_condi[k] to use (not all ansatz work the same unfortunately)
        i = 0
        index = 0
        for a in ansatze:
            j = i + a.num_potentials
            k=condi_index[index]
            self.slices.append((i, j,k))
            i = j
            index+=1
        super().__init__(num_potentials=i)
        self.ansatze = ansatze
        #Used to compute x,x_condi
        self.decompose=decompose #decompose x into conditional variable and variable
        self.reconstruct=reconstruct #reconstruct x from conditional variable and variable
        
        #parameters
        self.theta_no_rescale= torch.nn.Parameter(torch.zeros((self.num_potentials,)).to(device), requires_grad=True)
        self.rescale_theta = 1 #to set later with the hessian 
    
    
    #Learned weights for potential
    def theta(self):
        return self.theta_no_rescale * self.rescale_theta
    
    
    def Sampling_conditioning(self,x):
        #Compute Theta E[(d^2_ii)U] ~ (n_variables,n_variables) 
        #x is the images (n,L,L)
        
        wav_condi, wav_hf = self.decompose(x) #(n,N,N), (n,n_condi,Width,Width), (n,n_pred,Width,Width)
        
        #Potential
        #Hessian
        hess = self.hessian(wav_hf,wav_condi) #(n,n_potentials,n_variables,n_variables)
        hess = hess.transpose(1,3) #(n,n_variables,n_variables,n_potentials) #matrix is symmetric
        hess_theta = hess@self.theta() #(n,n_variables,n_variables)
        
        #hess_theta = self.hessian(wav_hf,wav_condi,self.theta()) #(n,1,n_variables,n_variables)
        #hess_theta = hess_theta[:,0] #(n,n_variables,n_variables)
        #Check if hess is well conditionned 
        eigvals, eigvectors = torch.linalg.eigh(hess_theta) #(n,n_variables) 
        eigvals=eigvals.detach().cpu()
        plt.plot(torch.transpose(eigvals,0,1).cpu())
        plt.title('Sampling_EigenValues')
        plt.show()
        print('Min_Eigen = '+str(eigvals.min()))
        print('Max_Eigen = '+str(eigvals.max()))
        print('Conditioning = '+str(eigvals.max()/eigvals.min()))
        
        #For Langevin
        return(eigvals.max()/eigvals.min(),1/eigvals.max())
        
    
    def Learning_conditioning(self,x):
        #x (n,L,L)
        wav_condi,wav_hf = self.decompose(x) 
        Grad_U = self.gradient(wav_hf,wav_condi) # (n, n_potentials,n_pred,Width,Width)
        Grad_U = Grad_U.reshape(Grad_U.shape[:2]+(-1,))
        H_nonorm = (Grad_U@ Grad_U.permute(0, 2, 1).conj()).mean(0)
        D = torch.diag(H_nonorm)  # (S,)
        H_norm = H_nonorm / torch.sqrt(D[None, :] * D[:, None])  # (S, S)
        Hs = torch.stack((H_nonorm, H_norm))  # (2, S, S)
        Hs = Hs / Hs.abs().amax((-1, -2), keepdim=True)
        eigvals, eigvectors = torch.linalg.eigh(Hs)  
        #Plot
        plt.plot(eigvals[0].cpu(),label='No_Normalisation')
        plt.plot(eigvals[1].cpu(),label='Diagonal_Renormalisation')
        plt.legend()
        plt.title('Hessian_EigenValues')
        plt.show()
        print('No_Normalisation_Conditioning = '+str(eigvals[0].cpu().max()/eigvals[0].cpu().min()))
        print('Diagonal_Renormalisation_Conditioning = '+str(eigvals[1].cpu().max()/eigvals[1].cpu().min()))
        
        #Return Diagonal_Renormalisation Optimal Learning Rate
        return(eigvals[1].cpu().min()/eigvals[1].cpu().max())
        
    def reduce(self, f, x, x_condi, theta=None):
        """ Applies the function over all Ansätze and reduces by concatenation/summation. (*, D) to (*, [M,], out). """
        ys = tuple(f(ansatz)(x,x_condi[k], None if theta is None else theta[i:j])
                   for ansatz, (i, j, k) in zip(self.ansatze, self.slices))  # tuple of (*, [M_i,], out)
        
        y = torch.cat(ys, dim=1)  # (*, M, out) #if Theta is not none it's not M but n_ansatz
        if theta is not None:
            y = y.sum(1)
            y=y[:,None] # (*, 1, out)
        
        return y
        
    def potential_batch(self,x,x_condi,theta):
        ys = tuple(ansatz.potential_batch(x,x_condi[k])
                   for ansatz, (i, j, k) in zip(self.ansatze, self.slices)) #(*,M)
        y = torch.cat(ys, dim=1)
        if theta is not None : 
            y = (y*theta).sum(1)[:,None] #(*,1)
        return(y)

    # Need to override all functions by adding sum or concatenation over all ansatzes to ensure optimization...

    
    def energy(self, x,x_condi, theta=None):
        return self.reduce(lambda a: a.energy, x,x_condi, theta)
    
    def gradient(self, x,x_condi, theta=None):
        return self.reduce(lambda a: a.gradient, x,x_condi, theta)
    
    def gradient_condi(self,x,x_condi,theta =None):
        #Compute gradient according to x_condi
        def reduce_condi(f, x, x_condi, theta=None):
            """ Applies the function over all Ansätze and reduces by concatenation/summation. (*, D) to (*, [M,], out). """
            ys = tuple(f(ansatz)(x,x_condi[k], None if theta is None else theta[i:j])
                    for ansatz, (i, j, k) in zip(self.ansatze, self.slices))  # tuple of (*, [M_i,], out):
            return(ys)
        return(reduce_condi(lambda a: a.gradient_condi, x,x_condi, theta))
        
        
    
    def hessian(self, x,x_condi, theta=None):
        return self.reduce(lambda a: a.hessian, x,x_condi, theta)
    
    def laplacian(self, x,x_condi, theta=None):
        return self.reduce(lambda a: a.laplacian, x,x_condi, theta)
    
    def laplacian_Hutchinson(self, x,x_condi,z, theta=None):
        
        ys = tuple((lambda a: a.laplacian_Hutchinson)(ansatz)(x,x_condi[k],z, None if theta is None else theta[i:j])
                   for ansatz, (i, j, k) in zip(self.ansatze, self.slices))  # tuple of (*, [M_i,], out)

        y = torch.cat(ys, dim=1)  # (*, M, out) #if Theta is not none it's not M but 1 
        if theta is not None:
            y = y.sum(1)
            y=y[:,None] # (*, 1, out)
        
        return y
        
    
    def compute_grad(self,x,theta,mean_mode = True):
        #Compute E[grad U grad U^T] ~ (n_potentials,n_potentials) and E[ΔU] ~ (n_potentials,)
        #x is the images (n,L,L)
        
        wav_condi, wav_hf = self.decompose(x) #(n,N,N), (n,n_condi,Width,Width), (n,n_pred,Width,Width)
        
        #Potential
        #Grad
        Grad_U = self.gradient(wav_hf,wav_condi,theta) # (n, n_potentials,n_pred,Width,Width)
        Grad_U = Grad_U.reshape(Grad_U.shape[:2]+(-1,)) # (n, n_potentials,variables)
        
        #Laplacian
        z=torch.randn(wav_hf.shape,device=wav_hf.device)
        Laplace_U=self.laplacian_Hutchinson(wav_hf,wav_condi,z,theta) # (n, n_potentials)
        
        #Laplace_U = self.laplacian(wav_hf,wav_condi,theta) # (n, n_potentials)
        
        #Compute E[Grad_U Grad_U^T]
        Grad_U = Grad_U @ Grad_U.transpose(-1,-2) #(n,n_potential,n_potential)
        
        if mean_mode == True : 
            Grad_U = Grad_U.mean(0) #(n_potential,n_potential)
            #Compute E[ΔU]
            Laplace_U = Laplace_U.mean(0) #(n_potentials,)
        
        return(Grad_U,Laplace_U)
        
    def compute_rescale(self,x):
        #Compute Diag(E[grad U grad U^T]) ~ (n_potentials,) 
        #x is the images (n,L,L)
        wav_condi, wav_hf = self.decompose(x) #(n,N,N), (n,n_condi,Width,Width), (n,n_pred,Width,Width)
        
       
        #Grad
        Grad_U = self.gradient(wav_hf,wav_condi) # (n, n_potentials,n_pred,Width,Width)
        Grad_U = Grad_U.reshape(Grad_U.shape[:2]+(-1,)) # (n, n_potentials,variables)
        
        #Compute E[Grad_U Grad_U^T]
        Grad_U = Grad_U @ Grad_U.transpose(-1,-2) #(n,n_potential,n_potential)
        Grad_U = Grad_U.mean(0) #(n_potential,n_potential)
        #Diag
        Grad_Diag=torch.diagonal(Grad_U)
        
        self.rescale_theta = 1/torch.sqrt(Grad_Diag)
      

        

