import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Wavelet(nn.Module):
    def __init__(self, BMT, m1, m2,mode = 'Periodic'):
        
        """Create the Wavelet Object from a low pass filter
        
        Parameters:
        BMT (tensor): low pass filter h ~(1,1,m1+m2+1) (h(-m1),...,h(-1),h(0),h(1),....,h(m2))
        m1 (int): number of coefficients before h(0) in BMT
        m2 (int): number of coefficients after h(0) in BMT
        mode (string) : padding, "Symmetric" or "Periodic" border conditions
    
        """
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.register_buffer('dec_lo', self.bmt_h(BMT))
        self.register_buffer('dec_hi', self.bmt_g(BMT))
        self.register_buffer('rec_lo', self.bmt_h_tilde(BMT))
        self.register_buffer('rec_hi', self.bmt_g_tilde(BMT))
        #self.dec_lo = self.bmt_h(BMT)
        #self.dec_hi = self.bmt_g(BMT)
        #self.rec_lo = self.bmt_h_tilde(BMT)
        #self.rec_hi = self.bmt_g_tilde(BMT)
        
        if mode == 'Symmetric':
            self.pad_reconstruct = self.pad_reconstruct_symmetric
            self.pad = self.pad_symmetric
        elif mode == 'Periodic':
            self.pad_reconstruct = self.pad_reconstruct_periodic
            self.pad = self.pad_periodic
        
    #Construction of the high pass and reconstruction orthogonal filters from the low pass filter
    def bmt_h(self,BMT):
        return (torch.flip(BMT, (-1,)))

    def bmt_h_tilde(self,BMT):
        return (BMT)

    def bmt_g_tilde(self,BMT):
        minus = torch.tensor([(-1) ** (j) for j in range(self.m1 + self.m2 + 1)],device=BMT.device)
        return (self.bmt_h(BMT) * minus)

    def bmt_g(self,BMT):
        return (torch.flip(self.bmt_g_tilde(BMT), (-1,)))
        
   
    #1d padding
    def pad_symmetric(self, x, n1, n2):
        # Reflective padding
        # x  ~(batch,k,N)
        # x_pad ~(batch,k,n1+N+n2)
        # ...x_2 x_1｜x_0 x_1 ... x_N-1 ｜x_N-2 x_N-3 .... x_1 ｜ x_0 x_1
        #  n1                   N              n2
        N = x.shape[2]
        k = (n1 + N + n2) // (2 * N - 2)
        r = ((2 * N - 2) - n1) % (2 * N - 2)

        x_rep = torch.concat([x, torch.flip(x[:,:, 1:-1], (2,))], axis=2)  # x_0 x_1 ... x_N-1 ｜x_N-2 x_N-3 ....x_1
        x_pad = torch.tile(x_rep, (k + 2,))  # ~(...,2*(N-2)*(k+2))
        x_pad = x_pad[:,:, r:]
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(...,m1+N+m2)
        
    def pad_periodic(self, x, n1, n2):
        # Periodic padding
        # x  ~(batch,k,N)
        # x_pad ~(batch,k,n1+N+n2)
        # ...x_N-2 x_N-1｜x_0 x_1 ... x_N-1 ｜x_0 x_1 ... x_N-1 ｜ x_0 x_1 ...
        #  n1                   N              n2
        N = x.shape[2]
        k = (n1 + N + n2) // N 
        r = (N  - n1) % N

        x_pad = torch.tile(x, (k + 2,))  # ~(batch,k,N*(k+2))
        x_pad = x_pad[:,:, r:] 
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(...,m1+N+m2)
        
    def pad_reconstruct_periodic(self, x, n1, n2, mode):
        # x  ~(batch,k,N/2)
        # x_pad ~(batch,k,n1+N+n2)
        
        #mode is not used here

        # ...x_0｜x_0 x_2 ... x_N-2 ｜ x_N-4 .... x_2 x_0 ｜ x_0 x_2
        #  Insert 0 then ...x_2 0｜x_0 0 x_2 ... x_N-2 0 ｜x_N-2 0 x_N-4 .... x_2 0 ｜ x_0 0 x_2
        #                     n1            N                       n2

        N = x.shape[2] * 2
        k = (n1 + N + n2) // N 
        r = (N  - n1) % N 
        
        #we want to do the padding on the last axis
        dim_last=len(x.shape)-1
        x_rep = torch.repeat_interleave(x, 2, dim=dim_last) #x_0 x_0 x_2 x_2 ..........x_N-2 x_N-2
        x_rep[:,:, 1::2] = torch.zeros_like(x_rep[:,:, 1::2])  # x_0 0 x_2 ... x_N-2 0

        x_pad = torch.tile(x_rep, (k + 2,))  # ~(batch,k,N*(k+2))
        x_pad = x_pad[:,:, r:]
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(batch,k,m1+N+m2)

    def pad_reconstruct_symmetric(self, x, n1, n2, mode):
        # x  ~(batch,k,N/2)
        # x_pad ~(batch,k,n1+N+n2)

        """Central 0, Miror N/2-1  padding"""  # mode=0
        # ...x_2｜x_0 x_2 ... x_N-2 ｜x_N-2 x_N-4 .... x_2 ｜ x_0 x_2

        """Miror 0, Central N/2-1  padding"""  # mode=1
        # x_pad ~(batch,k,n1+N+n2)

        # ...x_0｜x_0 x_2 ... x_N-2 ｜ x_N-4 .... x_2 x_0 ｜ x_0 x_2
        #  Insert 0 then ...x_2 0｜x_0 0 x_2 ... x_N-2 0 ｜x_N-2 0 x_N-4 .... x_2 0 ｜ x_0 0 x_2
        #                     n1            N                       n2

        N = x.shape[2] * 2
        k = (n1 + N + n2) // (2 * N - 2)
        r = ((2 * N - 2) - n1) % (2 * N - 2)

        #we want to do the padding on the last axis
        dim_last=len(x.shape)-1
        if mode == 0:
            x_rep = torch.concat([x, torch.flip(x[:,:, 1:], (2,))], axis=dim_last)  # x_0 x_2 ... x_N-2 ｜x_N-2 x_N-3 ....x_2
            x_rep = torch.repeat_interleave(x_rep, 2, dim=dim_last)  # x_0 x_0 x_2 ... x_N-2 x_N-2｜x_N-2 x_N-2  ....x_2 x_2
        elif mode == 1:
            x_rep = torch.concat([x, torch.flip(x[:,:, :-1], (2,))], axis=dim_last)  # x_0 x_2 ... x_N-2 ｜x_N-4 x_N-3 ....x_0
            x_rep = torch.repeat_interleave(x_rep, 2, dim=dim_last)  # x_0 x_0 x_2 ... x_N-2 x_N-2｜x_N-2 x_N-2  ....x_0 x_0
        else:
            print('mode=0 or mode=1')
        x_rep[:,:, 1::2] = torch.zeros_like(x_rep[:,:, 1::2])  # x_0 0 x_2 ... x_N-2 0｜x_N-2 0  ....x_2 0

        x_pad = torch.tile(x_rep, (k + 2,))  # ~(...,2*(N-2)*(k+2))
        x_pad = x_pad[:,:, r:]
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(...,m1+N+m2)

    def dwt(self, x):
        #Direct 1d fast wavelet transform 
        # x  ~(batch,k,N) -> has to be of dim 3 for convolution to work
        # Please give N even
        k = x.shape[1]
        N = x.shape[2]
        n_batch = x.shape[0]
        m1, m2 = self.m1, self.m2
        # Low Freq
        x_pad = self.pad(x.reshape(n_batch*k,1,N), m1, m2)   #~(batch*k,1,N+m1+m2)
        a = torch.nn.functional.conv1d(x_pad, self.dec_lo, padding='valid', groups=1,stride=2)  # ~(...,1,N/2)
        # High Freq
        x_pad = self.pad(x.reshape(n_batch*k,1,N), m2 - 1, m1 + 1)
        d = torch.nn.functional.conv1d(x_pad, self.dec_hi, padding='valid', groups=1,stride=2)  # ~(...1,N/2)
     
        
        return (a.reshape(n_batch,k,N//2), d.reshape(n_batch,k,N//2))

    def idwt(self, a, d):
        #Direct 1d fast inverse wavelet transform
        # Takes (a,d) with a low freq and d high freq filters
        # a,d ~(batch,k,N/2)
        m1, m2 = self.m1, self.m2
        k = a.shape[1]
        M = a.shape[2]
        n_batch = a.shape[0]
        
        a, d = self.pad_reconstruct(a.reshape(n_batch*k,1,M), m2, m1, mode=0), self.pad_reconstruct(d.reshape(n_batch*k,1,M), m1 + 1, m2 - 1, mode=1) #(n_batch*k,1,2*M+m1+m2)
        
        x = torch.nn.functional.conv1d(a, self.rec_lo, padding='valid',
                                       groups=1) + torch.nn.functional.conv1d(d, self.rec_hi,
                                                                              padding='valid', groups=1)                                                    
        return (x.reshape(n_batch,k,M*2))

    def Wav_2d(self, x):
        """2d fast wavelet transform

        Parameters:
        x (tensor): ~(batch,N,N) -> higher dimension won't work unfortunately,Please Give N even
    
        Returns:
        tuple of tensors: ~(batch,N/2,N/2) Low frequencies,High horizontal frequencies,High vertical frequencies, High diagonal frequencies
    
        """
        cA, cD = self.dwt(x)  # ~(batch,N,N/2)
        cAA, cAD = self.dwt(cA.swapaxes(1, 2))  # ~(batch,N/2,N/2)
        cDA, cDD = self.dwt(cD.swapaxes(1, 2))  # ~(batch,N/2,N/2)
        return (cAA.swapaxes(1, 2), cAD.swapaxes(1, 2), cDA.swapaxes(1, 2), cDD.swapaxes(1, 2))

    def Wav_2d_sqformat(self, x):
        """2d fast wavelet transform

        Parameters:
        x (tensor): ~(batch,N,N) -> higher dimension won't work unfortunately,Please Give N even
    
        Returns:
        tensor : ~(batch,N,N) 2d fast wavelet transform
    
        """
        cAA, cAD, cDA, cDD = self.Wav_2d(x)
        return (torch.concat([torch.concat([cAA, cAD], axis=2),
                              torch.concat([cDA, cDD], axis=2)],
                             axis=1))

    def Inv_2d(self, cAA, cAD, cDA, cDD):
        """2d Inverse fast wavelet transform

        Parameters:
        cAA (tensor): ~(batch,N/2,N/2) Low frequencies
        cAD (tensor): ~(batch,N/2,N/2) High horizontal frequencies
        cDA (tensor): ~(batch,N/2,N/2) High vertical frequencies
        cDD (tensor): ~(batch,N/2,N/2) High diagonal frequencies
    
        Returns:
        tensor : ~(batch,N,N) Inverse Fast Waveket transform
    
        """
        # cA,cAD,cDA,cDD  ~(batch,N/2,N/2) -> higher dimension won't work unfortunately
        cA = self.idwt(cAA.swapaxes(1, 2), cAD.swapaxes(1, 2)).swapaxes(1, 2)  # ~(batch,N,N/2)
        cD = self.idwt(cDA.swapaxes(1, 2), cDD.swapaxes(1, 2)).swapaxes(1, 2)  # ~(batch,N,N/2)
        return (self.idwt(cA, cD))

    def Inv_2d_sqformat(self, x):
        """2d Inverse fast wavelet transform

        Parameters:
        x (tensor): ~(batch,N,N) -> higher dimension won't work unfortunately, Please give N even
    
        Returns:
        tensor : ~(batch,N,N) 2d Inverse fast wavelet transform"""
        
        M, N = x.shape[1], x.shape[2]
        cAA, cAD, cDA, cDD = x[:, :M // 2, :N // 2], x[:, :M // 2, N // 2:], x[:, M // 2:, :N // 2], x[:,
                                                                                                           M // 2:,
                                                                                                           N // 2:]
        return (self.Inv_2d(cAA, cAD, cDA, cDD))

    # Wavelet Packet Decomposition

    def Packet_2d(self, x, tree):
        """2d Waveket Packet Transform

        Parameters:
        x (tensor): ~(batch,N,N) -> higher dimension won't work unfortunately,Please Give N even
        tree (Tree object): Frequential decomposition to adopt
    
        Returns:
        tensor : ~(batch,N,N) Wavelet Packet Transform
    
        """
        return (self.Node_Packet_2d(x, tree.nodes))

    def Node_Packet_2d(self, x, node):
        # x  ~(batch,N,N) -> higher dimension won't work unfortunately
        #node is a Treenode object
        if node == None:
            return x
        else:
            if node.index_x % 2 == 0:
                if node.index_y % 2 == 0:
                    cAA, cAD, cDA, cDD = self.Wav_2d(x)
                else:
                    cAD, cAA, cDD, cDA = self.Wav_2d(x)
            else:
                if node.index_y % 2 == 0:
                    cDA, cDD, cAA, cAD = self.Wav_2d(x)
                else:
                    cDD, cDA, cAD, cAA = self.Wav_2d(x)

            return torch.concat(
                [torch.concat([self.Node_Packet_2d(cAA, node.aa), self.Node_Packet_2d(cAD, node.ad)], axis=2),
                 torch.concat([self.Node_Packet_2d(cDA, node.da), self.Node_Packet_2d(cDD, node.dd)], axis=2)],
                axis=1)


    def Inv_Packet_2d(self, x, tree):
        """2d Inverse Waveket Packet Transform

        Parameters:
        x (tensor): ~(batch,N,N) -> higher dimension won't work unfortunately,Please Give N even
        tree (Tree object): Frequential decomposition to adopt
    
        Returns:
        tensor : ~(batch,N,N) Inverse Wavelet Packet Transform
    
        """
        return (self.Node_Inv_Packet_2d(x, tree.nodes))

    def Node_Inv_Packet_2d(self, x, node):
        # x  ~(batch,N,N) -> higher dimension won't work unfortunately
        if node == None:
               
            return x
        else:
            
            N = x.shape[2]
            
            
                
            
            if node.index_x % 2 == 0:
                if node.index_y % 2 == 0:
                    return self.Inv_2d(
                        self.Node_Inv_Packet_2d(x[:, :N // 2, :N // 2], node.aa),
                        self.Node_Inv_Packet_2d(x[:, :N // 2, N // 2:], node.ad),
                        self.Node_Inv_Packet_2d(x[:, N // 2:, :N // 2], node.da),
                        self.Node_Inv_Packet_2d(x[:, N // 2:, N // 2:], node.dd))
                else:
                    return self.Inv_2d(
                        self.Node_Inv_Packet_2d(x[:, :N // 2, N // 2:], node.ad),
                        self.Node_Inv_Packet_2d(x[:, :N // 2, :N // 2], node.aa),
                        self.Node_Inv_Packet_2d(x[:, N // 2:, N // 2:], node.dd),
                        self.Node_Inv_Packet_2d(x[:, N // 2:, :N // 2], node.da))
            else:
                if node.index_y % 2 == 0:
                    return self.Inv_2d(
                        self.Node_Inv_Packet_2d(x[:, N // 2:, :N // 2], node.da),
                        self.Node_Inv_Packet_2d(x[:, N // 2:, N // 2:], node.dd),
                        self.Node_Inv_Packet_2d(x[:, :N // 2, :N // 2], node.aa),
                        self.Node_Inv_Packet_2d(x[:, :N // 2, N // 2:], node.ad))
                else:
                    return self.Inv_2d(
                        self.Node_Inv_Packet_2d(x[:, N // 2:, N // 2:], node.dd),
                        self.Node_Inv_Packet_2d(x[:, N // 2:, :N // 2], node.da),
                        self.Node_Inv_Packet_2d(x[:, :N // 2, N // 2:], node.ad),
                        self.Node_Inv_Packet_2d(x[:, :N // 2, :N // 2], node.aa))
                        
                        
                        



