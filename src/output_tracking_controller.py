import numpy as np
import scipy as sp
import scipy.linalg
from scipy import sparse
import quadprog
import time
import torch

class TorchOutputTrackingController:
    def __init__(self, A, B, C, d=None, N = 1, u_min = 0., u_max = 1., Q = 1., R = 1.):

        """
        Output tracking controller of a discrete-time linear system
        Inputs:
            - N: Control horizon
            - u_min: Minimum input
            - u_max: Maximum input
            - q: Tracking cost
            - R: Actuation cost
        """

        nx = A.shape[0] # Number of reduced-order states
        nu = B.shape[1] # Number of inputs
        ny = C.shape[0] # Number of outputs
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.Q = torch.tensor(Q)
        self.R = torch.tensor(R)
        self.A = A
        self.B = B
        self.d = torch.zeros(nx) if d is None else d # Drift term

        assert len(self.Q) == nx
        assert len(self.R) == nu

        # Compute Gamma
        Gamma = torch.zeros((ny*N,nu*N))
        for i in range(N):
            for j in range(i+1):
                Gamma[ny*i:ny*(i+1),nu*j:nu*(j+1)] = ((C @ torch.linalg.matrix_power(A, i-j)) @ B)
        
        # Compute Delta
        Delta = torch.zeros((ny*N,nx*N))
        for i in range(N):
            for j in range(i+1):
                Delta[ny*i:ny*(i+1),nx*j:nx*(j+1)] = ((C @ torch.linalg.matrix_power(A, i-j)))
        
        dd = torch.tile(self.d, (N,))


        # Compute RR
        RR = torch.sparse.spdiags(torch.tile(self.R, (N,)), torch.tensor([0]), (N*nu,N*nu)) # EXPENSIVE

        # Compute Omega
        Omega = torch.zeros((ny*N,nx))
        for i in range(N):
            Omega[ny*i:ny*(i+1),:] = C @ torch.linalg.matrix_power(A, i+1)

        # Compute QQ, H
        QQ = torch.sparse.spdiags(torch.tile(self.Q, (N,)), torch.tensor([0]), (N*nx,N*nx)) # EXPENSIVE

        QQGamma = QQ @ Gamma

        H = Gamma.T @ QQGamma + RR
        QQOmega = QQ @ Omega
        OmegaQQGamma = QQOmega.T @ Gamma

        # Compute constraint matrices
        L = torch.vstack((torch.eye(N*nu), -torch.eye(N*nu)))
        uu_max = u_max*torch.ones((N,2))
        uu_min = -u_min*torch.ones((N,2))
        W = torch.vstack((uu_max, uu_min)).flatten()

        # Save QP matrices
        self.H = H
        self.L = -L.T
        self.W = -W
        self.QQ = QQ
        self.Gamma = Gamma
        self.DeltadQQGamma = (Delta @ dd).T @ (QQ @ Gamma)
        self.OmegaQQGamma = OmegaQQGamma
        self.uu_max = uu_max
        self.uu_min = uu_min

    def compute_input(self, y_des, x0):
        f = y_des.flatten() @ (self.QQ @ self.Gamma) - x0 @ self.OmegaQQGamma - self.DeltadQQGamma
        return quadprog.solve_qp(
            self.H.numpy().astype(np.float64), 
            f.numpy().astype(np.float64), 
            self.L.numpy().astype(np.float64), 
            self.W.numpy().astype(np.float64), 
            0
        )[0]
