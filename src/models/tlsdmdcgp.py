import torch
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp
import numpy as np
import time
import copy

from .exact_gp import DiscreteDynamicsIndependentGPs

class tlsDMDcGPModel(torch.nn.Module):
    """ Gaussian process + tlsDMDc reduced order dynamics """
    def __init__(self, nx, ny, nu=1, A=None, B=None, C=None, y_mean=None):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nu = nu

        self.A = A
        self.B = B
        self.C = C
        self.y_mean = torch.zeros(ny) if y_mean is None else y_mean

        if A is None:
            self.trained = False
        else:
            self.trained = True

    def train(self, training_dataset, training_iter=100, lr=0.1, path='data/gptlsdmdc_model.pth', 
              min_sigma=0.1, pod_modes=None, gp_data_skip=1, cuda=False):

        # ******** First, compute DMDc model ********
        start_time = time.time()

        assert self.ny == training_dataset.n_points * training_dataset.n_scalars

        print(f'Number of snapshots: {len(training_dataset)-1}')

        # Now fit linear dynamics
        Y0 = training_dataset.Y.T - self.y_mean.unsqueeze(-1)
        Y1 = training_dataset.Y_plus.T - self.y_mean.unsqueeze(-1)
        U0 = training_dataset.U.T

        if pod_modes is None:
            self.C = training_dataset.POD(y_mean=self.y_mean)[0][:,:self.nx]
        else:
            self.C = pod_modes[:,:self.nx]

        # Note: self.C.T is the pseudoinverse of self.C
        H0 = self.C.T @ Y0
        H1 = self.C.T @ Y1

        U, _, _ = torch.linalg.svd(torch.vstack((H0, U0, H1)), full_matrices=True)
        U11 = U[:self.nx+self.nu,:self.nx+self.nu]
        U21 = U[self.nx+self.nu:,:self.nx+self.nu]

        # Transition and control matrix
        AB = U21.matmul(torch.linalg.inv(U11))
        self.A = AB[:self.nx,:self.nx]
        self.B = AB[:self.nx,self.nx:]

        # Compute DMD training error
        if self.nu > 0:
            dmd_error = torch.linalg.norm((H1 - (self.A @ H0 + self.B @ U0)), ord='fro')/ \
                        torch.linalg.norm(H1, ord='fro') * 100
        else:
            dmd_error = torch.linalg.norm((H1 - self.A @ H0), ord='fro')/ \
                        torch.linalg.norm(H1, ord='fro') * 100

        mid_time = time.time()
        print(f'DMD model computed. Error: {dmd_error} %')
        print(f'Maximum eigenvalue magnitude: {torch.max(torch.abs(torch.linalg.eig(self.A)[0]))}')
        print(f'DMD training done. Time: {time.time() - start_time} s.')

        # ******** Now, compute GP error model ********
        train_x = H0
        train_u = U0
        train_xp = H1 - self.A @ H0 - self.B @ U0 # DMDc model error

        T = gp_data_skip
        self.gp_error_model = DiscreteDynamicsIndependentGPs(train_x[:,::T].T, train_u[:,::T].T, train_xp[:,::T].T, min_sigma=min_sigma)

        self.gp_error_model.train(training_iter=training_iter, lr=lr, verbose=True, model_path=path, cuda=cuda)

        # Compute total error
        print(f'GP training done. Time: {time.time() - mid_time} s.')
        print(f'Total training done. Time: {time.time() - start_time} s.')
        self.trained = True

    def forward(self, x, u, return_std=False):
        x_err, x_std = self.gp_error_model.predict(x, u)
        x_next = self.A @ x + self.B @ u + x_err
        y_next = self.C @ x_next + self.y_mean.flatten()
        if return_std:
            return x_next, y_next, x_std
        return x_next, y_next

    def project(self, y):
        assert y.shape[0] == self.y_mean.shape[0]
        if y.ndim == 1:
            return self.C.T @ (y - self.y_mean)
        return self.C.T @ (y - self.y_mean)
    
    def linearize(self, x, u):
        # linearize GP component
        A, B, d = self.gp_error_model.linearize(x, u)

        return self.A + A, self.B + B, d
