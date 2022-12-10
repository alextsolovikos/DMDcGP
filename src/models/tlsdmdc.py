import torch
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp
import numpy as np
import time
import copy

class tlsDMDcModel(torch.nn.Module):
    """
    PyTorch class for a DMDc model
    """
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

    def train(self, training_dataset, pod_modes=None):
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

        print(f'DMD model computed. Error: {dmd_error} %')
        print(f'Maximum eigenvalue magnitude: {torch.max(torch.abs(torch.linalg.eig(self.A)[0]))}')
        print(f'DMD training done. Time: {time.time() - start_time} s.')

        self.trained = True

    def test(self, test_dataset):
        """
        Not implemented yet
        """
        raise ValueError('Not implemented yet.')

    def forward(self, x, u):
        if self.nu > 0:
            x_next = self.A @ x + self.B @ u
            y_next = self.C @ x_next + self.y_mean.flatten()
            return x_next, y_next
        else:
            x_next = self.A @ x
            y_next = self.C @ x_next + self.y_mean.flatten()
            return x_next, y_next

    def project(self, y, orthogonal=True):
        assert y.shape[0] == self.y_mean.shape[0]
        if y.ndim == 1:
            if orthogonal:
                return self.C.T @ (y - self.y_mean)
            else:
                return torch.linalg.pinv(self.C) @ (y - self.y_mean)

        if orthogonal:
            return self.C.T @ (y - self.y_mean)
        else:
            return torch.linalg.pinv(self.C) @ (y - self.y_mean)

    def plot_eigenvalues(self, ax=None, color=None):
        eigs = torch.linalg.eig(self.A)[0]

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')

        # Unit circle
        circle = plt.Circle((0,0), 1.0, edgecolor='k', facecolor='none', linewidth=1)
        ax.add_artist(circle)

        # Eigenvalues
        if color is None:
            color = 'r'

        ax.scatter(torch.real(eigs), torch.imag(eigs), s=20, marker='o', facecolor='none', edgecolor=color, zorder=15, clip_on=False)

        ax.set_axisbelow(True)
        plt.grid(True)
