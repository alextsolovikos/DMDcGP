import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
# import h5py

class StateND:
    """
    Dataset for holding N-D state vector time series.
    """
    def __init__(self, n_states=1, n_inputs=1, info=None, cuda=False):

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_points = 1
        self.n_scalars = n_states
        self.is_empty = True
        self.info = info

        # Placeholders for data
        self.Y = torch.zeros((0,n_states))
        self.Y_plus = torch.zeros((0,n_states))
        self.U = torch.zeros((0,n_inputs))

        if cuda:
            self.Y = self.Y.cuda()
            self.Y_plus = self.Y_plus.cuda()
            self.U = self.U.cuda()

    def append(self, Y, Y_plus, U=None):

        assert Y.shape[1] == self.n_states
        assert Y_plus.shape[1] == self.n_states
        if (self.n_inputs > 0):
            assert U is not None
            assert U.shape[1] == self.n_inputs

        # Append data 
        self.Y = torch.vstack((self.Y, Y)) # Time step t, dimensions: (BATCH_SIZE, ny)
        self.Y_plus = torch.vstack((self.Y_plus, Y_plus)) # Time step t+1, dimensions: (BATCH_SIZE, ny)
        if self.n_inputs > 0:
            self.U = torch.vstack((self.U, U)) # Time step t, dimensions: (BATCH_SIZE, ny)

        self.is_empty = False

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, j):
        if torch.is_tensor(j):
            j = j.tolist()
        return self.Y[j], self.Y_plus[j], self.U[j]

    def POD(self, y_mean=None):
        """ Do not run POD - just return unit vector basis """
        return torch.eye(self.n_states), torch.eye(self.n_states), torch.eye(self.n_states)


class Field3D(Dataset):
    """
    Dataset for holding a 3D flowfield on an orthogonal grid.
    Each dataset is initialized with a given grid size. The actual grid and the snapshots
    are then read from a Nek5000 history point output file.
    """

    def __init__(self, grid_size=[16,16,16], scalars={'ux': 1, 'uy': 2, 'uz': 3}, n_inputs=0, info=None):

        assert len(grid_size) == 3
        self.grid_size = grid_size
        self.nx, self.ny, self.nz = grid_size
        self.n_points = self.nx * self.ny * self.nz

        self.scalars = scalars
        self.n_scalars = len(scalars)
        self.n_inputs = n_inputs

        self.n_steps = 0
        self.is_empty = True

        self.info = info

        # Placeholders
        self.coordsX = torch.zeros((self.grid_size)) # x-coordinate of 3D grid
        self.coordsY = torch.zeros((self.grid_size)) # x-coordinate of 3D grid
        self.coordsZ = torch.zeros((self.grid_size)) # x-coordinate of 3D grid
        self.Y_raw = torch.zeros((0,*self.grid_size,self.n_scalars)) # Hold all scalar fields here
        self.U = torch.zeros((0,self.n_inputs)) # Control inputs

    def append_from_hpts(self, snapshots_path, inputs_path=None, n_steps=None):

        # Load grid
        n_points = int(np.loadtxt(snapshots_path, dtype=int, max_rows=1))
        assert n_points == self.n_points
        x, y, z = pd.read_csv(snapshots_path, delimiter=' ', dtype='float64', header=None, skipinitialspace=True, skiprows=1, nrows=n_points).values.T
        self.coordsX, self.coordsY, self.coordsZ = (torch.from_numpy(f.reshape(*self.grid_size)) for f in [x, y, z])

        # Load inputs
        if self.n_inputs > 0:
            self.U = torch.from_numpy(np.expand_dims(np.loadtxt(inputs_path, dtype=np.float64, skiprows=1), -1)[9::10])

        # Load flowfield
        flowfield = pd.read_csv(
            snapshots_path, delimiter=' ', dtype='float64', header=None,
            skipinitialspace=True, skiprows=n_points+1,
            nrows=None if n_steps is None else n_steps*n_points
        ).values
        flowfield = flowfield.reshape(n_steps, *self.grid_size, -1)

        scalar_columns = [v for k, v in self.scalars.items()]
        self.Y_raw = torch.from_numpy(flowfield[...,scalar_columns])

        self.Y = self.Y_raw[:-1].flatten(start_dim=1)
        self.Y_plus = self.Y_raw[1:].flatten(start_dim=1)

        self.n_steps = n_steps-1
        self.is_empty = False

    def append(self, dataset2):
        assert torch.allclose(self.X, dataset2.X)
        assert torch.allclose(self.Y, dataset2.Y)
        assert torch.allclose(self.Z, dataset2.Z)
        assert self.n_scalars == dataset2.n_scalars

        self.U = torch.vstack((self.U, dataset2.U))
        self.inputs = torch.vstack((self.inputs, dataset2.inputs))
        self.n_steps = self.U.shape[0]
        self.info = self.info + ' | ' + dataset2.info

    def time_and_z_average(self):
        assert not self.is_empty

        self.U = torch.tile(torch.mean(self.U, axis=(0,3), keepdims=True), (1,1,1,self.nz,1))

        print('Time and z-averaging done!')
        
    def append(self, dataset2):
        assert torch.allclose(self.coordsX, dataset2.coordsX)
        assert torch.allclose(self.coordsY, dataset2.coordsY)
        assert torch.allclose(self.coordsZ, dataset2.coordsZ)
        assert self.n_scalars == dataset2.n_scalars

        self.Y_raw = torch.vstack((self.Y_raw, dataset2.Y_raw))
        self.Y = torch.vstack((self.Y, dataset2.Y))
        self.Y_plus = torch.vstack((self.Y_plus, dataset2.Y_plus))
        self.U = torch.vstack((self.U, dataset2.U))
        self.n_steps = self.Y_plus.shape[0]
        self.info = self.info + ' | ' + dataset2.info  

    def ensemble_average(self, ensemble_len, ensemble_period, n_ensembles=1, symmetric=False):
        assert not self.is_empty

        ensemble_ids = torch.vstack((
            [torch.arange(ensemble_len) + k * ensemble_period for k in range(n_ensembles)]
        ))

        self.Y_raw = self.Y_raw[ensemble_ids]
        self.Y_raw = torch.mean(self.Y_raw, axis=0)

        if symmetric:
            self.Y_raw = (self.Y_raw + torch.flip(self.Y_raw, [3])) / 2.

        if self.n_inputs > 0:
            self.U = self.U[ensemble_ids]
            self.U = torch.mean(self.U, axis=0)

        self.n_steps = ensemble_len - 1

        self.Y = self.Y_raw[:-1].flatten(start_dim=1)
        self.Y_plus = self.Y_raw[1:].flatten(start_dim=1)

        print('Ensemble averaging done!')

    def __len__(self):
        return self.n_steps - 1

    def __getitem__(self, j):
        if torch.is_tensor(j):
            j = j.tolist()

        return self.U[j], self.U[j+1], self.inputs[j]

    def x(self):
        return self.X.flatten()

    def y(self):
        return self.Y.flatten()

    def z(self):
        return self.Z.flatten()

    def mean(self):
        return np.mean(self.snapshots(), axis=-1)

    def tmp_fix(self):
        self.n_states = self.n_scalars
        self.coordsX = self.X
        self.coordsY = self.Y
        self.coordsZ = self.Z
        self.Y = self.U[:-1].flatten(start_dim=1)
        self.Y_plus = self.U[1:].flatten(start_dim=1)
        self.U = self.inputs[:-1]

    def POD(self, y_mean=None):
        snapshots = self.Y.T
        if y_mean is None:
            y_mean = torch.zeros((self.n_points * self.n_scalars, 1))
        elif y_mean.ndim == 1:
            y_mean = y_mean.reshape(-1,1)

        print('Running svd')

        return torch.linalg.svd(snapshots - y_mean, full_matrices=False)
