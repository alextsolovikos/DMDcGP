import torch
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp
import numpy as np
import time
import copy
from tqdm import tqdm

import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean, ZeroMean

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ZeroMean()
            self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DiscreteDynamicsIndependentGPs(object):
    def __init__(self, train_x, train_u, train_xp, min_sigma=0.1):

        self.n_train = train_x.shape[0]
        self.nx = train_x.shape[1]
        self.nu = train_u.shape[1]
        self.nxp = train_xp.shape[1]

        train_input = torch.cat((train_x, train_u), dim=1)

        obs_noise_constraint = gpytorch.constraints.GreaterThan(min_sigma**2)

        # GP model
        models = []
        for i in range(self.nxp):
            likelihood_i = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=obs_noise_constraint)
            models.append(ExactGPModel(train_input, train_xp[:,i], likelihood_i))

        self.model = gpytorch.models.IndependentModelList(*models)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*[model.likelihood for model in models])

    def train(self, training_iter=100, lr=0.1, verbose=True, model_path='data/gp_model.pth', cuda=False):
        mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        # Move data to GPU
        if cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

        bar = tqdm(range(training_iter), desc=f'Training GP Hyperparameters | Loss = {0:.4f}', leave=True)
        for i in bar:
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -mll(output, self.model.train_targets)
            loss.backward()
            bar.set_description(f'Training GP Hyperparameters | Loss = {loss.item():.4f}', refresh=True)
            # if verbose:
            #     print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()

        print('Finished Training')

        # Move back to CPU
        if cuda:
            self.model = self.model.cpu()
            self.likelihood = self.likelihood.cpu()

        self.model.eval()
        self.likelihood.eval()

    def load(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)


    def predict(self, test_x, test_u):
        test_x = torch.Tensor(test_x).reshape(-1,self.nx)
        test_u = torch.Tensor(test_u).reshape(-1,self.nu)

        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions (use the same test points)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_input = torch.cat((test_x, test_u), dim=1)

            # This contains predictions for all outputs as a list
            predictions = self.likelihood(*self.model(*[test_input for k in range(self.nxp)]))

        n_test = test_x.shape[0]
        mean_xp = torch.zeros((n_test, self.nxp))
        std_xp = torch.zeros((n_test, self.nxp))

        for i, prediction in enumerate(predictions):
            mean_xp[:,i] = prediction.mean
            std_xp[:,i] = prediction.stddev

        return mean_xp.detach().flatten(), std_xp.detach().flatten()

    def dx_with_grad(self, x, u):
        x = x.reshape(-1,self.nx)
        u = u.reshape(-1,self.nu)
        test_input = torch.cat((x, u), dim=1)
        predictions = self.likelihood(*self.model(*[test_input for k in range(self.nxp)]))
        mean_xp = torch.zeros_like(x)
        for i, prediction in enumerate(predictions):
            mean_xp[:,i] = prediction.mean
        return mean_xp
        
    def linearize(self, x, u):        
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            u = torch.from_numpy(u).float()
        
        x = x.requires_grad_(True)
        u = u.requires_grad_(True)
        A, B = torch.autograd.functional.jacobian(self.dx_with_grad, inputs=(x, u))
        xp_mean, _ = self.predict(x, u)
        d = - A.matmul(x) - B.matmul(u) + xp_mean
    
        return A[0].detach().cpu(), B[0].detach().cpu(), d[0].detach().cpu()