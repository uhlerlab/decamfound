
import numpy as np 
from numpy.linalg import lstsq
import causaldag as cd
import ipdb

import torch
import gpytorch
from gpytorch.kernels import AdditiveStructureKernel, ScaleKernel, RBFKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions.normal import Normal


class DeCAMFoundGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_parents, likelihood, include_node=True):
        assert num_parents > 0
        super(DeCAMFoundGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.num_parents = num_parents
        self.obs_active_dims = list(range(num_parents))
        self.k_obs = AdditiveStructureKernel(RBFKernel(has_lengthscale=True), active_dims=self.obs_active_dims, num_dims=train_x.shape[1])
        n_pcss = train_x.shape[1] - num_parents
        if n_pcss > 0:
            self.pcss_node_active_dim = [n_pcss + num_parents - 1]
            self.pcss_before_dims = list(range(num_parents, n_pcss + num_parents))
            self.k_confound_node = LinearKernel(active_dims=self.pcss_node_active_dim)
            self.k_confound_before = ScaleKernel(RBFKernel(active_dims=self.pcss_before_dims, ard_num_dims=n_pcss))
            if include_node:
                self.k_decamfound = self.k_obs + self.k_confound_node + self.k_confound_before
            else:
                self.k_decamfound = self.k_obs + self.k_confound_before
        else:
            self.k_decamfound = self.k_obs

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.k_decamfound(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def log_marginal_like_MAP(X_pa, pcss_before, X_node, include_node, use_gpu=False, verbose=False, training_iter=100):
    x_train = torch.cat([X_pa, pcss_before], axis=1)
    likelihood = GaussianLikelihood()
    model = DeCAMFoundGPModel(x_train, X_node, X_pa.shape[1], likelihood, include_node=include_node)

    likelihood.double()
    model.double()

    if use_gpu:
        x_train = x_train.cuda()
        X_node = X_node.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter): # Perform gradient descent in for -marginal liklihood
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, X_node)
        loss.backward()
        if verbose and (i % 50 == 0):
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

    avg_mll = -1 * loss.item()
    model.eval()
    likelihood.eval()
    f_pred_mean = model(x_train)
    return avg_mll, f_pred_mean.mean.cpu().detach().numpy()


def decamfound_mll_score(G, suffstats, include_node=False, make_decomposable=True, use_gpu=False, verbose=False, training_iter=100):
    topological_order = G.topological_sort()
    log_marg_like = 0
    obs_data = torch.from_numpy(suffstats['obs_data']).double()
    N = obs_data.shape[0]
    pcss = torch.from_numpy(suffstats['pcss']).double()
    assert (pcss.shape[1] == 0) or (pcss.shape[1] == obs_data.shape[1])

    node_marg_likes = []
    node_fitted_vals = dict()
    for ix, node in enumerate(topological_order):
        X_node = obs_data[:, node]
        pa_node = sorted(list(G.parents_of(node)))
        if len(pa_node) == 0: # source node in graph
            if pcss.shape[1] == 0:
                est_noise_sd = X_node.std().item() # MLE estimate of noise
                node_marg_likes.append(Normal(loc=0, scale=est_noise_sd).log_prob(X_node).mean().item()) 
            else:
                X_node_resid = X_node - pcss[:, node] # For a source node, s_j - r_j = s_j = pcss[node]
                est_noise_sd = X_node_resid.std().item()
                node_marg_likes.append(Normal(loc=0, scale=est_noise_sd).log_prob(X_node_resid).mean().item()) 
        else:
            X_pa = obs_data[:, pa_node]
            if pcss.shape[1] > 0:
                if make_decomposable:
                    pcss_before = pcss[:, pa_node + [node]] # Only use parent node pcss variables 
                else:
                    pcss_before = pcss[:, topological_order[:(ix + 1)]]
            else:
                pcss_before = torch.zeros((obs_data.shape[0], 0))
            if not include_node and (pcss.shape[1] > 0):
                X_node = X_node - pcss[:, node]
            avg_mll, f_preds_nodes = log_marginal_like_MAP(X_pa, pcss_before, X_node, include_node=include_node, use_gpu=use_gpu, verbose=verbose, training_iter=training_iter)
            node_marg_likes.append(avg_mll)
            node_fitted_vals[node] = f_preds_nodes

    node_marg_likes = np.array(node_marg_likes)
    return  N*np.sum(node_marg_likes), N*node_marg_likes, node_fitted_vals # Divided by N for normalization when training the GPs
