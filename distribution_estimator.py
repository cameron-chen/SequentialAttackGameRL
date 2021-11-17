import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import GameGeneration
from graph_convolution import GraphConvolution
from def_act_gen import dist_est, Def_Action_Generator, Def_A2C_Sample_Generator
from sampling import gen_init_def_pos
import configuration as config
import numpy as np
from argparse import ArgumentParser
import mix_den as MD
from mix_den import MixtureDensityNetwork


from torchvision import datasets, transforms

import tqdm
from matplotlib import pyplot as plt

# import normflow as nf

import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
rcParams['figure.dpi'] = 300


from torch import distributions
from torch.nn.parameter import Parameter

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

nets = lambda: nn.Sequential(nn.Linear(10, 5000), nn.LeakyReLU(), nn.Linear(5000, 5000), nn.LeakyReLU(), nn.Linear(5000, 10), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(10, 5000), nn.LeakyReLU(), nn.Linear(5000, 5000), nn.LeakyReLU(), nn.Linear(5000, 10))
masks = torch.from_numpy(np.random.rand(1000,5,10))
prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            print("MASK: ")
            print(self.mask[i])
            print("Z Value: ")
            print(z)
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x



class Distribution_Estimator(nn.Module):
    def __init__(self, num_targ, num_res):
        super(Distribution_Estimator, self).__init__()
        num_feat = round(num_res/2)
        self.num_targ = num_targ
        self.num_res = num_res
        self.conv1 = nn.Conv2d(1000, 1000, num_feat+1, 2)
        self.bn = nn.BatchNorm2d(1000)
        self.conv2 = nn.Conv2d(1000, 1000, round((num_feat+1)/4), 2)
        self.ln = nn.Linear(2000, 1000)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.bn(self.conv1(x))
        x = self.conv2(x)
        x = self.sig(self.ln(x.view(-1)))
        return x

class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)  

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu


def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI

def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


class DistributionEstimator(object):
    def __init__(self, num_targ, num_res, num_feature, payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints, device):
        self.num_targ = num_targ
        self.num_res = num_res
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.def_constraints = def_constraints
        self.device = device

        self.act_gen = Def_Action_Generator(self.num_targ, self.num_res, device).to(device)
        self.def_samp_gen = Def_A2C_Sample_Generator(payoff_matrix, adj_matrix, norm_adj_matrix, num_feature, num_res,
                                                    self.act_gen, device).to(device)

    def initial(self):
        return Distribution_Estimator(self.num_targ, self.num_res).to(self.device)

    def train(self, episodes=200, test=0):
        #distribution_estimator = MixtureDensityNetwork(1, 1, n_components=3)

        distribution_estimator = RealNVP(nets, nett, masks, prior)

        optimizer = torch.optim.Adam([p for p in distribution_estimator.parameters() if p.requires_grad==True], lr=1e-4)

        #distribution_estimator = Distribution_Estimator(self.num_targ, self.num_res).to(self.device)

        #dist_optim = optim.Adam(distribution_estimator.parameters(), lr=0.001)
        #criterion = nn.MSELoss()

        start = time.time()
        dist_est_loss_list = []
        lr = 0.001
        for i_episode in range(episodes):

            state = torch.zeros(self.num_targ, 2, dtype=torch.int32, device=self.device)
            def_cur_loc = gen_init_def_pos(self.num_targ, self.num_res, self.def_constraints, threshold=1)
            for t, res in enumerate(def_cur_loc):
                state[(res == 1).nonzero(), 0] += int(sum(res))

            act_estimates = self.def_samp_gen(state.unsqueeze(0), def_cur_loc)
            actions, act_probs, act_dist, codes = dist_est(act_estimates)

            #noisy_moons = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)

            a_est = act_estimates.detach().numpy()

            print("size of a_est")
            print(act_estimates.size())

            loss = -distribution_estimator.log_prob(torch.from_numpy(a_est)).mean()
        
            optimizer.zero_grad()

            #dist_optim.zero_grad()
            dist_estimates = distribution_estimator(act_estimates.unsqueeze(0))

            #pi_variable, sigma_variable, mu_variable = dist_estimates
            #loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, act_probs)

            #loss = MD.loss(dist_estimates, act_probs).mean()
            #loss = criterion(dist_estimates.view(-1), act_probs.view(-1))
            dist_est_loss_list.append(loss.item())

            if i_episode % 10 == 9:
                print("Episode", i_episode + 1, "-- Loss:", loss.item())
                # print("Actions:", len(act_dist.values()))

            if test and i_episode % 100 == 99:
                dist_dict = {k:[] for k,v in act_dist.items()}
                for i,p in enumerate(dist_estimates):
                    dist_dict[codes[i]].append(p.item())

                real_act_probs = [count/len(actions) for (code, count) in act_dist.items()]
                est_act_probs = [sum(p)/len(p) for (code, p) in dist_dict.items()]

                plt.figure(figsize=(20,5))
                plt.title("Distribution Estimate vs. Real Distribution")
                plt.xlabel("Action")
                plt.ylabel("Probability")
                plt.plot(est_act_probs, label = "Estimated Action Probabilities")
                plt.plot(real_act_probs, label = "Real Action Probabilities")
                plt.legend()
                plt.show()

            loss.backward(retain_graph=True)
            optimizer.step()

            if t % 500 == 0:
                print('iter %s:' % t, 'loss = %.3f' % loss)

            lr = lr * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print("\nTotal Runtime:", round((time.time() - start) / 60, 4), "min\n")

        if test:
            return distribution_estimator, dist_est_loss_list
        else:
            return distribution_estimator


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free',
                              num_res=config.NUM_RESOURCE, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, _ = game_gen.gen_game()
    def_constraints = [[0, 2], [1, 3], [4]]

    print("\nTraining Distribution Estimator for Defender Actions")
    dist_est_obj = DistributionEstimator(config.NUM_TARGET, config.NUM_RESOURCE, config.NUM_FEATURE, payoff_matrix,
                                         adj_matrix, norm_adj_matrix, def_constraints, device)
    distribution_estimator, loss_list = dist_est_obj.train(episodes=200, test=1)

    plt.figure(figsize=(20, 10))
    plt.title("Distribution Estimator Loss (1000 action samples)")
    plt.xlabel("Episode")
    plt.ylabel("MSE Loss")
    plt.plot(loss_list)
    plt.show()
