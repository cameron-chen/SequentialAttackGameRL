import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import GameGeneration
from graph_convolution import GraphConvolution
from def_act_gen import dist_est, Def_Action_Generator
from sampling import gen_init_def_pos
import configuration as config
import numpy as np
from argparse import ArgumentParser
import mix_den as MD
from mix_den import MixtureDensityNetwork


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


class Def_A2C_Sample_Generator(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature,
                 num_resource, device):
        super(Def_A2C_Sample_Generator, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norma_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)
        self.num_resource = num_resource
        self.device = device
        self.noise_feat = 2

        self.gc1 = GraphConvolution(num_feature+self.noise_feat, 32)
        self.bn = nn.BatchNorm1d(self.num_target)

        self.gc2 = GraphConvolution(32, 16)

        self.ln_value1 = nn.Linear(16 * self.num_target, 32)
        self.ln_value2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.act_gen = Def_Action_Generator(self.num_target, self.num_resource, device).to(device)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state, def_cur_loc):
        batch_size = len(state)
        noise = torch.rand((1, state.size(1), self.noise_feat)).to(self.device)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1), noise), 2)

        x = self.relu(self.bn(self.gc1(x, self.norma_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norma_adj_matrix)))
        # x = self.relu(self.ln1(x))
        x = x.view(-1, 16 * self.num_target)

        act_estimates = self.act_gen(x.squeeze(), def_cur_loc)
        for i in range(1000-1):
            act_estimates = torch.cat((act_estimates, self.act_gen(x.squeeze(), def_cur_loc)))

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        return act_estimates.view(1000, self.num_resource, self.num_target)


class DistributionEstimator(object):
    def __init__(self, num_targ, num_res, num_feature, payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints, device):
        self.num_targ = num_targ
        self.num_res = num_res
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.def_constraints = def_constraints
        self.device = device

        self.def_samp_gen = Def_A2C_Sample_Generator(payoff_matrix, adj_matrix, norm_adj_matrix, num_feature, num_res,
                                                    device).to(device)

    def initial(self):
        return Distribution_Estimator(self.num_targ, self.num_res).to(self.device)

    def train(self, episodes=200, test=0):
        #distribution_estimator = MixtureDensityNetwork(1, 1, n_components=3)
        distribution_estimator = Distribution_Estimator(self.num_targ, self.num_res).to(self.device)
        dist_optim = optim.Adam(distribution_estimator.parameters(), lr=0.001)
        criterion = nn.MSELoss()

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

            dist_optim.zero_grad()
            dist_estimates = distribution_estimator(act_estimates.unsqueeze(0))

            #pi_variable, sigma_variable, mu_variable = dist_estimates
            #loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, act_probs)

            #loss = MD.loss(dist_estimates, act_probs).mean()
            loss = criterion(dist_estimates.view(-1), act_probs.view(-1))
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

            loss.backward()
            dist_optim.step()

            lr = lr * 0.95
            for param_group in dist_optim.param_groups:
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
