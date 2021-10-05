import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import GameGeneration
from graph_convolution import GraphConvolution
from def_gan import Def_Action_Generator, dist_est
from sampling import gen_init_def_pos
import configuration as config


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
        x = self.bn(self.conv2(x))
        x = self.sig(self.ln(x.view(-1)))
        return x


class Def_A2C_Action_Generator(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature,
                 num_resource, device):
        super(Def_A2C_Action_Generator, self).__init__()
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

        self.act_gen = Def_Action_Generator(device).to(device)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state):
        batch_size = len(state)
        noise = torch.rand((1, state.size(1), self.noise_feat)).to(self.device)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1), noise), 2)

        x = self.relu(self.bn(self.gc1(x, self.norma_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norma_adj_matrix)))
        x = x.view(-1, 16 * self.num_target)

        act_estimates = self.act_gen(x.squeeze())
        for i in range(1000-1):
            act_estimates = torch.cat((act_estimates, self.act_gen(x.squeeze())))

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

        self.def_act_gen = Def_A2C_Action_Generator(payoff_matrix, adj_matrix, norm_adj_matrix, num_feature, num_res,
                                                    device).to(device)

    def train(self, test=None, episodes=150):
        distribution_estimator = Distribution_Estimator(self.num_targ, self.num_res).to(self.device)
        dist_optim = optim.Adam(distribution_estimator.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        print("Training Distribution Estimator for Defender Actions")
        start = time.time()
        dist_est_loss_list = []
        lr = 0.001
        for i_episode in range(episodes):
            state = torch.zeros(self.num_targ, 2, dtype=torch.int32, device=device)
            def_cur_loc = gen_init_def_pos(self.num_targ, self.num_res, self.def_constraints, threshold=1)
            for t, res in enumerate(def_cur_loc):
                state[(res == 1).nonzero(), 0] += int(sum(res))

            act_estimates = self.def_act_gen(state.unsqueeze(0))
            actions, act_probs, act_dist = dist_est(act_estimates)

            dist_optim.zero_grad()
            dist_estimate = distribution_estimator(act_estimates.unsqueeze(0))

            loss = criterion(dist_estimate, act_probs)
            dist_est_loss_list.append(loss.item())

            if test and i_episode % 10 == 9:
                print("\nEpisode", i_episode + 1)
                print("Loss:", loss.item())

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

    dist_est_obj = DistributionEstimator(config.NUM_TARGET, config.NUM_RESOURCE, config.NUM_FEATURE, payoff_matrix,
                                         adj_matrix, norm_adj_matrix, def_constraints, device)
    distribution_estimator, loss_list = dist_est_obj.train(test=1, episodes=150)

    plt.figure(figsize=(20, 10))
    plt.title("Distribution Estimator Loss (1000 action samples)")
    plt.xlabel("Episode")
    plt.ylabel("MSE Loss")
    plt.plot(loss_list)
    plt.show()
