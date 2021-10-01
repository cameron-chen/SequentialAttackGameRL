from __future__ import print_function
import torch

import networkx as nx
import numpy as np
import random
import math
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Pure_Strategy = namedtuple('Pure_Strategy', ('type', 'trained_strategy', 'probability', 'name', 'value'))
# seed = 42


class GameGeneration(object):
    def __init__(self, num_target, graph_type, num_res, device):
        self.num_target = num_target
        self.graph_type = graph_type
        self.num_res = num_res
        self.device = device
        self.min_value = 1

    # Game Generation
    def gen_game(self):
        # random.seed(seed)
        if self.graph_type == 'random_scale_free':
            # graph = nx.scale_free_graph(self.num_target, seed=seed)
            graph = nx.scale_free_graph(self.num_target)
            adj_matrix = torch.from_numpy(nx.to_numpy_matrix(graph))
            for i in range(self.num_target):
                for j in range(i + 1, self.num_target):
                    if adj_matrix[i, j] > 0 or adj_matrix[j, i] > 0:
                        adj_matrix[i, j] = random.uniform(-0.2, -0.05)
                        # adj_matrix[i, j] = -0.01
                        adj_matrix[j, i] = adj_matrix[i, j]
                    else:
                        adj_matrix[i, j] = self.min_value
                        adj_matrix[j, i] = self.min_value
                adj_matrix[i, i] = 0

        adj_hat = adj_matrix + torch.eye(self.num_target)
        adj_hat = self.normalize(adj_hat)

        adj_matrix = adj_matrix.float().to(self.device)
        adj_hat = adj_hat.float().to(self.device)

        # Generate payoff matrix of the game
        # torch.manual_seed(seed)
        payoff_matrix = torch.rand(self.num_target, 4, dtype=torch.float32, device=self.device)
        payoff_matrix[:, 0] = payoff_matrix[:, 0] * 0.9 + 0.1
        payoff_matrix[:, 1] = payoff_matrix[:, 1] * 0.9 - 1.0
        # payoff_matrix[:, 2] = payoff_matrix[:, 2] * 0.9 + 0.1
        # payoff_matrix[:, 3] = payoff_matrix[:, 3] * 0.9 - 1.0
        payoff_matrix[:, 2] = -payoff_matrix[:, 1].clone()
        payoff_matrix[:, 3] = -payoff_matrix[:, 0].clone()

        # Generate defender resource constraints
        '''
        groups = random.randint(2, self.num_res-1)
        def_constraints = [[] for _ in range(groups)]
        count = self.num_res
        for g in def_constraints:
            empty_ct = len([x for x in def_constraints if len(x) < 1])
            group_max = count - empty_ct + 1
            if empty_ct == 1:
                add_res = count
            else:
                add_res = random.randint(1, group_max)
            for i in range(add_res):
                res = random.randint(0, self.num_target - 1)
                while res in (item for group in def_constraints for item in group):
                    res = random.randint(0, self.num_target - 1)
                g.append(res)
                count -= 1
        '''
        def_constraints = random.sample(range(0, self.num_target-1), int(self.num_res/2))

        return payoff_matrix, adj_matrix, adj_hat, def_constraints

    # Normalize adj matrix
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.matmul(r_mat_inv, mx)
        return mx


class ReplayMemoryEpisode(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def add_episode(self, episode):
        """Saves an episode."""
        self.memory.append(episode)

    def sample(self, batch_size, num_time_step):
        sampled_episode = random.sample(self.memory, batch_size)
        batch = []
        for episode in sampled_episode:
            point = np.random.randint(0, len(episode)+1-num_time_step)
            batch.append(episode[point:point+num_time_step])
        return batch

    def __len__(self):
        return len(self.memory)


class ReplayMemoryTransition(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=10, graph_type='random_scale_free', num_res=5, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
    #print(payoff_matrix)
    #print(adj_matrix)
    #print(norm_adj_matrix)
    print(def_constraints)