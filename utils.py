from __future__ import print_function
import torch

import networkx as nx
import numpy as np
import random
import math
from collections import namedtuple, deque

from game_simulation import GameSimulation

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

        # def_constraints = random.sample(range(0, self.num_target-1), int(self.num_res/2))

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


def play_game(def_strat, att_strat, payoff_matrix, adj_matrix, def_constraints, d_option, a_option):
        def_utility_average = 0.0
        att_utility_average = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_target = payoff_matrix.size(0)
        lstm_hidden_size = configuration.LSTM_HIDDEN_SIZE
        n_sample = 50

        for i_sample in range(n_sample):
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=device)
            if 'GAN' in d_option:
                def_init_loc = gen_init_def_pos(num_target, configuration.NUM_RESOURCE, def_constraints, threshold=1)
                for t, res in enumerate(def_init_loc):
                    init_state[(res == 1).nonzero(), 0] += int(sum(res))
            else:
                entries = torch.randint(0, num_target, [configuration.NUM_RESOURCE, ])
                for t in range(0, len(entries)):
                    init_state[entries[t], 0] += 1

            state = init_state
            init_attacker_observation = torch.zeros(num_target, 2, dtype=torch.int32, device=device)
            init_attacker_observation[:, 0] = -1
            attacker_observation = init_attacker_observation
            num_att = configuration.NUM_ATTACK

            if 'LSTM' in d_option:
                d_action_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                d_action_cell_state = torch.zeros(1, lstm_hidden_size, device=device)
                d_value_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                d_value_cell_state = torch.zeros(1, lstm_hidden_size, device=device)

            if 'LSTM' in a_option:
                a_action_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                a_action_cell_state = torch.zeros(1, lstm_hidden_size, device=device)
                a_value_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                a_value_cell_state = torch.zeros(1, lstm_hidden_size, device=device)

            # for t in range(config.NUM_STEP):
            while num_att > 0:
                with torch.no_grad():
                    if 'LSTM' in d_option:
                        def_actor, def_critic, d_action_hidden_state, d_action_cell_state, d_value_hidden_state, d_value_cell_state \
                            = def_strat(state=state.unsqueeze(0), action_hidden_state=d_action_hidden_state,
                                        action_cell_state=d_action_cell_state, value_hidden_state=d_value_hidden_state,
                                        value_cell_state=d_value_cell_state)
                    else:
                        def_actor, def_critic = def_strat(state=state.unsqueeze(0))

                    if 'LSTM' in a_option:
                        att_actor, att_critic, a_action_hidden_state, a_action_cell_state, a_value_hidden_state, a_value_cell_state \
                            = att_strat(state=attacker_observation.unsqueeze(0),
                                        action_hidden_state=a_action_hidden_state,
                                        action_cell_state=a_action_cell_state, value_hidden_state=a_value_hidden_state,
                                        value_cell_state=a_value_cell_state)
                    else:
                        att_actor, att_critic = att_strat(state=attacker_observation.unsqueeze(0))

                    if num_att < configuration.NUM_ATTACK and state[:, 0].sum() == 0:
                        def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=device)
                    elif 'GAN' in d_option:
                        def_action = GameSimulation.sample_def_action_from_res_dist(state=state, distributions=def_actor.squeeze(0),
                                                                                    device=device)
                    else:
                        def_action = GameSimulation.sample_def_action_from_distribution(state=state, distributions=def_actor.squeeze(0),
                                                                                        def_constraints=def_constraints,
                                                                                        device=device)
                    att_action = GameSimulation.sample_att_action_from_distribution(distribution=att_actor.squeeze(0),
                                                                                    num_att=num_att,
                                                                                    device=device)

                    if 'GAN' in d_option:
                        next_state, def_immediate_utility, att_immediate_utility \
                            = GameSimulation.gen_next_state_from_def_res(state, def_action, att_action, payoff_matrix,
                                                                         adj_matrix)
                    else:
                        next_state, def_immediate_utility, att_immediate_utility \
                            = GameSimulation.gen_next_state(state=state, def_action=def_action, att_action=att_action,
                                                            payoff_matrix=payoff_matrix, adj_matrix=adj_matrix)
                    next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                               def_action=def_action,
                                                                               att_action=att_action)

                    def_utility_average += def_immediate_utility
                    att_utility_average += att_immediate_utility

                    state = next_state
                    attacker_observation = next_att_observation
                    num_att -= sum(att_action).item()

        def_utility_average /= n_sample
        att_utility_average /= n_sample

        return def_utility_average.item(), att_utility_average.item()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=10, graph_type='random_scale_free', num_res=5, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
    #print(payoff_matrix)
    #print(adj_matrix)
    #print(norm_adj_matrix)
    print(def_constraints)