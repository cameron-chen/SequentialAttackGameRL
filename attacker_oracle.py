from __future__ import print_function

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random

from attacker_model import Att_A2C_GCN, Att_A2C_GCN_LSTM
from defender_model import Def_A2C_GCN, Def_A2C_GCN_LSTM
from utils import ReplayMemoryTransition, ReplayMemoryEpisode, GameGeneration, Transition
from game_simulation import GameSimulation
from optimization import Optimization
import configuration as config
from utils import Pure_Strategy


class AttackerOracle(object):
    def __init__(self, def_strategy, payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints, device):
        self.def_strategy = def_strategy
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.def_constraints = def_constraints
        self.num_att = config.NUM_ATTACK
        self.device = device

    def update_def_strategy(self, def_strategy):
        self.def_strategy = def_strategy

    def train(self, option, policy=None, test=None):
        if option == 'A2C-GCN':
            optimization = Optimization(batch_size=config.BATCH_SIZE_TRANSITION, num_step=config.NUM_STEP,
                                        gamma=config.GAMMA, device=self.device, payoff_matrix=self.payoff_matrix,
                                        adj_matrix=self.adj_matrix, norm_adj_matrix=self.norm_adj_matrix)
            policy_net = Att_A2C_GCN(payoff_matrix=self.payoff_matrix, norm_adj_matrix=self.norm_adj_matrix,
                                     num_feature=config.NUM_FEATURE).to(self.device)
            if policy is not None:
                policy_net.load_state_dict(policy.state_dict())
            target_net = Att_A2C_GCN(payoff_matrix=self.payoff_matrix, norm_adj_matrix=self.norm_adj_matrix,
                                     num_feature=config.NUM_FEATURE).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            # print('Running Attacker A2C-GCN...')
            att_set = self.train_A2C(policy_net=policy_net, target_net=target_net, optimization=optimization,
                                     memory_size=config.MEMORY_SIZE_TRANSITION, learn_rate=config.LR_TRANSITION,
                                     entropy_coeff=config.ENTROPY_COEFF_TRANS, test=test)

            return att_set
        elif option == 'A2C-GCN-LSTM':
            optimization = Optimization(batch_size=config.BATCH_SIZE_EPISODE, num_step=config.NUM_STEP,
                                        gamma=config.GAMMA, device=self.device, payoff_matrix=self.payoff_matrix,
                                        adj_matrix=self.adj_matrix, norm_adj_matrix=self.norm_adj_matrix)
            policy_net = Att_A2C_GCN_LSTM(payoff_matrix=self.payoff_matrix, norm_adj_matrix=self.norm_adj_matrix,
                                          num_feature=config.NUM_FEATURE,
                                          lstm_hidden_size=config.LSTM_HIDDEN_SIZE).to(self.device)
            if policy is not None:
                policy_net.load_state_dict(policy.state_dict())
            target_net = Att_A2C_GCN_LSTM(payoff_matrix=self.payoff_matrix, norm_adj_matrix=self.norm_adj_matrix,
                                          num_feature=config.NUM_FEATURE,
                                          lstm_hidden_size=config.LSTM_HIDDEN_SIZE).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            # print('Running Attacker A2C-GCN-LSTM...')
            att_set = self.train_A2C_LSTM(policy_net=policy_net, target_net=target_net, optimization=optimization,
                                          memory_size=config.MEMORY_SIZE_EPISODE,
                                          lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
                                          learn_rate=config.LR_EPISODE, entropy_coeff=config.ENTROPY_COEFF_EPS,
                                          test=test)

            return att_set
        elif option == 'DQN-GCN':
            return
        elif option == 'DQN-GCN-LSTM':
            return
        else:
            print('Option is unavailable!')

    def train_A2C(self, policy_net, target_net, optimization, memory_size, learn_rate, entropy_coeff, test):
        # def_mixed strategy is a tensor, each element is a tuple <type, >
        num_target = self.payoff_matrix.size(0)
        optimizer = optim.RMSprop(policy_net.parameters(), learn_rate)
        memory = ReplayMemoryTransition(memory_size)

        init_attacker_observation = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
        init_attacker_observation[:, 0] = -1
        # state = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)

        att_avg_utils = []

        for i_episode in range(config.NUM_EPISODE):
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
            entries = torch.randint(0, num_target,
                                    [config.NUM_RESOURCE, ])  # defender can have multiple resources at one target
            # entries = random.sample(range(0, num_target), config.NUM_RESOURCE)  # defender can only have one resource per target
            for t in range(0, len(entries)):
                init_state[entries[t], 0] += 1
            state = init_state
            attacker_observation = init_attacker_observation
            num_att = self.num_att
            def_pure_strategy = GameSimulation.sample_pure_strategy(self.def_strategy)
            if 'LSTM' in def_pure_strategy.type and 'A2C' in def_pure_strategy.type:
                def_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                def_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                def_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                def_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
            # temp = count()

            # for t in range(config.NUM_STEP):
            while num_att > 0:
                with torch.no_grad():
                    actor, critic \
                        = policy_net(state=attacker_observation.unsqueeze(0))
                    att_action = GameSimulation.sample_att_action_from_distribution(distribution=actor.squeeze(0),
                                                                                    num_att=num_att,
                                                                                    device=self.device)

                # -------------------------------- Start sample defender action --------------------------------
                if def_pure_strategy.type == 'uniform':
                    def_action = GameSimulation.sample_def_action_uniform(state=state, adj_matrix=self.adj_matrix
                                                                          , device=self.device)
                elif def_pure_strategy.type == 'suqr':
                    def_action = GameSimulation.sample_def_action_suqr(state=state, payoff=self.payoff_matrix,
                                                                       adj=self.adj_matrix, device=self.device)
                elif def_pure_strategy.type == 'A2C-GCN':
                    def_action = GameSimulation.sample_def_action_A2C(num_attack_remain=num_att, state=state,
                                                                      trained_strategy=def_pure_strategy.trained_strategy,
                                                                      def_constraints=self.def_constraints,
                                                                      device=self.device)
                elif def_pure_strategy.type == 'A2C-GCN-LSTM':
                    def_action, def_action_hidden_state, def_action_cell_state, \
                    def_value_hidden_state, def_value_cell_state = GameSimulation.sample_def_action_A2C_LSTM(
                        num_attack_remain=num_att, state=state, trained_strategy=def_pure_strategy.trained_strategy,
                        action_hidden_state=def_action_hidden_state, action_cell_state=def_action_cell_state,
                        value_hidden_state=def_value_hidden_state, value_cell_state=def_value_cell_state,
                        def_constraints=self.def_constraints, device=self.device)
                elif def_pure_strategy.type == 'A2C-GCN-GAN':
                    def_action = GameSimulation.sample_def_action_GAN(step=(self.num_att - num_att), state=state,
                                                                      trained_strategy=def_pure_strategy.trained_strategy,
                                                                      device=self.device)
                # -------------------------------- End sample defender action --------------------------------

                if def_pure_strategy.type == 'A2C-GCN-GAN':
                    next_state, def_immediate_utility, att_immediate_utility \
                        = GameSimulation.gen_next_state_from_def_res(state, def_action, att_action, self.payoff_matrix,
                                                                     self.adj_matrix)
                else:
                    next_state, def_immediate_utility, att_immediate_utility \
                        = GameSimulation.gen_next_state(state, def_action, att_action, self.payoff_matrix,
                                                        self.adj_matrix)
                next_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                       def_action=def_action, att_action=att_action)

                # Store the transition in memory
                att_immediate_utility = torch.tensor([att_immediate_utility], device=self.device)
                memory.push(attacker_observation.unsqueeze(0), att_action.unsqueeze(0), next_observation.unsqueeze(0)
                            , att_immediate_utility.unsqueeze(0))

                attacker_observation = next_observation
                num_att -= sum(att_action).item()

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                optimization.optimize_Att_A2C(memory, policy_net, target_net, optimizer, entropy_coeff)

            # Update the target network, copying all weights and biases in DQN
            if i_episode % config.TARGET_UPDATE_TRANSITION == 0:
                target_net.load_state_dict(policy_net.state_dict())

            learn_rate = learn_rate * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = learn_rate

            # Evaluation only, should not be included in the runtime evaluation
            if i_episode > 0 and i_episode % 10 == 0:
                att_utility_average = 0.0
                def_utility_average = 0.0
                for i_sample in range(config.NUM_SAMPLE):
                    init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
                    entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])
                    for t in range(0, len(entries)):
                        init_state[entries[t], 0] += 1
                    state = init_state
                    attacker_observation = init_attacker_observation
                    num_att = self.num_att
                    def_pure_strategy = GameSimulation.sample_pure_strategy(self.def_strategy)
                    if 'LSTM' in def_pure_strategy.type and 'A2C' in def_pure_strategy.type:
                        def_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        def_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        def_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        def_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                    # for t in range(config.NUM_STEP):
                    while num_att > 0:
                        with torch.no_grad():
                            actor, critic \
                                = policy_net(state=attacker_observation.unsqueeze(0))
                            att_action = GameSimulation.sample_att_action_from_distribution(
                                distribution=actor.squeeze(0),
                                num_att=num_att,
                                device=self.device)

                        # -------------------------------- Start sample defender action --------------------------------
                        if def_pure_strategy.type == 'uniform':
                            def_action = GameSimulation.sample_def_action_uniform(state=state,
                                                                                  adj_matrix=self.adj_matrix,
                                                                                  device=self.device)
                        elif def_pure_strategy.type == 'suqr':
                            def_action = GameSimulation.sample_def_action_suqr(state=state, payoff=self.payoff_matrix,
                                                                               adj=self.adj_matrix, device=self.device)
                        elif def_pure_strategy.type == 'A2C-GCN':
                            def_action = GameSimulation.sample_def_action_A2C(num_attack_remain=num_att, state=state,
                                                                              trained_strategy=def_pure_strategy.trained_strategy,
                                                                              def_constraints=self.def_constraints,
                                                                              device=self.device)
                        elif def_pure_strategy.type == 'A2C-GCN-LSTM':
                            def_action, def_action_hidden_state, def_action_cell_state, \
                            def_value_hidden_state, def_value_cell_state = GameSimulation.sample_def_action_A2C_LSTM(
                                num_attack_remain=num_att, state=state, trained_strategy=def_pure_strategy.trained_strategy,
                                action_hidden_state=def_action_hidden_state, action_cell_state=def_action_cell_state,
                                value_hidden_state=def_value_hidden_state, value_cell_state=def_value_cell_state,
                                def_constraints=self.def_constraints, device=self.device)
                        elif def_pure_strategy.type == 'A2C-GCN-GAN':
                            def_action = GameSimulation.sample_def_action_GAN(step=(self.num_att - num_att), state=state,
                                                                              trained_strategy=def_pure_strategy.trained_strategy,
                                                                              device=self.device)
                        # -------------------------------- End sample defender action --------------------------------

                        if def_pure_strategy.type == 'A2C-GCN-GAN':
                            next_state, def_immediate_utility, att_immediate_utility \
                                = GameSimulation.gen_next_state_from_def_res(state, def_action, att_action, self.payoff_matrix,
                                                                             self.adj_matrix)
                        else:
                            next_state, def_immediate_utility, att_immediate_utility \
                                = GameSimulation.gen_next_state(state, def_action, att_action, self.payoff_matrix,
                                                                self.adj_matrix)
                        next_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                               def_action=def_action,
                                                                               att_action=att_action)

                        att_utility_average += att_immediate_utility
                        def_utility_average += def_immediate_utility
                        attacker_observation = next_observation
                        state = next_state
                        num_att -= sum(att_action).item()

                att_utility_average /= config.NUM_SAMPLE
                def_utility_average /= config.NUM_SAMPLE
                # print(i_episode, ": Attacker Oracle: ", att_utility_average)
                att_avg_utils.append((policy_net, att_utility_average.item(), def_utility_average.item()))

                if test is not None:
                    print('Episode %d, Attacker Utility: %.4f, Defender Utility: %.4f'
                          % (i_episode, att_utility_average.item(), def_utility_average.item()))

        # best_policy_net, best_att_util, def_util = max(att_avg_utils, key=lambda x: x[1])

        return att_avg_utils  # best_policy_net, best_att_util, def_util

    def train_A2C_LSTM(self, policy_net, target_net, optimization, memory_size, lstm_hidden_size, learn_rate,
                       entropy_coeff, test):
        # def_mixed strategy is a tensor, each element is a tuple <type, >
        num_target = self.payoff_matrix.size(0)
        optimizer = optim.RMSprop(policy_net.parameters(), learn_rate)
        memory = ReplayMemoryEpisode(memory_size)

        init_attacker_observation = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
        init_attacker_observation[:, 0] = -1
        # state = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
        init_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
        init_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
        init_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
        init_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)

        att_avg_utils = []

        for i_episode in range(config.NUM_EPISODE):
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
            entries = torch.randint(0, num_target,
                                    [config.NUM_RESOURCE, ])  # defender can have multiple resources at one target
            # entries = random.sample(range(0, num_target), config.NUM_RESOURCE)  # defender can only have one resource per target
            for t in range(0, len(entries)):
                init_state[entries[t], 0] += 1
            state = init_state
            attacker_observation = init_attacker_observation
            num_att = self.num_att
            def_pure_strategy = GameSimulation.sample_pure_strategy(self.def_strategy)
            if 'LSTM' in def_pure_strategy.type and 'A2C' in def_pure_strategy.type:
                def_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                def_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                def_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                def_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)

            temp_episode = []
            action_hidden_state = init_action_hidden_state
            action_cell_state = init_action_cell_state
            value_hidden_state = init_value_hidden_state
            value_cell_state = init_value_cell_state

            # for t in range(config.NUM_STEP):
            while num_att > 0:
                with torch.no_grad():
                    actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                        = policy_net(state=attacker_observation.unsqueeze(0)
                                     , action_hidden_state=action_hidden_state, action_cell_state=action_cell_state,
                                     value_hidden_state=value_hidden_state, value_cell_state=value_cell_state)
                    att_action = GameSimulation.sample_att_action_from_distribution(distribution=actor.squeeze(0),
                                                                                    num_att=num_att,
                                                                                    device=self.device)

                # -------------------------------- Start sample defender action --------------------------------
                if def_pure_strategy.type == 'uniform':
                    def_action = GameSimulation.sample_def_action_uniform(state=state, adj_matrix=self.adj_matrix,
                                                                          device=self.device)
                elif def_pure_strategy.type == 'suqr':
                    def_action = GameSimulation.sample_def_action_suqr(state=state, payoff=self.payoff_matrix,
                                                                       adj=self.adj_matrix, device=self.device)
                elif def_pure_strategy.type == 'A2C-GCN':
                    def_action = GameSimulation.sample_def_action_A2C(num_attack_remain=num_att, state=state,
                                                                      trained_strategy=def_pure_strategy.trained_strategy,
                                                                      def_constraints=self.def_constraints,
                                                                      device=self.device)
                elif def_pure_strategy.type == 'A2C-GCN-LSTM':
                    def_action, def_action_hidden_state, def_action_cell_state, \
                    def_value_hidden_state, def_value_cell_state = GameSimulation.sample_def_action_A2C_LSTM(
                        num_attack_remain=num_att, state=state, trained_strategy=def_pure_strategy.trained_strategy,
                        action_hidden_state=def_action_hidden_state, action_cell_state=def_action_cell_state,
                        value_hidden_state=def_value_hidden_state, value_cell_state=def_value_cell_state,
                        def_constraints=self.def_constraints, device=self.device)
                elif def_pure_strategy.type == 'A2C-GCN-GAN':
                    def_action = GameSimulation.sample_def_action_GAN(step=(self.num_att - num_att), state=state,
                                                                      trained_strategy=def_pure_strategy.trained_strategy,
                                                                      device=self.device)
                # -------------------------------- End sample defender action --------------------------------

                if def_pure_strategy.type == 'A2C-GCN-GAN':
                    next_state, def_immediate_utility, att_immediate_utility \
                        = GameSimulation.gen_next_state_from_def_res(state, def_action, att_action, self.payoff_matrix,
                                                                     self.adj_matrix)
                else:
                    next_state, def_immediate_utility, att_immediate_utility \
                        = GameSimulation.gen_next_state(state, def_action, att_action, self.payoff_matrix, self.adj_matrix)
                next_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                       def_action=def_action, att_action=att_action)

                # Store the transition in memory
                att_immediate_utility = torch.tensor([att_immediate_utility], device=self.device)
                temp_transition = Transition(attacker_observation.unsqueeze(0), att_action.unsqueeze(0)
                                             , next_observation.unsqueeze(0), att_immediate_utility.unsqueeze(0))
                temp_episode.append(temp_transition)

                attacker_observation = next_observation
                num_att -= sum(att_action).item()

                # Move to the next state
                state = next_state

            memory.add_episode(temp_episode)
            # Update the target network, copying all weights and biases in DQN
            if i_episode % config.TARGET_UPDATE_EPISODE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if i_episode % config.OPTIMIZE_UPDATE_EPISODE == 0:
                # Perform one step of the optimization (on the target network)
                optimization.optimize_Att_A2C_LSTM(memory, policy_net, target_net, optimizer,
                                                   lstm_hidden_size, entropy_coeff)

            learn_rate = learn_rate * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = learn_rate

            # Evaluation only, should not be included in the runtime evaluation
            if i_episode > 0 and i_episode % 10 == 0:
                att_utility_average = 0.0
                def_utility_average = 0.0
                for i_sample in range(config.NUM_SAMPLE):
                    init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
                    entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])
                    for t in range(0, len(entries)):
                        init_state[entries[t], 0] += 1
                    state = init_state
                    action_hidden_state = init_action_hidden_state
                    action_cell_state = init_action_cell_state
                    value_hidden_state = init_value_hidden_state
                    value_cell_state = init_value_cell_state
                    attacker_observation = init_attacker_observation
                    num_att = self.num_att
                    def_pure_strategy = GameSimulation.sample_pure_strategy(self.def_strategy)
                    if 'LSTM' in def_pure_strategy.type and 'A2C' in def_pure_strategy.type:
                        def_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        def_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        def_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        def_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                    # for t in range(config.NUM_STEP):
                    while num_att > 0:
                        with torch.no_grad():
                            actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                                = policy_net(state=attacker_observation.unsqueeze(0),
                                             action_hidden_state=action_hidden_state,
                                             action_cell_state=action_cell_state,
                                             value_hidden_state=value_hidden_state, value_cell_state=value_cell_state)
                            att_action = GameSimulation.sample_att_action_from_distribution(
                                distribution=actor.squeeze(0),
                                num_att=num_att,
                                device=self.device)

                        # -------------------------------- Start sample defender action --------------------------------
                        if def_pure_strategy.type == 'uniform':
                            def_action = GameSimulation.sample_def_action_uniform(state=state,
                                                                                  adj_matrix=self.adj_matrix,
                                                                                  device=self.device)
                        elif def_pure_strategy.type == 'A2C-GCN':
                            def_action = GameSimulation.sample_def_action_A2C(num_attack_remain = num_att, state=state,
                                                                              trained_strategy=def_pure_strategy.trained_strategy,
                                                                              def_constraints=self.def_constraints,
                                                                              device=self.device)
                        elif def_pure_strategy.type == 'A2C-GCN-LSTM':
                            def_action, def_action_hidden_state, def_action_cell_state, \
                            def_value_hidden_state, def_value_cell_state = GameSimulation.sample_def_action_A2C_LSTM(
                                num_attack_remain=num_att, state=state, trained_strategy=def_pure_strategy.trained_strategy,
                                action_hidden_state=def_action_hidden_state, action_cell_state=def_action_cell_state,
                                value_hidden_state=def_value_hidden_state, value_cell_state=def_value_cell_state,
                                def_constraints=self.def_constraints, device=self.device)
                        elif def_pure_strategy.type == 'A2C-GCN-GAN':
                            def_action = GameSimulation.sample_def_action_GAN(step=(self.num_att - num_att), state=state,
                                                                              trained_strategy=def_pure_strategy.trained_strategy,
                                                                              device=self.device)
                        # -------------------------------- End sample defender action --------------------------------

                        if def_pure_strategy.type == 'A2C-GCN-GAN':
                            next_state, def_immediate_utility, att_immediate_utility \
                                = GameSimulation.gen_next_state_from_def_res(state, def_action, att_action,
                                                                             self.payoff_matrix,
                                                                             self.adj_matrix)
                        else:
                            next_state, def_immediate_utility, att_immediate_utility \
                                = GameSimulation.gen_next_state(state, def_action, att_action, self.payoff_matrix,
                                                                self.adj_matrix)
                        next_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                               def_action=def_action,
                                                                               att_action=att_action)

                        att_utility_average += att_immediate_utility
                        def_utility_average += def_immediate_utility
                        attacker_observation = next_observation
                        state = next_state
                        num_att -= sum(att_action).item()

                att_utility_average /= config.NUM_SAMPLE
                def_utility_average /= config.NUM_SAMPLE
                att_avg_utils.append((policy_net, att_utility_average.item(), def_utility_average.item()))

                if test is not None:
                    print('Episode %d, Attacker Utility: %.4f, Defender Utility: %.4f'
                          % (i_episode, att_utility_average.item(), def_utility_average.item()))

        best_policy_net, best_att_util, def_util = max(att_avg_utils, key=lambda x: x[1])

        return att_avg_utils  # best_policy_net, best_att_util, def_util


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free', num_res=config.NUM_RESOURCE,
                              device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
    # print(payoff_matrix)

    trained_def_A2C_GCN_model = Def_A2C_GCN(payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                            norm_adj_matrix=norm_adj_matrix, num_feature=config.NUM_FEATURE).to(device)
    # path1 = "defender_1_A2C_GCN_state_dict.pth"
    # trained_def_A2C_GCN_model.load_state_dict(torch.load(path1, map_location='cpu'))
    # trained_def_A2C_GCN_model.eval()
    # att_mixed_strategy = [Pure_Strategy('A2C-GCN', trained_att_A2C_GCN_model, 1.0)]

    trained_def_A2C_GCN_LSTM_model = Def_A2C_GCN_LSTM(payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                                      norm_adj_matrix=norm_adj_matrix, num_feature=config.NUM_FEATURE,
                                                      lstm_hidden_size=config.LSTM_HIDDEN_SIZE).to(device)
    # path2 = "defender_1_A2C_GCN_LSTM_state_dict.pth"
    # trained_def_A2C_GCN_LSTM_model.load_state_dict(torch.load(path2, map_location='cpu'))
    # trained_def_A2C_GCN_LSTM_model.eval()

    def_mixed_strategy = [Pure_Strategy('A2C-GCN-LSTM', trained_def_A2C_GCN_LSTM_model, 0.0, 'def', 0),
                          Pure_Strategy('A2C-GCN', trained_def_A2C_GCN_model, 1.0, 'def', 0),
                          Pure_Strategy('uniform', [], 0.0, 'defr', 0)]
    att_oracle = AttackerOracle(def_strategy=def_mixed_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
    new_att_set = att_oracle.train(option='A2C-GCN', test=1)
    new_att_lstm = att_oracle.train(option='A2C-GCN-LSTM', test=1)

    '''
    path1 = "attacker_2_A2C_GCN_state_dict.pth"
    path2 = "attacker_2_A2C_GCN_LSTM_state_dict.pth"
    torch.save(trained_A2C_GCN.state_dict(), path1)
    torch.save(trained_A2C_GCN_LSTM.state_dict(), path2)
    '''


if __name__ == '__main__':
    test()
    print('Complete!')
