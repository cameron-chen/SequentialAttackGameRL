from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import matplotlib.pyplot as plt

from defender_model import Def_A2C_GCN, Def_A2C_GCN_LSTM, Def_A2C_GCN_Full
from def_gan import Def_A2C_GAN
from def_act_gen import Def_Action_Generator
from utils import ReplayMemoryTransition, ReplayMemoryEpisode, GameGeneration, Transition, ReplayMemoryTransitionV
from game_simulation import GameSimulation
from optimization import Optimization
import configuration as config
from utils import Pure_Strategy
from attacker_model import Att_A2C_GCN, Att_A2C_GCN_LSTM
from defender_discriminator import DefDiscriminator
from def_act_gen import Def_Action_Generator
from distribution_estimator import DistributionEstimator
from sampling import gen_init_def_pos, gen_next_loc, gen_all_actions, gen_all_valid_actions, gen_val_mask


class DefenderOracle(object):
    def __init__(self, att_strategy, payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints, device):
        self.att_strategy = att_strategy
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.def_constraints = def_constraints
        self.num_att = config.NUM_ATTACK
        self.device = device

    def update_att_strategy(self, att_strategy):
        self.att_strategy = att_strategy

    def train(self, option, discriminator=None, act_gen=None, dist_estimator=None, policy=None, test=None, all_moves={}, nc=0, ent=0, loss=1):
        if option == 'A2C-GCN':
            optimization = Optimization(batch_size=config.BATCH_SIZE_TRANSITION, num_step=config.NUM_STEP,
                                        gamma=config.GAMMA, device=self.device,
                                        payoff_matrix=self.payoff_matrix, norm_adj_matrix=self.norm_adj_matrix)
            policy_net = Def_A2C_GCN(payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix,
                                     norm_adj_matrix=self.norm_adj_matrix, num_feature=config.NUM_FEATURE).to(self.device)
            if policy is not None:
                # Train a pre-trained model
                policy_net.load_state_dict(policy.state_dict())
            target_net = Def_A2C_GCN(payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix,
                                     norm_adj_matrix=self.norm_adj_matrix, num_feature=config.NUM_FEATURE).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            #print('Running Defender A2C-GCN...')
            def_set = self.train_A2C(policy_net, target_net, optimization=optimization, memory_size=config.MEMORY_SIZE_TRANSITION,
                                                lr=config.LR_TRANSITION, entropy_coeff=config.ENTROPY_COEFF_TRANS, test=test)

            return def_set
        if option == 'A2C-GCN-Full':
            optimization = Optimization(batch_size=config.BATCH_SIZE_TRANSITION, num_step=config.NUM_STEP,
                                        gamma=config.GAMMA, device=self.device,
                                        payoff_matrix=self.payoff_matrix, norm_adj_matrix=self.norm_adj_matrix)
            policy_net = Def_A2C_GCN_Full(payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix,
                                        norm_adj_matrix=self.norm_adj_matrix, num_feature=config.NUM_FEATURE).to(self.device)
            target_net = Def_A2C_GCN_Full(payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix,
                                        norm_adj_matrix=self.norm_adj_matrix, num_feature=config.NUM_FEATURE).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            # Calculate entire action space
            def_set = self.train_A2C_Full(policy_net, target_net, all_moves, self.def_constraints, optimization=optimization,
                                            memory_size=config.MEMORY_SIZE_TRANSITION, lr=config.LR_TRANSITION, 
                                            entropy_coeff=config.ENTROPY_COEFF_TRANS, test=test)
            return def_set
        elif option == 'A2C-GCN-LSTM':
            optimization = Optimization(batch_size=config.BATCH_SIZE_EPISODE, num_step=config.NUM_STEP,
                                        gamma=config.GAMMA, device=self.device,
                                        payoff_matrix=self.payoff_matrix, norm_adj_matrix=self.norm_adj_matrix)
            policy_net = Def_A2C_GCN_LSTM(payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix,
                                          norm_adj_matrix=self.norm_adj_matrix, num_feature=config.NUM_FEATURE,
                                          lstm_hidden_size=config.LSTM_HIDDEN_SIZE).to(self.device)
            if policy is not None:
                # Train a pre-trained model
                policy_net.load_state_dict(policy.state_dict())
            target_net = Def_A2C_GCN_LSTM(payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix,
                                          norm_adj_matrix=self.norm_adj_matrix, num_feature=config.NUM_FEATURE,
                                          lstm_hidden_size=config.LSTM_HIDDEN_SIZE).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            #print('Running Defender A2C-GCN-LSTM...')
            def_set = self.train_A2C_LSTM(policy_net, target_net, optimization=optimization,
                                            memory_size=config.MEMORY_SIZE_EPISODE, lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
                                            lr=config.LR_EPISODE, entropy_coeff=config.ENTROPY_COEFF_EPS, test=test)

            return def_set
        elif option == 'A2C-GCN-GAN':
            policy_net = Def_A2C_GAN(payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix, 
                                    norm_adj_matrix=self.norm_adj_matrix, num_feature=config.NUM_FEATURE,
                                    num_resource=config.NUM_RESOURCE, def_constraints=self.def_constraints,
                                    act_gen=act_gen, discriminator=discriminator, dist_estimator=dist_estimator, 
                                    device=self.device).to(self.device)
            if policy is not None:
                # Train a pre-trained model
                policy_net.load_state_dict(policy.state_dict())

            update_layers = {policy_net.gc1.weight, policy_net.gc1.bias, policy_net.bn.weight, policy_net.bn.bias, 
                            policy_net.gc2.weight, policy_net.gc2.bias, policy_net.ln1.weight, policy_net.ln1.bias, 
                            policy_net.ln_value1.weight, policy_net.ln_value1.bias, policy_net.ln_value2.weight, 
                            policy_net.ln_value2.bias, policy_net.act_gen.l1.weight, policy_net.act_gen.l1.bias, 
                            policy_net.act_gen.l2.weight, policy_net.act_gen.l2.bias, policy_net.act_gen.l3.weight, 
                            policy_net.act_gen.l3.bias, policy_net.act_gen.bn.weight, policy_net.act_gen.bn.bias}

            def_set = self.train_A2C_GAN(policy_net, update_layers=update_layers, lr=config.LR_EPISODE, entropy_coeff=config.ENTROPY_COEFF_EPS, 
                                        test=test, nc=nc, ent=ent, loss=loss)

            return def_set
        elif option == 'DQN-GCN':
            return
        elif option == 'DQN-GCN-LSTM':
            return
        else:
            print('Option is unavailable!')

    def train_A2C(self, policy_net, target_net, optimization, memory_size, lr, entropy_coeff, test):
        # def_mixed strategy is a tensor, each element is a tuple <type, >
        num_target = self.payoff_matrix.size(0)
        optimizer = optim.RMSprop(policy_net.parameters(), lr)
        memory = ReplayMemoryTransition(memory_size)

        init_attacker_observation = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
        init_attacker_observation[:, 0] = -1

        def_avg_utils = []

        for i_episode in range(config.NUM_EPISODE):

            # Initialize state and observation
            # This is a special case when the defender has not been assigned to any targets.
            # We just randomly assign initial locations
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
            entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])  # defender can have multiple resources at one target
            # entries = random.sample(range(0, num_target), config.NUM_RESOURCE)  # defender can only have one resource per target
            for t in range(0, len(entries)):
                init_state[entries[t], 0] += 1
            state = init_state
            attacker_observation = init_attacker_observation
            num_att = self.num_att

            # Sample a pure strategy of the attacker
            att_pure_strategy = GameSimulation.sample_pure_strategy(self.att_strategy)
            if 'LSTM' in att_pure_strategy.type and 'A2C' in att_pure_strategy.type:
                att_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)

            # Run the game through an entire episode
            # for t in range(config.NUM_STEP):
            while num_att > 0:
                if num_att < self.num_att and state[:, 0].sum() == 0:
                    def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=self.device)
                else:
                    with torch.no_grad():
                        actor, critic = policy_net(state=state.unsqueeze(0))
                        def_action = GameSimulation.sample_def_action_from_distribution(state=state, distributions=actor.squeeze(0),
                                                                                        def_constraints=self.def_constraints, device=self.device)

                # -------------------------------- Start sample attacker action --------------------------------
                if att_pure_strategy.type == 'uniform':
                    att_action = GameSimulation.sample_att_action_uniform(state=state, device=self.device)
                elif att_pure_strategy.type == 'suqr':
                    att_action = GameSimulation.sample_att_action_suqr(state=state, payoff=self.payoff_matrix, device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN':
                    att_action = \
                        GameSimulation.sample_att_action_A2C(observation=attacker_observation,
                                                trained_strategy=att_pure_strategy.trained_strategy,
                                                num_att=num_att,
                                                device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN-LSTM':
                    att_action, att_action_hidden_state, att_action_cell_state, \
                    att_value_hidden_state, att_value_cell_state = GameSimulation.sample_att_action_A2C_LSTM(
                        observation=attacker_observation, trained_strategy=att_pure_strategy.trained_strategy,
                        action_hidden_state=att_action_hidden_state, action_cell_state=att_action_cell_state,
                        value_hidden_state=att_value_hidden_state, value_cell_state=att_value_cell_state,
                        num_att=num_att, device=self.device)
                # -------------------------------- End sample defender action --------------------------------
                # print('Debugging-----')
                # print(state)
                # print(def_action)
                # print(att_action)
                # print(attacker_observation)
                # print('End debugging-----')

                next_state, def_immediate_utility, att_immediate_utility \
                    = GameSimulation.gen_next_state(state=state, def_action=def_action, att_action=att_action,
                                                    payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)
                next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                           def_action=def_action, att_action=att_action)

                # Store the transition in memory
                def_immediate_utility = torch.tensor([def_immediate_utility], device=self.device)
                memory.push(state.unsqueeze(0), def_action.unsqueeze(0), next_state.unsqueeze(0),
                            def_immediate_utility.unsqueeze(0))

                attacker_observation = next_att_observation
                num_att -= sum(att_action).item()

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                optimization.optimize_Def_A2C(memory=memory, policy_net=policy_net, target_net=target_net,
                                              optimizer=optimizer, entropy_coeff=entropy_coeff)

            # Update the target network, copying all weights and biases in DQN
            if i_episode % config.TARGET_UPDATE_TRANSITION == 0:
                target_net.load_state_dict(policy_net.state_dict())

            lr = lr * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluation only, should not be included in the runtime evaluation
            if i_episode > 0 and i_episode % 10 == 0:
                def_utility_average = 0.0
                att_utility_average = 0.0
                for i_sample in range(config.NUM_SAMPLE):
                    init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
                    entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])
                    for t in range(0, len(entries)):
                        init_state[entries[t], 0] += 1
                    state = init_state
                    attacker_observation = init_attacker_observation
                    num_att = self.num_att
                    att_pure_strategy = GameSimulation.sample_pure_strategy(self.att_strategy)
                    if 'LSTM' in att_pure_strategy.type and 'A2C' in att_pure_strategy.type:
                        att_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        att_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        att_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        att_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                    # for t in range(config.NUM_STEP):
                    while num_att > 0:
                        if num_att < self.num_att and state[:, 0].sum() == 0:
                            def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=self.device)
                        else:
                            with torch.no_grad():
                                actor, critic = policy_net(state=state.unsqueeze(0))
                                def_action = GameSimulation.sample_def_action_from_distribution(state=state,
                                                                                                distributions=actor.squeeze(0),
                                                                                                def_constraints=self.def_constraints,
                                                                                                device=self.device)

                        # -------------------------------- Start sample attacker action --------------------------------
                        if att_pure_strategy.type == 'uniform':
                            att_action = GameSimulation.sample_att_action_uniform(state=state, device=self.device)
                        elif att_pure_strategy.type == 'suqr':
                            att_action = GameSimulation.sample_att_action_suqr(state=state, payoff=self.payoff_matrix,
                                                                               device=self.device)
                        elif att_pure_strategy.type == 'A2C-GCN':
                            att_action = GameSimulation.sample_att_action_A2C(observation=attacker_observation,
                                                            trained_strategy=att_pure_strategy.trained_strategy,
                                                            num_att=num_att,
                                                            device=self.device)
                        elif att_pure_strategy.type == 'A2C-GCN-LSTM':
                            att_action, att_action_hidden_state, att_action_cell_state, \
                            att_value_hidden_state, att_value_cell_state = GameSimulation.sample_att_action_A2C_LSTM(
                                observation=attacker_observation, trained_strategy=att_pure_strategy.trained_strategy,
                                action_hidden_state=att_action_hidden_state, action_cell_state=att_action_cell_state,
                                value_hidden_state=att_value_hidden_state, value_cell_state=att_value_cell_state,
                                num_att=num_att, device=self.device)
                        # -------------------------------- End sample attacker action --------------------------------

                        next_state, def_immediate_utility, att_immediate_utility \
                            = GameSimulation.gen_next_state(state=state, def_action=def_action, att_action=att_action,
                                                            payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)
                        next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                        def_action=def_action, att_action=att_action)

                        def_utility_average += def_immediate_utility
                        att_utility_average += att_immediate_utility
                        attacker_observation = next_att_observation
                        state = next_state
                        num_att -= sum(att_action).item()

                def_utility_average /= config.NUM_SAMPLE
                att_utility_average /= config.NUM_SAMPLE

                def_avg_utils.append((policy_net, def_utility_average.item(), att_utility_average.item()))

                if test is not None:
                    print('Episode %d, Defender Utility: %.4f, Attacker Utility: %.4f'
                        % (i_episode, def_utility_average.item(), att_utility_average.item()))

        best_policy_net, best_def_util, att_util = max(def_avg_utils, key=lambda x: x[1])

        return def_avg_utils  # best_policy_net, best_def_util, att_util


    def train_A2C_Full(self, policy_net, target_net, all_moves, def_constraints, optimization, memory_size, lr, entropy_coeff, test):
        # def_mixed strategy is a tensor, each element is a tuple <type, >
        num_target = self.payoff_matrix.size(0)
        num_res = config.NUM_RESOURCE
        optimizer = optim.RMSprop(policy_net.parameters(), lr)
        memory = ReplayMemoryTransitionV(memory_size)

        init_attacker_observation = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
        init_attacker_observation[:, 0] = -1

        def_utils = []
        atk_utils = []
        def_set = []
        payoff_loss = []

        for i_episode in range(config.NUM_EPISODE):
            print("Episode", i_episode)
            # Initialize state and observation
            # This is a special case when the defender has not been assigned to any targets.
            # We just randomly assign initial locations
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
            def_init_loc = gen_init_def_pos(num_target, num_res, self.adj_matrix, self.def_constraints, threshold=1)
            for res in def_init_loc:
                init_state[(res == 1).nonzero(), 0] += int(sum(res))

            state = init_state
            def_cur_loc = def_init_loc
            attacker_observation = init_attacker_observation
            num_att = self.num_att

            # Sample a pure strategy of the attacker
            att_pure_strategy = GameSimulation.sample_pure_strategy(self.att_strategy)
            if 'LSTM' in att_pure_strategy.type and 'A2C' in att_pure_strategy.type:
                att_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)

            # Run the game through an entire episode
            # for t in range(config.NUM_STEP):
            def_total_util = 0
            atk_total_util = 0
            time_step = 1
            while num_att > 0:
                optimizer.zero_grad()
                if num_att < self.num_att and state[:, 0].sum() == 0:
                    def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=self.device)
                else:
                    # Generating full action distribution with validity mask
                    val_moves = gen_all_valid_actions(def_cur_loc, self.adj_matrix, def_constraints)
                    val_mask = gen_val_mask(all_moves, val_moves)
                    actor, critic = policy_net(state.unsqueeze(0), val_mask)
                    def_action, def_prob = GameSimulation.sample_def_action_full(actor, all_moves)

                # -------------------------------- Start sample attacker action --------------------------------
                if att_pure_strategy.type == 'uniform':
                    att_action = GameSimulation.sample_att_action_uniform(state=state, device=self.device)
                elif att_pure_strategy.type == 'suqr':
                    att_action = GameSimulation.sample_att_action_suqr(state=state, payoff=self.payoff_matrix, device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN':
                    att_action = \
                        GameSimulation.sample_att_action_A2C(observation=attacker_observation,
                                                trained_strategy=att_pure_strategy.trained_strategy,
                                                num_att=num_att,
                                                device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN-LSTM':
                    att_action, att_action_hidden_state, att_action_cell_state, \
                    att_value_hidden_state, att_value_cell_state = GameSimulation.sample_att_action_A2C_LSTM(
                        observation=attacker_observation, trained_strategy=att_pure_strategy.trained_strategy,
                        action_hidden_state=att_action_hidden_state, action_cell_state=att_action_cell_state,
                        value_hidden_state=att_value_hidden_state, value_cell_state=att_value_cell_state,
                        num_att=num_att, device=self.device)
                # -------------------------------- End sample defender action --------------------------------
                next_state, def_immediate_utility, att_immediate_utility \
                    = GameSimulation.gen_next_state_from_def_res(state=state, def_action=def_action, att_action=att_action,
                                                    payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)
                next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                           def_action=def_action, att_action=att_action)
                '''
                # Store the transition in memory
                def_immediate_utility = torch.tensor([def_immediate_utility], device=self.device)
                memory.push(state.unsqueeze(0), def_action.unsqueeze(0), next_state.unsqueeze(0),
                            def_immediate_utility.unsqueeze(0), val_mask.unsqueeze(0), def_prob)

                # Perform one step of the optimization (on the target network)
                loss = optimization.optimize_Def_A2C_Full(memory=memory, policy_net=policy_net, target_net=target_net, all_moves=all_moves,
                                                        def_constraints=self.def_constraints, optimizer=optimizer, entropy_coeff=entropy_coeff)
                '''
                _, expected_state_action_value = target_net(state.unsqueeze(0) , val_mask)
                advantage = expected_state_action_value - critic

                critic_loss = F.mse_loss(critic, expected_state_action_value)

                log_distributions = torch.log(actor + 1e-10)

                temp_distributions = log_distributions * actor
                temp_distributions = temp_distributions.sum()
                actor_loss = advantage.detach() * temp_distributions
                actor_loss = -actor_loss.mean()

                entropy_term = -(actor * torch.log(actor + 1e-10)).sum()
                loss = critic_loss + actor_loss + entropy_coeff * entropy_term
                
                print("Payoff loss:", loss.item())
                payoff_loss.append(loss.item())

                # Move to the next state
                state = next_state
                def_cur_loc = def_action
                attacker_observation = next_att_observation
                num_att -= sum(att_action).item()
                time_step += 1

                def_total_util += def_immediate_utility.item()
                atk_total_util += att_immediate_utility.item()

            def_utils.append(def_total_util)
            atk_utils.append(atk_total_util)
            
            # Update the target network, copying all weights and biases in DQN
            if i_episode % config.TARGET_UPDATE_TRANSITION == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if i_episode > 0 and i_episode % 10 == 0:
                def_set.append((policy_net, def_utils[-1], atk_utils[-1]))

            print("Defender Utility:", def_utils[-1])
            print("Attacker Utility:", atk_utils[-1], "\n")

            lr = lr * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return def_set, def_utils, atk_utils, payoff_loss


    def train_A2C_LSTM(self, policy_net, target_net, optimization, memory_size, lstm_hidden_size, lr, entropy_coeff, test):
        # print('Running GCN-LSTM...')
        num_target = self.payoff_matrix.size(0)
        optimizer = optim.RMSprop(policy_net.parameters(), lr)
        memory = ReplayMemoryEpisode(memory_size)

        init_action_hidden_state = torch.zeros(1, lstm_hidden_size, device=self.device)
        init_action_cell_state = torch.zeros(1, lstm_hidden_size, device=self.device)
        init_value_hidden_state = torch.zeros(1, lstm_hidden_size, device=self.device)
        init_value_cell_state = torch.zeros(1, lstm_hidden_size, device=self.device)

        init_attacker_observation = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
        init_attacker_observation[:, 0] = -1

        def_avg_utils = []

        for i_episode in range(config.NUM_EPISODE):
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
            entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])  # defender can have multiple resources at one target
            # entries = random.sample(range(0, num_target), config.NUM_RESOURCE)  # defender can only have one resource per target
            for t in range(0, len(entries)):
                init_state[entries[t], 0] += 1

            state = init_state
            attacker_observation = init_attacker_observation
            num_att = self.num_att

            action_hidden_state = init_action_hidden_state
            action_cell_state = init_action_cell_state
            value_hidden_state = init_value_hidden_state
            value_cell_state = init_value_cell_state

            temp_episode = []
            att_pure_strategy = GameSimulation.sample_pure_strategy(mixed_strategy=self.att_strategy)
            if 'LSTM' in att_pure_strategy.type and 'A2C' in att_pure_strategy.type:
                att_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
            # for t in range(config.NUM_STEP):
            while num_att > 0:

                if num_att < self.num_att and state[:, 0].sum() == 0:
                    def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=self.device)
                else:
                    with torch.no_grad():
                        actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                            = policy_net(state=state.unsqueeze(0), action_hidden_state=action_hidden_state,
                                         action_cell_state=action_cell_state,
                                         value_hidden_state=value_hidden_state, value_cell_state=value_cell_state)
                    def_action = GameSimulation.sample_def_action_from_distribution(state= state, distributions=actor.squeeze(0),
                                                                                    def_constraints=self.def_constraints, device=self.device)

                # -------------------------------- Start sample attacker action --------------------------------
                if att_pure_strategy.type == 'uniform':
                    att_action = GameSimulation.sample_att_action_uniform(state=state, device=self.device)
                elif att_pure_strategy.type == 'suqr':
                    att_action = GameSimulation.sample_att_action_suqr(state=state, payoff=self.payoff_matrix,
                                                                       device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN':
                    att_action = GameSimulation.sample_att_action_A2C(observation=attacker_observation,
                                                                      trained_strategy=att_pure_strategy.trained_strategy,
                                                                      num_att=num_att,
                                                                      device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN-LSTM':
                    att_action, att_action_hidden_state, att_action_cell_state, \
                    att_value_hidden_state, att_value_cell_state = GameSimulation.sample_att_action_A2C_LSTM(
                        observation=attacker_observation, trained_strategy=att_pure_strategy.trained_strategy,
                        action_hidden_state=att_action_hidden_state, action_cell_state=att_action_cell_state,
                        value_hidden_state=att_value_hidden_state, value_cell_state=att_value_cell_state,
                        num_att=num_att, device=self.device)
                # -------------------------------- End sample attacker action --------------------------------
                # print('Debugging-----')
                # print(state)
                # print(def_action)
                # print(att_action)
                # print(attacker_observation)
                # print('End debugging-----')

                next_state, def_immediate_utility, att_immediate_utility \
                    = GameSimulation.gen_next_state(state=state, def_action=def_action, att_action=att_action,
                                                    payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)
                next_att_observation = GameSimulation.gen_next_observation(attacker_observation, def_action,
                                                                           att_action)

                attacker_observation = next_att_observation
                num_att -= sum(att_action).item()

                temp_transition = Transition(state.unsqueeze(0), def_action.unsqueeze(0)
                                             , next_state.unsqueeze(0), def_immediate_utility.unsqueeze(0))
                temp_episode.append(temp_transition)

                # Move to the next state
                state = next_state

            memory.add_episode(temp_episode)
            # Update the target network, copying all weights and biases in DQN
            if i_episode % config.TARGET_UPDATE_EPISODE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if i_episode % config.OPTIMIZE_UPDATE_EPISODE == 0:
                # Perform one step of the optimization (on the target network)
                optimization.optimize_Def_A2C_LSTM(memory=memory, policy_net=policy_net, target_net=target_net,
                                                   optimizer=optimizer,
                                                   lstm_hidden_size=lstm_hidden_size, entropy_coeff=entropy_coeff)

            lr = lr * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluation
            if i_episode > 0 and i_episode % 10 == 0:
                def_utility_average = 0.0
                att_utility_average = 0.0
                for i_sample in range(config.NUM_SAMPLE):
                    init_action_hidden_state = torch.zeros(1, lstm_hidden_size, device=self.device)
                    init_action_cell_state = torch.zeros(1, lstm_hidden_size, device=self.device)
                    init_value_hidden_state = torch.zeros(1, lstm_hidden_size, device=self.device)
                    init_value_cell_state = torch.zeros(1, lstm_hidden_size, device=self.device)

                    init_attacker_observation = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
                    init_attacker_observation[:, 0] = -1

                    init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
                    entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])
                    for t in range(0, len(entries)):
                        init_state[entries[t], 0] += 1
                    state = init_state
                    attacker_observation = init_attacker_observation
                    num_att = self.num_att

                    action_hidden_state = init_action_hidden_state
                    action_cell_state = init_action_cell_state
                    value_hidden_state = init_value_hidden_state
                    value_cell_state = init_value_cell_state

                    att_pure_strategy = GameSimulation.sample_pure_strategy(mixed_strategy=self.att_strategy)
                    if 'LSTM' in att_pure_strategy.type and 'A2C' in att_pure_strategy.type:
                        att_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        att_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        att_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                        att_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)

                    # for t in range(config.NUM_STEP):
                    while num_att > 0:

                        if num_att < self.num_att and state[:, 0].sum() == 0:
                            def_action = torch.zeros(num_target, num_target, dtype=torch.float32,
                                                     device=self.device)
                        else:
                            with torch.no_grad():
                                actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                                    = policy_net(state=state.unsqueeze(0),
                                                 action_hidden_state=action_hidden_state, action_cell_state=action_cell_state,
                                                 value_hidden_state=value_hidden_state, value_cell_state=value_cell_state)
                            def_action = GameSimulation.sample_def_action_from_distribution(state=state,
                                                                                            distributions=actor.squeeze(0),
                                                                                            def_constraints=self.def_constraints,
                                                                                            device=self.device)

                        # -------------------------------- Start sample attacker action --------------------------------
                        if att_pure_strategy.type == 'uniform':
                            att_action = GameSimulation.sample_att_action_uniform(state=state, device=self.device)
                        elif att_pure_strategy.type == 'suqr':
                            att_action = GameSimulation.sample_att_action_suqr(state=state, payoff=self.payoff_matrix,
                                                                               device=self.device)
                        elif att_pure_strategy.type == 'A2C-GCN':
                            att_action = GameSimulation.sample_att_action_A2C(observation=attacker_observation,
                                                            trained_strategy=att_pure_strategy.trained_strategy,
                                                            num_att=num_att,
                                                            device=self.device)
                        elif att_pure_strategy.type == 'A2C-GCN-LSTM':
                            att_action, att_action_hidden_state, att_action_cell_state, \
                            att_value_hidden_state, att_value_cell_state = GameSimulation.sample_att_action_A2C_LSTM(
                                observation=attacker_observation, trained_strategy=att_pure_strategy.trained_strategy,
                                action_hidden_state=att_action_hidden_state, action_cell_state=att_action_cell_state,
                                value_hidden_state=att_value_hidden_state, value_cell_state=att_value_cell_state,
                                num_att=num_att, device=self.device)
                        # -------------------------------- End sample attacker action --------------------------------

                        next_state, def_immediate_utility, att_immediate_utility \
                            = GameSimulation.gen_next_state(state=state, def_action=def_action, att_action=att_action,
                                                            payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)
                        next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                                   def_action=def_action,
                                                                                   att_action=att_action)

                        attacker_observation = next_att_observation
                        num_att -= sum(att_action).item()

                        def_utility_average += def_immediate_utility
                        att_utility_average += att_immediate_utility
                        state = next_state

                def_utility_average /= config.NUM_SAMPLE
                att_utility_average /= config.NUM_SAMPLE
                def_avg_utils.append((policy_net, def_utility_average.item(), att_utility_average.item()))

                if test is not None:
                    print('Episode %d, Defender Utility: %.4f, Attacker Utility: %.4f'
                        % (i_episode, def_utility_average.item(), att_utility_average.item()))

        best_policy_net, best_def_util, att_util = max(def_avg_utils, key=lambda x:x[1])

        return def_avg_utils  # best_policy_net, best_def_util, att_util

    def train_A2C_GAN(self, policy_net, update_layers, lr, entropy_coeff, test=None, nc=0, ent=0, loss=1):
        # def_mixed strategy is a tensor, each element is a tuple <type, >
        # torch.autograd.set_detect_anomaly(True)
        num_target = self.payoff_matrix.size(0)
        num_res = config.NUM_RESOURCE
        optimizer = optim.Adam(update_layers, lr)

        init_attacker_observation = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
        init_attacker_observation[:, 0] = -1

        def_utils = []
        atk_utils = []
        def_set = []            # store tuples of (policy, def_util, atk_util)
        payoff_loss = []        # loss from actor-critic
        attempts_list = []     # number of attempts per episode to generate a valid defender action
        num_acts_list = []     # number of unique actions generated per episode

        for i_episode in range(config.NUM_EPISODE):
            if test: print("\nEpisode", i_episode)
            # Initialize state and observation
            # Defender is assigned to locations according to defender constraints
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
            def_init_loc = gen_init_def_pos(num_target, num_res, self.adj_matrix, self.def_constraints, threshold=1)
            for res in def_init_loc:
                init_state[(res == 1).nonzero(), 0] += int(sum(res))

            state = init_state              # Defender resources are not differentiated in 'state'
            def_cur_loc = def_init_loc      # Defender resources are differentiated in 'def_cur_loc'
            attacker_observation = init_attacker_observation
            num_att = self.num_att

            # Sample a pure strategy of the attacker
            att_pure_strategy = GameSimulation.sample_pure_strategy(self.att_strategy)
            if 'LSTM' in att_pure_strategy.type and 'A2C' in att_pure_strategy.type:
                att_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)

            # act_gen = Def_Action_Generator(config.NUM_TARGET, config.NUM_RESOURCE, self.device)

            # Run the game through an entire episode
            def_total_util = 0.0
            atk_total_util = 0.0
            f_def_total = torch.tensor(0.0)
            time_step = 1
            while num_att > 0:
                if num_att < self.num_att and state[:, 0].sum() == 0:
                    def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=self.device)
                else:
                    optimizer.zero_grad()
                    print("\nGenerating Defender action...", time_step)
                    total_attempts = 0
                    if test:
                        def_action, critic, prob, attempts, num_actions = policy_net(state.unsqueeze(0), def_cur_loc, test=test, nc=nc, ent=ent)
                    else:
                        def_action, critic, prob = policy_net(state.unsqueeze(0), def_cur_loc, test=test, nc=nc, ent=ent)
                    redo = 1
                    total_attempts += attempts
                    while not torch.is_tensor(def_action):
                        if test: 
                            print("Time step", time_step, ": redo", redo)
                            def_action, critic, prob, attempts, num_actions = policy_net(state.unsqueeze(0), def_cur_loc, test=test, nc=nc, ent=ent)
                        else:
                            def_action, critic, prob = policy_net(state.unsqueeze(0), def_cur_loc, test=test, nc=nc, ent=ent)
                        redo += 1
                        total_attempts += attempts
                        if total_attempts >= 100:
                            print("Attempts > 100: Using random valid defender action.")
                            def_action = gen_next_loc(def_cur_loc, self.adj_matrix, self.def_constraints)

                    attempts_list.append(total_attempts)
                    num_acts_list.append(num_actions)

                # -------------------------------- Start sample attacker action --------------------------------
                if att_pure_strategy.type == 'uniform':
                    att_action = GameSimulation.sample_att_action_uniform(state=state, device=self.device)
                elif att_pure_strategy.type == 'suqr':
                    att_action = GameSimulation.sample_att_action_suqr(state=state, payoff=self.payoff_matrix, device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN':
                    att_action = \
                        GameSimulation.sample_att_action_A2C(observation=attacker_observation,
                                                trained_strategy=att_pure_strategy.trained_strategy,
                                                num_att=num_att,
                                                device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN-LSTM':
                    att_action, att_action_hidden_state, att_action_cell_state, \
                    att_value_hidden_state, att_value_cell_state = GameSimulation.sample_att_action_A2C_LSTM(
                        observation=attacker_observation, trained_strategy=att_pure_strategy.trained_strategy,
                        action_hidden_state=att_action_hidden_state, action_cell_state=att_action_cell_state,
                        value_hidden_state=att_value_hidden_state, value_cell_state=att_value_cell_state,
                        num_att=num_att, device=self.device)
                # -------------------------------- End sample attacker action --------------------------------

                next_state, def_immediate_utility, att_immediate_utility \
                    = GameSimulation.gen_next_state_from_def_res(state=state, def_action=def_action, att_action=att_action,
                                                                    payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)

                next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                            def_action=def_action, att_action=att_action)

                if test:
                    print("Target Attacked:", (att_action == 1).nonzero().item())
                    print("Defender Action:", def_action)
                    print("Def Util:", def_immediate_utility.item(), "-- Atk Util:", att_immediate_utility.item())

                # Perform one step of the optimization -- FIGURE OUT LOSS THAT INCLUDES ESTIMATED PROBABILITY (prob)
                critic_loss = F.mse_loss(critic.squeeze(), def_immediate_utility)
                advantage = def_immediate_utility - critic
                actor_loss = advantage*prob.detach()
                pol_act = prob.detach()*def_action
                entropy_term = -(pol_act * torch.log(pol_act + 1e-10)).sum()
                loss = critic_loss + actor_loss + entropy_coeff * entropy_term

                if test: print(time_step, "Payoff loss:", loss.item())
                payoff_loss.append(loss.item())

                if loss:
                    loss.backward()
                    optimizer.step()

                # Move to the next state
                state = next_state
                def_cur_loc = def_action
                attacker_observation = next_att_observation
                num_att -= sum(att_action).item()

                time_step += 1
                def_total_util += def_immediate_utility
                atk_total_util += att_immediate_utility

            def_utils.append(def_total_util.item())
            atk_utils.append(atk_total_util.item())

            if i_episode > 0 and i_episode % 10 == 0:
                print(i_episode, "episodes run")
                def_set.append((policy_net, def_utils[-1], atk_utils[-1]))

            if test:                    
                print("\nDefender Utility:", def_utils[-1])
                print("Attacker Utility:", atk_utils[-1])

            lr = lr * 0.95
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if test:
            return def_set, def_utils, atk_utils, attempts_list, num_acts_list
        else:
            return def_set

    def compare_a2c_gan(self, a2c, gan, all_moves, test=1):
        num_target = self.payoff_matrix.size(0)
        num_res = config.NUM_RESOURCE

        init_attacker_observation = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=self.device)
        init_attacker_observation[:, 0] = -1

        def_utils = []
        atk_utils = []
        attempts_list = []     # number of attempts per episode to generate a valid defender action
        num_acts_list = []     # number of unique actions generated per episode
        f_prob_list = []
        d_prob_list = []
        f_def_utils = []

        for i_episode in range(config.NUM_EPISODE):
            if test: print("\nEpisode", i_episode)
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=self.device)
            def_init_loc = gen_init_def_pos(num_target, num_res, self.adj_matrix, self.def_constraints, threshold=1)
            for res in def_init_loc:
                init_state[(res == 1).nonzero(), 0] += int(sum(res))

            state = init_state              # Defender resources are not differentiated in 'state'
            def_cur_loc = def_init_loc      # Defender resources are differentiated in 'def_cur_loc'
            attacker_observation = init_attacker_observation
            num_att = self.num_att

            # Sample a pure strategy of the attacker
            att_pure_strategy = GameSimulation.sample_pure_strategy(self.att_strategy)
            if 'LSTM' in att_pure_strategy.type and 'A2C' in att_pure_strategy.type:
                att_action_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_action_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_hidden_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)
                att_value_cell_state = torch.zeros(1, config.LSTM_HIDDEN_SIZE, device=self.device)

            # Run the game through an entire episode
            def_total_util = 0.0
            atk_total_util = 0.0
            f_def_total = torch.tensor(0.0)
            time_step = 1
            while num_att > 0:
                if num_att < self.num_att and state[:, 0].sum() == 0:
                    def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=self.device)
                else:
                    print("\nGenerating GAN action...", time_step)
                    total_attempts = 0
                    if test:
                        def_action, critic, prob, attempts, num_actions = gan(state.unsqueeze(0), def_cur_loc, test=test)
                    else:
                        def_action, critic, prob = gan(state.unsqueeze(0), def_cur_loc, test=test)
                    redo = 1
                    total_attempts += attempts
                    while not torch.is_tensor(def_action):
                        if test: 
                            print("Time step", time_step, ": redo", redo)
                            def_action, critic, prob, attempts, num_actions = gan(state.unsqueeze(0), def_cur_loc, test=test)
                        else:
                            def_action, critic, prob = gan(state.unsqueeze(0), def_cur_loc, test=test)
                        redo += 1
                        total_attempts += attempts
                        if total_attempts >= 100:
                            print("Attempts > 100: Using random valid defender action.")
                            def_action = gen_next_loc(def_cur_loc, self.adj_matrix, self.def_constraints)

                    attempts_list.append(total_attempts)
                    num_acts_list.append(num_actions)

                # -------------------------------- Start sample attacker action --------------------------------
                if att_pure_strategy.type == 'uniform':
                    att_action = GameSimulation.sample_att_action_uniform(state=state, device=self.device)
                elif att_pure_strategy.type == 'suqr':
                    att_action = GameSimulation.sample_att_action_suqr(state=state, payoff=self.payoff_matrix, device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN':
                    att_action = \
                        GameSimulation.sample_att_action_A2C(observation=attacker_observation,
                                                trained_strategy=att_pure_strategy.trained_strategy,
                                                num_att=num_att,
                                                device=self.device)
                elif att_pure_strategy.type == 'A2C-GCN-LSTM':
                    att_action, att_action_hidden_state, att_action_cell_state, \
                    att_value_hidden_state, att_value_cell_state = GameSimulation.sample_att_action_A2C_LSTM(
                        observation=attacker_observation, trained_strategy=att_pure_strategy.trained_strategy,
                        action_hidden_state=att_action_hidden_state, action_cell_state=att_action_cell_state,
                        value_hidden_state=att_value_hidden_state, value_cell_state=att_value_cell_state,
                        num_att=num_att, device=self.device)
                # -------------------------------- End sample attacker action --------------------------------

                next_state, def_immediate_utility, att_immediate_utility \
                    = GameSimulation.gen_next_state_from_def_res(state=state, def_action=def_action, att_action=att_action,
                                                                    payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)

                next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                            def_action=def_action, att_action=att_action)

                print("Target Attacked:", (att_action == 1).nonzero().item())
                print("Defender Action:", def_action)
                print("Def Util:", def_immediate_utility.item(), "-- Atk Util:", att_immediate_utility.item())

                # Generating A2C action and probabilities
                val_moves = gen_all_valid_actions(def_cur_loc, self.adj_matrix, self.def_constraints)
                val_mask = gen_val_mask(all_moves, val_moves)
                f_actor, f_critic = a2c(state.unsqueeze(0), val_mask)

                gan_move = tuple([(res == 1).nonzero().item() for res in def_action])
                print("\nGAN move:", gan_move)
                print("GAN move probability:", f_actor[all_moves.index(gan_move)].item())
                d_prob_list.append(f_actor[all_moves.index(gan_move)].item())

                print("Best move:", all_moves[torch.argmax(f_actor)])
                print("Move probability:", max(f_actor).item())
                f_prob_list.append(max(f_actor).item())

                f_def_action, f_def_prob = GameSimulation.sample_def_action_full(f_actor, all_moves)
                _, f_def_util, f_att_util \
                    = GameSimulation.gen_next_state_from_def_res(state=state, def_action=f_def_action, att_action=att_action,
                                                                payoff_matrix=self.payoff_matrix, adj_matrix=self.adj_matrix)

                print("A2C Def Util:", f_def_util.item())
                f_def_total += f_def_util

                # Move to the next state
                state = next_state
                def_cur_loc = def_action
                attacker_observation = next_att_observation
                num_att -= sum(att_action).item()

                time_step += 1
                def_total_util += def_immediate_utility
                atk_total_util += att_immediate_utility

            def_utils.append(def_total_util.item())
            atk_utils.append(atk_total_util.item())
            f_def_utils.append(f_def_total.item())
              
            print("\nDefender Utility:", def_utils[-1])
            print("Attacker Utility:", atk_utils[-1])

        return def_utils, atk_utils, attempts_list, num_acts_list, d_prob_list, f_prob_list, f_def_utils


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free', num_res=config.NUM_RESOURCE, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
    # def_constraints = [[1, 3], [0, 2], [4]]
    print(def_constraints)

    trained_att_A2C_GCN_model = Att_A2C_GCN(payoff_matrix, norm_adj_matrix, config.NUM_FEATURE).to(device)
    #path1 = "attacker_2_A2C_GCN_state_dict.pth"
    #trained_att_A2C_GCN_model.load_state_dict(torch.load(path1, map_location='cpu'))
    #trained_att_A2C_GCN_model.eval()
    # att_mixed_strategy = [Pure_Strategy('A2C-GCN', trained_att_A2C_GCN_model, 1.0)]

    trained_att_A2C_GCN_LSTM_model = Att_A2C_GCN_LSTM(payoff_matrix, norm_adj_matrix, config.NUM_FEATURE, config.LSTM_HIDDEN_SIZE).to(device)
    #path2 = "attacker_2_A2C_GCN_LSTM_state_dict.pth"
    #trained_att_A2C_GCN_LSTM_model.load_state_dict(torch.load(path2, map_location='cpu'))
    #trained_att_A2C_GCN_LSTM_model.eval()

    att_mixed_strategy = [Pure_Strategy('A2C-GCN-LSTM', trained_att_A2C_GCN_LSTM_model, 0.25, 'att0', 0),
                          Pure_Strategy('A2C-GCN', trained_att_A2C_GCN_model, 0.25, 'att1', 0),
                          Pure_Strategy('uniform', [], 0.25, 'att2', 0),
                          Pure_Strategy('suqr', [], 0.25, 'att3', 0)]

    '''
    print("Training A2C-GCN")
    new_def_set = def_oracle.train(option='A2C-GCN', test=1)
    print("\nTraining A2C-GCN-LSTM")
    new_def_lstm = def_oracle.train(option='A2C-GCN-LSTM', test=1)
    '''
    
    print("\nTraining A2C Defender Oracle with Full Action Space")
    def_oracle_a2c = DefenderOracle(att_strategy=att_mixed_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
    all_moves = gen_all_actions(config.NUM_TARGET, config.NUM_RESOURCE)
    new_def, def_utils, atk_utils, _ = def_oracle_a2c.train(option='A2C-GCN-Full', all_moves=all_moves)
    def_a2c = new_def[-1][0]
    
    print("\nTraining Defender Discriminator")
    disc_obj = DefDiscriminator(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, norm_adj_matrix,
                                def_constraints, device, threshold=1)
    discriminator = disc_obj.train(episodes=1600)               # do episode=1600 for 3 resource game
    disc = discriminator
    # discriminator = disc_obj.initial()

    print("\nTraining Distribution Estimator")
    dist_estim_obj = DistributionEstimator(config.NUM_TARGET, config.NUM_RESOURCE, config.NUM_FEATURE, payoff_matrix,
                                            adj_matrix, norm_adj_matrix, def_constraints, device)
    # dist_estimator = dist_estim_obj.train()
    dist_estimator = dist_estim_obj.initial()

    print("\nTraining A2C-GCN-GAN-Generator")
    def_oracle_gan = DefenderOracle(att_strategy=att_mixed_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
    act_gen = Def_Action_Generator(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, device).to(device)
    def_gan_list, def_utils, atk_utils, attempts_list, num_acts_list \
        = def_oracle_gan.train(option='A2C-GCN-GAN', discriminator=discriminator, act_gen=act_gen, dist_estimator=dist_estimator, test=1)
    def_gan = def_gan_list[-1][0]
    '''
    print("\nTraining A2C-GCN-GAN-Generator with No Constraints")
    def_gan_nc = DefenderOracle(att_strategy=att_mixed_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
    act_gen = Def_Action_Generator(config.NUM_TARGET, config.NUM_RESOURCE, device).to(device)
    dist_estimator = dist_estim_obj.initial()
    discriminator = disc
    def_gan_list, def_utils_nc, atk_utils, attempts_nc, num_acts_nc \
        = def_gan_nc.train(option='A2C-GCN-GAN', discriminator=discriminator, act_gen=act_gen, dist_estimator=dist_estimator, test=1, nc=1, ent=0)

    print("\nTraining A2C-GCN-GAN-Generator with Entropy")
    def_gan_ent = DefenderOracle(att_strategy=att_mixed_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
    act_gen = Def_Action_Generator(config.NUM_TARGET, config.NUM_RESOURCE, device).to(device)
    dist_estimator = dist_estim_obj.initial()
    discriminator = disc
    def_gan_list, def_utils_ent, atk_utils, attempts_ent, num_acts_ent \
        = def_gan_ent.train(option='A2C-GCN-GAN', discriminator=discriminator, act_gen=act_gen, dist_estimator=dist_estimator, test=1, nc=0, ent=1)
    
    print("\nTraining A2C-GCN-GAN-Generator with No RL Loss")
    def_gan_nc = DefenderOracle(att_strategy=att_mixed_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
    act_gen = Def_Action_Generator(config.NUM_TARGET, config.NUM_RESOURCE, device).to(device)
    dist_estimator = dist_estim_obj.initial()
    discriminator = disc
    def_gan_list, def_utils_nrl, atk_utils, attempts_nrl, num_acts_nrl \
        = def_gan_nc.train(option='A2C-GCN-GAN', discriminator=discriminator, act_gen=act_gen, dist_estimator=dist_estimator, test=1, nc=0, ent=1, loss=0)
    '''
    
    def_utils, atk_utils, attempts_list, num_acts_list, d_prob_list, f_prob_list, f_def_utils \
         = def_oracle_gan.compare_a2c_gan(def_a2c, def_gan, all_moves)
    
    print("Average A2C utility:", sum(f_def_utils)/len(f_def_utils))
    print("Average GAN utility:", sum(def_utils)/len(def_utils))
    print(round(((time.time() - start) / 60), 4), 'min')
    
    plt.figure(figsize=(20, 10))
    plt.title("Defender/Attacker Utilities")
    plt.xlabel("Episode")
    plt.ylabel("Utility")
    plt.plot(def_utils, label="Defender Utility")
    # plt.plot(def_utils_nc, label="Defender Utility (no constraints)")
    # plt.plot(def_utils_ent, label="Defender Utility (with entropy)")
    # plt.plot(def_utils_nrl, label="Defender Utility (no RL loss)")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(20, 10))
    plt.title("Defender # of Tries for Valid Action")
    plt.xlabel("Time Step")
    plt.ylabel("# of Tries")
    plt.plot(attempts_list, label="# of Tries")
    # plt.plot(attempts_nc, label="no constraints")
    # plt.plot(attempts_ent, label="with entropy")
    # plt.plot(attempts_nrl, label="no RL loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title("Defender # of Unique Valid Actions")
    plt.xlabel("Time Step")
    plt.ylabel("# of Unique Actions")
    plt.plot(num_acts_list, label="# of Unique Actions")
    # plt.plot(num_acts_nc, label="no constraints")
    # plt.plot(num_acts_ent, label="with entropy")
    # plt.plot(num_acts_nrl, label="no RL loss")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(20, 10))
    plt.title("Utilities between Full Action A2C and A2C-GAN")
    plt.xlabel("Episode")
    plt.ylabel("Utility")
    plt.plot(def_utils, label="A2C GAN")
    plt.plot(f_def_utils, label="Full Action A2C")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(20, 10))
    plt.title("Probabilities between Full Action A2C and A2C-GAN")
    plt.xlabel("Time Step")
    plt.ylabel("Probability")
    plt.plot(f_prob_list, label="Max probability in A2C distribution")
    plt.plot(d_prob_list, label="GAN action probability in A2C distribution")
    plt.legend()
    plt.show()
    
    '''
    # Save trained model
    path1 = "defender_2_A2C_GCN_state_dict.pth"
    path2 = "defender_2_A2C_GCN_LSTM_state_dict.pth"
    torch.save(trained_A2C_GCN.state_dict(), path1)
    torch.save(trained_A2C_GCN_LSTM.state_dict(), path2)
    '''


if __name__ == '__main__':
    test()
