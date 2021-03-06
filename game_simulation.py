from __future__ import print_function

import torch
import math
import random

import configuration as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_suqr_probs(payoff):
    feats = []
    for target in payoff:
        s_util = 0
        for feature in target:
            s_util += math.exp(random.uniform(0, 1) * feature)
        feats.append(s_util)
    return [f/sum(feats) for f in feats]


class GameSimulation(object):
    @staticmethod
    def gen_next_state(state, def_action, att_action, payoff_matrix, adj_matrix):
        num_target = payoff_matrix.size(0)
        next_state = state.clone()
        def_immediate_utility = 0.0
        att_immediate_utility = 0.0

        next_state[:, 0] = def_action.sum(dim=0)

        protected = False
        for t in range(num_target):
            if att_action[t] == 1:
                # print(att_action[t])
                next_state[t, 1] = 1
                next_state[t, 0] = 0
                for tprime in range(num_target):
                    if def_action[tprime, t] >= 1:
                        def_immediate_utility = payoff_matrix[t, 0].clone()
                        att_immediate_utility = payoff_matrix[t, 3].clone()
                        protected = True
                        # break
                    else:
                        protected = False
                if not protected:
                    def_immediate_utility = payoff_matrix[t, 1].clone()
                    att_immediate_utility = payoff_matrix[t, 2].clone()
                # break

        # Defender resource movement cost
        is_start = state[:, 1].sum()
        for t in range(num_target):
            for tprime in range(num_target):
                if def_action[tprime, t] >= 1 and is_start > 0:
                    def_immediate_utility = def_immediate_utility + (adj_matrix[tprime, t] * def_action[tprime, t])

        return next_state, def_immediate_utility, att_immediate_utility

    @staticmethod
    def gen_next_observation(observation, def_action, att_action):
        next_observation = observation.clone()
        attack_idx = torch.nonzero(att_action).squeeze(1)  # .item()
        for idx in attack_idx:
            next_observation[idx, 1] = 1
            next_observation[idx, 0] = def_action[:, idx].sum()
        return next_observation

    @staticmethod
    def sample_pure_strategy(mixed_strategy):
        idx = 0
        pivot = torch.rand(1)
        init_prob = mixed_strategy[0].probability
        while pivot > init_prob:
            # if 0 < init_prob < 1:
            #     print("Checking!!!")
            idx += 1
            if idx >= len(mixed_strategy):
                idx -= 1
                break
            init_prob = init_prob + mixed_strategy[idx].probability

        return mixed_strategy[idx]

    @staticmethod
    def sample_att_action_from_distribution(distribution, num_att, device):  # distribution is of size num_target
        num_target = distribution.size(0)
        non_zero_prob_idx = torch.nonzero(distribution > 1e-4).squeeze(1)
        idx = 0
        count = 1  # make count = num_att and config.NUM_STEP = 1 for multiple attacks per time step
        att_action = torch.zeros(num_target, dtype=torch.float32, device=device)
        while count > 0:
            pivot = torch.rand(1)
            init_prob = distribution[non_zero_prob_idx[0]].clone()
            while pivot > init_prob:
                idx += 1
                if idx >= len(non_zero_prob_idx):
                    idx -= 1
                    break
                init_prob += distribution[non_zero_prob_idx[idx]]
            att_action[non_zero_prob_idx[idx].item()] = 1
            count -= 1
        return att_action

    @staticmethod
    def sample_def_action_from_distribution(state, distributions, def_constraints, device):
        num_target = state.size(0)
        def_current_location = torch.nonzero(state[:, 0])
        valid_action = False
        while not valid_action:
            def_action = torch.zeros(num_target, num_target).to(device)
            for target in def_current_location:
                probs = distributions[target.item()]

                non_zero_prob_idx = torch.nonzero(probs > 1e-4).squeeze(1)
                count = state[target, 0].item()
                while count > 0:
                    pivot = torch.rand(1)
                    init_prob = probs[non_zero_prob_idx[0]].clone()

                    idx = 0
                    while pivot > init_prob:
                        idx += 1

                        if idx >= len(non_zero_prob_idx):
                            idx -= 1
                            break
                        init_prob += probs[non_zero_prob_idx[idx]]

                    def_action[target, non_zero_prob_idx[idx]] += 1
                    count = count - 1
            # valid_action = check_def_constraints(def_action, def_constraints, state)  # for implementing defender constraints
            valid_action = True
            if not valid_action:
                print("Invalid Defender Move.")
        # print('Def action final:', def_action)
        return def_action
    
    @staticmethod
    def sample_def_action_full(distributions, all_moves):
        def_action = torch.zeros(config.NUM_RESOURCE, config.NUM_TARGET).to(device)
        pivot = torch.rand(1)
        idx = 0
        prob = distributions[idx]
        while pivot > prob:
            idx += 1
            prob += distributions[idx]
        def_code = all_moves[idx]

        for i,pos in enumerate(def_code):
            def_action[i][pos] = 1

        return def_action, distributions[idx]

    @staticmethod
    def sample_def_action_A2C(num_attack_remain, state, trained_strategy, def_constraints, device):
        num_target = state.size(0)
        if num_attack_remain < config.NUM_ATTACK and state[:, 0].sum() == 0:
            return torch.zeros(num_target, num_target, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                actor, critic = trained_strategy(state.unsqueeze(0))
            return GameSimulation.sample_def_action_from_distribution(state, actor.squeeze(0), def_constraints, device)

    @staticmethod
    def sample_def_action_A2C_LSTM(num_attack_remain, state, trained_strategy,
                                   action_hidden_state, action_cell_state, value_hidden_state, value_cell_state,
                                   def_constraints, device):
        num_target = state.size(0)
        if num_attack_remain < config.NUM_ATTACK and state[:, 0].sum() == 0:
            return torch.zeros(num_target, num_target, dtype=torch.float32, device=device), [], [], [], []
        else:
            with torch.no_grad():
                actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                    = trained_strategy(state.unsqueeze(0),
                                       action_hidden_state, action_cell_state, value_hidden_state, value_cell_state)
                def_action = GameSimulation.sample_def_action_from_distribution(state, actor.squeeze(0),
                                                                                def_constraints, device)
                return def_action, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state

    @staticmethod
    def sample_def_action_GAN(step, state, trained_strategy, device):
        num_target = state.size(0)
        if step > 0 and state[:, 0].sum() == 0:
            return torch.zeros(num_target, num_target, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                actor, critic = trained_strategy(state.unsqueeze(0))
            return GameSimulation.sample_def_action_from_res_dist(state, actor.squeeze(0), device)

    @staticmethod
    def sample_def_action_uniform(state, adj_matrix, device):
        num_target = adj_matrix.size(0)
        def_action = torch.zeros(num_target, num_target).to(device)

        # count = state[:, 1].sum()
        # if count == 0:  # the beginning of the game
        #     num_defender = num_resource
        #     while num_defender > 0:
        #         num_defender -= 1
        #         idx = torch.randint(0, num_target, (1,)).item()
        #         def_action[idx, idx] += 1
        #     return def_action

        def_location = torch.nonzero(state[:, 0])
        temp_adj_matrix = torch.where(adj_matrix == config.MIN_VALUE, 0, 1)
        for target in def_location:
            num_defender = state[target.item(), 0].item()
            neighbors = torch.nonzero(temp_adj_matrix[target.item(), :]).squeeze(1)
            while num_defender > 0:
                idx = torch.randint(0, len(neighbors), (1,)).item()
                new_target = neighbors[idx]
                def_action[target.item(), new_target.item()] += 1
                num_defender -= 1
        return def_action

    @staticmethod
    def sample_att_action_uniform(state, device):
        num_target = state.size(0)
        attacked = -1
        while attacked == -1:
            attacked = torch.randint(0, num_target, [1, ])
            if state[attacked, 1] == 1:
                attacked = -1
        attack = torch.zeros([num_target], dtype=torch.float32, device=device)
        attack[attacked] = 1
        return attack

    @staticmethod
    def sample_att_action_A2C(observation, trained_strategy, num_att, device):
        with torch.no_grad():
            actor, critic = trained_strategy(observation.unsqueeze(0))
        return GameSimulation.sample_att_action_from_distribution(actor.squeeze(0), num_att, device)

    @staticmethod
    def sample_att_action_A2C_LSTM(observation, trained_strategy,
                                   action_hidden_state, action_cell_state, value_hidden_state, value_cell_state,
                                   num_att, device):
        with torch.no_grad():
            actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                = trained_strategy(observation.unsqueeze(0), action_hidden_state,
                                   action_cell_state, value_hidden_state, value_cell_state)
            att_action = GameSimulation.sample_att_action_from_distribution(actor.squeeze(0), num_att, device)
            return att_action, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state

    @staticmethod
    def sample_def_action_suqr(state, payoff, adj, device):
        num_target = config.NUM_TARGET
        def_action = torch.zeros(num_target, num_target).to(device)
        def_location = torch.nonzero(state[:, 0])
        temp_adj_matrix = torch.where(adj == config.MIN_VALUE, 0, 1)

        for target in def_location:
            num_defender = state[target.item(), 0].item()
            neighbors = torch.nonzero(temp_adj_matrix[target.item(), :]).squeeze(1)
            def_qr = get_suqr_probs(payoff)
            while num_defender > 0:
                init_prob = def_qr[0]
                pivot = torch.rand(1)
                idx = 0
                while pivot > init_prob:
                    idx += 1
                    if idx >= len(def_qr):
                        idx -= 1
                        break
                    init_prob += def_qr[idx]
                if idx > (len(neighbors) - 1):
                    continue
                else:
                    new_target = neighbors[idx]
                    def_action[target.item(), new_target.item()] += 1
                    num_defender -= 1

        return def_action

    @staticmethod
    def sample_att_action_suqr(state, payoff, device):
        num_target = config.NUM_TARGET
        attack = torch.zeros([num_target], dtype=torch.float32, device=device)

        attacked = True
        atk_qr = get_suqr_probs(payoff)
        try_count = 0
        while attacked:
            if try_count > 25:
                atk_qr = get_suqr_probs(payoff)
                try_count = 0
            init_prob = atk_qr[0]
            pivot = torch.rand(1)
            idx = 0
            while pivot > init_prob:
                idx += 1
                if idx >= len(atk_qr):
                    idx -= 1
                    break
                init_prob += atk_qr[idx]
            if state[idx, 1] == 1:
                attacked = True
                try_count += 1
            else:
                attack[idx] = 1
                attacked = False

        return attack

    @staticmethod
    def sample_def_action_from_res_dist(state, distributions, device):
        # Returns Defender action as a (k x t) matrix where k is the number of resources and t is the number of targets
        num_target = state.size(0)
        num_resource = distributions.size(0)
        def_action = torch.zeros(num_resource, num_target).to(device)
        for i in range(num_resource):
            probs = distributions[i]
            non_zero_prob_idx = torch.nonzero(probs > 1e-4).squeeze(1)
            pivot = torch.rand(1).to(device)
            init_prob = probs[non_zero_prob_idx[0]].clone().to(device)
            idx = 0
            while pivot > init_prob:
                idx += 1
                if idx >= len(non_zero_prob_idx):
                    idx = idx - 1
                    break
                init_prob += probs[non_zero_prob_idx[idx]]
            def_action[i, non_zero_prob_idx[idx]] += 1

        return def_action

    @staticmethod
    def gen_next_state_from_def_res(state, def_action, att_action, def_cur_loc, payoff_matrix, adj_matrix):
        # Defender resource is not destroyed if it meets an attack
        num_target = payoff_matrix.size(0)
        num_resource = def_action.size(0)
        next_state = state.clone()
        def_immediate_utility = 0.0
        att_immediate_utility = 0.0

        next_state[:, 0] = def_action.sum(dim=0)

        protected = False
        for t in range(num_target):
            if att_action[t] == 1:
                # print(att_action[t])
                next_state[t, 1] = 1
                # next_state[t, 0] = 0
                for tprime in range(num_resource):
                    if def_action[tprime, t] >= 1:
                        def_immediate_utility = payoff_matrix[t, 0].clone()
                        att_immediate_utility = payoff_matrix[t, 3].clone()
                        protected = True
                        break
                if not protected:
                    def_immediate_utility = payoff_matrix[t, 1].clone()
                    att_immediate_utility = payoff_matrix[t, 2].clone()
                break

        # Calculating Defender movement cost
        is_start = state[:, 1].sum()
        for t in range(num_target):
            for tprime in range(num_resource):
                if def_action[tprime, t] >= 1 and def_cur_loc[tprime, t] < 1 and is_start > 0:
                    prev_loc = torch.nonzero(def_cur_loc[tprime])[0].item()
                    def_immediate_utility += (adj_matrix[tprime, prev_loc] * def_action[tprime, t])

        return next_state, def_immediate_utility, att_immediate_utility



