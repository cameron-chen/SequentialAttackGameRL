from __future__ import print_function
import torch

import networkx as nx
import numpy as np
import random
import time
from collections import namedtuple, deque

from game_simulation import GameSimulation
from sampling import gen_init_def_pos
import configuration as config

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TransitionV = namedtuple('TransitionV', ('state', 'action', 'next_state', 'reward', 'mask', 'prob'))
Pure_Strategy = namedtuple('Pure_Strategy', ('type', 'trained_strategy', 'probability', 'name', 'value'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                res = random.randint(0, self.num_res - 1)
                while res in (item for group in def_constraints for item in group):
                    res = random.randint(0, self.num_res - 1)
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

class ReplayMemoryTransitionV(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = TransitionV(*args)
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
        lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        n_sample = 50

        for i_sample in range(n_sample):
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=device)
            if 'GAN' in d_option:
                def_init_loc = gen_init_def_pos(num_target, config.NUM_RESOURCE, def_constraints, threshold=1)
                for t, res in enumerate(def_init_loc):
                    init_state[(res == 1).nonzero(), 0] += int(sum(res))
            else:
                entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])
                for t in range(0, len(entries)):
                    init_state[entries[t], 0] += 1

            state = init_state
            init_attacker_observation = torch.zeros(num_target, 2, dtype=torch.int32, device=device)
            init_attacker_observation[:, 0] = -1
            attacker_observation = init_attacker_observation
            num_att = config.NUM_ATTACK

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

                    if num_att < config.NUM_ATTACK and state[:, 0].sum() == 0:
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


def gen_init_def_pos(num_targ, num_res, adj_matrix, def_constraints, threshold=1):
    next_loc_available = [[x for x in range(num_targ)] for _ in range(num_res)]
    loc = torch.zeros((num_res, num_targ), dtype=torch.float, device=device)
    for constraint in def_constraints:
        res_positions = []
        if len(constraint) > 1:
            for j,res in enumerate(constraint):
                if j > 0:
                    neighbor_list = next_loc_available[res].copy()
                    for k in res_positions:
                        for k in res_positions:
                            del_list = []
                            for move in neighbor_list:
                                if adj_matrix[k][move] == config.MIN_VALUE:
                                    del_list.append(move)
                            neighbor_list = [t for t in neighbor_list if t not in del_list]
                    if len(neighbor_list) < 1:
                        return []
                    pos = random.choice(neighbor_list)
                else:
                    pos = random.choice(next_loc_available[res])
                loc[res][pos] = 1
                res_positions.append(pos)
        else:
            loc[constraint[0]][random.choice(next_loc_available[constraint[0]])] = 1

    return loc


def gen_def_move_rand(num_targ, num_res, adj_matrix, def_constraints, threshold=1):
    cur_loc = gen_init_def_pos(num_targ, num_res, adj_matrix, def_constraints, threshold)
    if len(cur_loc) < 1:
        cur_loc = torch.zeros((num_res, num_targ), dtype=torch.float, device=device)
        for row in cur_loc:
            row[random.randint(0, num_targ - 1)] = 1

    next_loc = torch.zeros((num_res, num_targ), dtype=torch.float, device=device)
    for res in next_loc:
        res[random.randint(0, num_targ - 1)] = 1

    return cur_loc, next_loc


def check_move(cur_loc, next_loc, adj_matrix, threshold=1, test=None):
    valid = True
    for i, res in enumerate(next_loc):
        a = (cur_loc[i] == 1).nonzero()[0].item()
        b = (res == 1).nonzero()[0].item()
        if adj_matrix[a][b] == config.MIN_VALUE:
            valid = False
            if test: print("Moving resource", i + 1, "from target", a, "to target", b, "is invalid.")
    return valid


def check_constraints(next_loc, adj_matrix, def_constraints, threshold=1, test=None):
    valid = True
    for group in def_constraints:
        if test: print("Group:", group)
        for res in group:
            pos = (next_loc[res] == 1).nonzero()[0].item()
            res_group = [x for x in group if x != res]
            for other_res in res_group:
                other_pos = (next_loc[other_res] == 1).nonzero().item()
                if adj_matrix[pos][other_pos] == config.MIN_VALUE:
                    valid = False
                    if test: print("Move is invalid.")
    return valid


def gen_samples(num_target, num_res, adj_matrix, def_constraints, threshold=1, sample_number=500, samples=[]):
    count = 0
    t_samps = []
    f_samps = []
    start = time.time()
    limit = sample_number/5
    while len(t_samps) < sample_number:
        cur_loc, next_loc = gen_def_move_rand(num_target, num_res, adj_matrix, def_constraints, threshold)
        samp = cur_loc.tolist() + next_loc.tolist()
        if samp not in samples:
            val = check_move(cur_loc, next_loc, adj_matrix, threshold)
            check = check_constraints(next_loc, adj_matrix, def_constraints, threshold)
            def_trans = torch.cat((cur_loc, next_loc))
            if val and check:
                count += val
                if len(t_samps) < sample_number:
                    t_samps.append((def_trans, torch.tensor(1, dtype=torch.float, device=device)))
                    samples.append(samp)
                    if len(t_samps) % limit == 0: print(len(t_samps), "valid samples generated")
            elif len(f_samps) < sample_number:
                f_samps.append((def_trans, torch.tensor(0, dtype=torch.float, device=device)))
                samples.append(samp)

    print(round((time.time() - start)/60, 4), "min")

    sample_set = t_samps + f_samps
    random.shuffle(sample_set)

    return sample_set, samples


def gen_next_loc(cur_loc, adj_matrix, def_constraints, threshold=1, next_loc_available=[]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_res, num_targ = cur_loc.size()
    # Getting list of available moves for each resource
    if len(next_loc_available) < 1:
        next_loc_available = [[] for _ in range(num_res)]
        for i,res in enumerate(cur_loc):
            res_pos = (res == 1).nonzero()[0].item()
            neighbors = [x for x in range(num_targ) if adj_matrix[res_pos][x] != config.MIN_VALUE]
            next_loc_available[i].extend(neighbors)

    # Generating new target for each resource that meets constraints, using available moves from above
    next_loc = torch.zeros((num_res, num_targ), dtype=torch.float, device=device)
    for constraint in def_constraints:
        res_positions = []
        if len(constraint) > 1:
            for j,res in enumerate(constraint):
                if j > 0:
                    neighbor_list = next_loc_available[res].copy()
                    for k in res_positions:
                        del_list = []
                        for move in neighbor_list:
                            if adj_matrix[k][move] == config.MIN_VALUE:
                                del_list.append(move)
                        neighbor_list = [t for t in neighbor_list if t not in del_list]
                    if len(neighbor_list) < 1:
                        return []
                    pos = random.choice(neighbor_list)
                else:
                    pos = random.choice(next_loc_available[res])
                next_loc[res][pos] = 1
                res_positions.append(pos)
        else:
            next_loc[constraint[0]][random.choice(next_loc_available[constraint[0]])] = 1

    return next_loc


def gen_valid_def_move(num_targ, num_res, adj_matrix, def_constraints, threshold=1):
    move_generated = False
    while not move_generated:
        cur_loc = gen_init_def_pos(num_targ, num_res, adj_matrix, def_constraints, threshold)
        if len(cur_loc) < 1:
            cur_loc = torch.zeros((num_res, num_targ), dtype=torch.float, device=device)
            for row in cur_loc:
                row[random.randint(0, num_targ - 1)] = 1

        next_loc = gen_next_loc(cur_loc, adj_matrix, def_constraints, threshold)
        if len(next_loc) < 1:
            continue
        if sum([sum(res) for res in next_loc]) == num_res:
            move_generated = True

    return cur_loc, next_loc


def convert_to_real(next_loc):
    # Converting binary next_loc to real action values
    num_targ = next_loc.size(1)
    for i,res in enumerate(next_loc):
        idx = (res == 1).nonzero()[0].item()
        next_loc[i] = torch.where(res != 0, res, torch.rand(num_targ).to(device))
        res[idx] = 0
        res[idx] = max(res) + (1-max(res))*torch.rand(1).to(device)

    return next_loc


def convert_to_real_adj(next_loc, cur_loc, adj_matrix, threshold=1, valid=1):
    # Converting binary next_loc to action values that meets adjacency constraint
    num_res, num_tar = cur_loc.size()
    if valid:
        pos = [res.nonzero().item() for res in cur_loc]
    else:
        pos = [torch.argmax(res) for res in next_loc]

    for i,res in enumerate(next_loc):
        idx = (res == 1).nonzero()[0].item()
        val = [n for n in range(num_tar) if adj_matrix[pos[i]][n] != config.MIN_VALUE]
        res[val] = (1/(2*threshold+1))*torch.rand(len(val)).to(device)
        res[idx] = 1-sum(res[val])

    return next_loc


def gen_samples_greedy(num_target, num_res, adj_matrix, def_constraints, threshold=1, sample_number=500, samples=[]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_samps = []
    f_samps = []
    start = time.time()

    while len(t_samps) < sample_number:
        cur_loc, next_loc = gen_valid_def_move(num_target, num_res, adj_matrix, def_constraints, threshold)
        samp = cur_loc.tolist() + next_loc.tolist()
        if samp not in samples:
            next_loc_real = convert_to_real_adj(next_loc, cur_loc, adj_matrix, threshold)
            def_trans = torch.cat((cur_loc, next_loc_real))
            t_samps.append((def_trans, torch.tensor(1, dtype=torch.float, device=device)))
            samples.append(samp)
            if len(t_samps) % sample_number == 0: print(len(t_samps), "valid samples generated")

    while len(f_samps) < sample_number:
        cur_loc, next_loc = gen_def_move_rand(num_target, num_res, adj_matrix, def_constraints, threshold)
        samp = cur_loc.tolist() + next_loc.tolist()
        if samp not in samples:
            val = check_move(cur_loc, next_loc, adj_matrix, threshold)
            if val:
                check = check_constraints(next_loc, adj_matrix, def_constraints, threshold)
            else:
                check = False
            if not val or not check:
                next_loc_real = convert_to_real_adj(next_loc, cur_loc, adj_matrix, threshold, valid=0)
                def_trans = torch.cat((cur_loc, next_loc_real))
                f_samps.append((def_trans, torch.tensor(0, dtype=torch.float, device=device)))
                samples.append(samp)
                if len(f_samps) % sample_number == 0: print(len(f_samps), "invalid samples generated")

    print(round((time.time() - start)/60, 4), "min")

    sample_set = t_samps + f_samps
    random.shuffle(sample_set)

    return sample_set, samples


def gen_possible_moves(def_cur_loc, adj_matrix, threshold=1):
    # Generating possible moves for each resource
    num_res, num_tar = def_cur_loc.size()
    possible_moves = []
    for res in def_cur_loc:
        loc = (res == 1).nonzero().item()
        possible_moves.append([n for n in range(num_tar) if adj_matrix[loc][n] != config.MIN_VALUE])
    return possible_moves


def calculate_combinations(def_cur_loc, possible_moves, adj_matrix, def_constraints, threshold=1):
    # Calculating number of moves
    num_res, num_tar = def_cur_loc.size()
    combinations = 1
    for i,c in enumerate(def_constraints):
        if len(c) < 2:
            combinations *= len(possible_moves[c[0]])
        else:
            combo = 0
            for loc in possible_moves[c[0]]:
                diff = []
                for other_res in c[1:]:
                    for other_loc in possible_moves[other_res]:
                        if adj_matrix[loc][other_loc] != config.MIN_VALUE:
                            diff.append(other_loc)
                combo += len(diff)
            combinations *= combo
        
    return combinations


def gen_all_actions(num_targ, num_res):
    all_moves = set()
    while len(all_moves) < num_targ**num_res:
        loc = torch.zeros((num_res, num_targ), dtype=torch.float)
        for res in loc:
            res[random.randint(0, num_targ - 1)] = 1
        all_moves.add(tuple([(res == 1).nonzero().item() for res in loc]))

    return sorted(list(all_moves))


def gen_all_valid_actions(cur_loc, adj_matrix, def_constraints, threshold=1):
    val_moves = set()
    possible_moves = gen_possible_moves(cur_loc, adj_matrix, threshold)
    combinations = calculate_combinations(cur_loc, possible_moves, adj_matrix, def_constraints, threshold)
    while len(val_moves) < combinations:
        loc = gen_next_loc(cur_loc, adj_matrix, def_constraints, threshold)
        val_moves.add(tuple([(res == 1).nonzero().item() for res in loc]))

    return sorted(list(val_moves))


def gen_val_mask(all_moves, val_moves):
    mask = torch.zeros(len(all_moves), dtype=torch.float, device=device)
    for i,move in enumerate(all_moves):
        if move in set(val_moves):
            mask[i] = 1
    
    return mask


def optimality_loss(state, def_cur_loc, def_constraints, payoff_matrix, adj_matrix, def_action, att_action, def_util, threshold=1):
    print("\nCalculating optimality loss...")
    possible_moves = gen_possible_moves(def_cur_loc, def_constraints, threshold)
    combinations = calculate_combinations(def_cur_loc, possible_moves, def_constraints, threshold)

    all_moves = {}
    while len(all_moves.keys()) < combinations:
        poss_loc = gen_next_loc(def_cur_loc, def_constraints, threshold, possible_moves)
        loc = tuple([(res == 1).nonzero().item() for res in poss_loc])
        if loc not in all_moves.keys():
            all_moves[loc] = GameSimulation.gen_next_state_from_def_res(state, poss_loc, att_action, 
                                                                        payoff_matrix, adj_matrix)[1]
    best_move = max(all_moves, key=all_moves.get)
    print("Best move:", best_move, "\tUtil:", all_moves[best_move])
    actual_move = tuple([(res == 1).nonzero().item() for res in def_action])
    print("Actual move:", actual_move, "\tUtil:", def_util)

    return (all_moves[best_move]-def_util)**2    



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free', num_res=config.NUM_RESOURCE, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
    print(payoff_matrix)
    print(adj_matrix)
    print(norm_adj_matrix)
    print(def_constraints)
