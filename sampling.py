import torch
import random
import time

from game_simulation import GameSimulation
import configuration as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


