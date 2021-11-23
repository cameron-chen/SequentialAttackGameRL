from __future__ import print_function

import torch
import time
import sys

from utils import GameGeneration
import configuration as config
from utils import Pure_Strategy, play_game
from nash import gambit_nash, strat_dom, strat_num
from plot import plot
import nash

from attacker_oracle import AttackerOracle
from defender_oracle import DefenderOracle
import game_simulation
from defender_discriminator import DefDiscriminator, Def_Disc_CNN

d_option = 'A2C-GCN'
a_option = 'A2C-GCN'
if len(sys.argv) > 2:
    if 'LSTM' in sys.argv[1]:
        d_option += str(sys.argv[1])
        a_option += str(sys.argv[2])
    else:
        a_option += str(sys.argv[2])
elif len(sys.argv) > 1:
    d_option += str(sys.argv[1])
print('Defender:', d_option)
print('Attacker:', a_option)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game
game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free', num_res=config.NUM_RESOURCE,
                          device=device)
payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
def_constraints = [[1, 3], [0, 2], [4]]

# Initialization
payoff = dict()  # payoff matrix
def_pure_set = dict()  # set of pure strategies / trained policies of the defender
def_mix_set = dict()  # set of mixed strategies of the defender over double oracle iterations
att_pure_set = dict()  # for the attacker
att_mix_set = dict()  # for the attacker

def_u_ub = []  # defender's utilities w.r.t defender oracle
def_u_nash = []  # defender's utility w.r.t Nash equilibrium
att_u_ub = []  # for the attacker
att_u_nash = []  # for the attacker

# Defender Discriminator
if 'GAN' in d_option:
    print("\nTraining Defender Discriminator")
    disc_obj = DefDiscriminator(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, norm_adj_matrix,
                                def_constraints, device, threshold=1)
    discriminator = disc_obj.train()
else:
    discriminator = None

# Initializing Defender and Attacker sample strategies (uniform)
init_att_mix_strategy = [Pure_Strategy('uniform', [], 0.2, 'atkr', 0.0),
                        Pure_Strategy('suqr', [], 0.8, 'atk_suqr', 0.0)]
def_oracle = DefenderOracle(att_strategy=init_att_mix_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                            norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
print("\nGenerating Initial Defender Strategies...")
init_def_pure_set = def_oracle.train(option=d_option, discriminator=discriminator)

init_def_mix_strategy = [Pure_Strategy('uniform', [], 0.2, 'defr', 0.0),
                        Pure_Strategy('suqr', [], 0.8, 'def_suqr', 0.0)]
att_oracle = AttackerOracle(def_strategy=init_def_mix_strategy, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                            norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
print("Generating Initial Attacker Strategies...")
init_att_pure_set = att_oracle.train(option=a_option)

def_strategy_count = 0
for new_def in init_def_pure_set:
    cur_def_strategy, cur_def_ub, _ = new_def
    def_pure_set["def" + str(def_strategy_count)] = cur_def_strategy
    def_strategy_count += 1

att_strategy_count = 0
for new_att in init_att_pure_set:
    cur_att_strategy, cur_att_ub, _ = new_att
    att_pure_set["att" + str(att_strategy_count)] = cur_att_strategy
    att_strategy_count += 1

# Double Oracle Algorithm
iter = 1
converge = False
start = time.time()
while not converge:
    print("\n\nIteration", iter)

    # PART 1
    # Computing payoffs between new policies
    for x in def_pure_set.keys():
        for a in att_pure_set.keys():
            if (x, a) not in payoff.keys():
                payoff[(x, a)] = play_game(def_pure_set[x], att_pure_set[a], payoff_matrix,
                                            adj_matrix, def_constraints, d_option, a_option)
    print("\nNew payoffs calculated.\n")

    # PART 2: CORE LP with gambit
    # def_mix, att_mix, u_d, u_a = nash.solveNash(def_pure_set, att_pure_set, payoff)
    def_mix, att_mix, u_d, u_a, def_strats, atk_strats = gambit_nash(def_pure_set, att_pure_set, payoff, dom=1)
    if len(def_mix) < 1:
        break

    def_mix_set["x" + str(iter)] = def_mix
    att_mix_set["a" + str(iter)] = att_mix

    def_mix_strat = []
    att_mix_strat = []

    dom_def_set = {key:val for key,val in def_pure_set.items() if key in def_strats}
    for i, (key, value) in enumerate(dom_def_set.items()):
        def_mix_strat.append(Pure_Strategy(d_option, value, def_mix[i], key, 0))

    dom_atk_set = {key:val for key,val in att_pure_set.items() if key in atk_strats}
    for j, (key, value) in enumerate(dom_atk_set.items()):
        att_mix_strat.append(Pure_Strategy(a_option, value, att_mix[j], key, 0))

    # PART 3: Oracle Training
    # Training Defender Oracle
    print('Training Defender Oracle', iter, ':', d_option, "-- Def Set:", len(def_pure_set.keys()))

    def_local_optim = []
    for i_samp in range(3):
        def_oracle = DefenderOracle(att_strategy=att_mix_strat, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
        new_def_pure_set = def_oracle.train(option=d_option, discriminator=discriminator)
        for n_def in new_def_pure_set:
            def_local_optim.append(n_def)

    while max(def_local_optim, key=lambda x: x[1])[1] < u_d - 0.01:
        print("Finding better defender policy...")
        def_oracle = DefenderOracle(att_strategy=att_mix_strat, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)
        new_def_pure_set = def_oracle.train(option=d_option, discriminator=discriminator)
        for n_def in new_def_pure_set:
            def_local_optim.append(n_def)
        if len(def_local_optim) >= 50:
            break
    print('Defender Local Optima:', [round(x[1], 4) for x in def_local_optim], len(def_local_optim))
    new_def, def_util_x, _ = def_local_optim.pop(
        def_local_optim.index(max(def_local_optim, key=lambda x: x[1])))

    def_pure_set["def" + str(def_strategy_count)] = new_def
    def_strategy_count += 1

    '''
    # For adding multiple strategies per oracle iteration
    for i_strat in range(len(def_local_optim)):
        new_def, new_def_util, new_att_u_lb = def_local_optim.pop(def_local_optim.index(max(def_local_optim, key=lambda x: x[1])))
        if new_def_util > u_d:
            def_pure_set["def"+str(def_strategy_count)] = new_def
            def_strategy_count += 1
    '''

    # Training Attacker Oracle
    print('\nTraining Attacker Oracle', iter, ':', a_option, "-- Att Set:", len(att_pure_set.keys()))

    att_local_optim = []
    for j_samp in range(3):
        att_oracle = AttackerOracle(def_strategy=def_mix_strat, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)

        new_att_pure_set = att_oracle.train(option=a_option)
        for n_att in new_att_pure_set:
            att_local_optim.append(n_att)

    while max(att_local_optim, key=lambda x: x[1])[1] < u_a - 0.01:
        print("Finding better attacker policy...")
        att_oracle = AttackerOracle(def_strategy=def_mix_strat, payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                                    norm_adj_matrix=norm_adj_matrix, def_constraints=def_constraints, device=device)

        new_att_pure_set = att_oracle.train(option=a_option)
        for n_att in new_att_pure_set:
            att_local_optim.append(n_att)
        if len(att_local_optim) >= 50:
            break
    print('Attacker Local Optima:', [round(x[1], 4) for x in att_local_optim], len(att_local_optim))
    new_att, att_util_y, _ = att_local_optim.pop(
        att_local_optim.index(max(att_local_optim, key=lambda x: x[1])))

    att_pure_set["att" + str(att_strategy_count)] = new_att
    att_strategy_count += 1

    '''
    # For adding multiple strategies per oracle iteration
    for j_strat in range(len(att_local_optim)):
        new_att, new_att_util, new_def_u_lb = att_local_optim.pop(att_local_optim.index(max(att_local_optim, key=lambda x: x[1])))
        if new_att_util > u_a:
            att_pure_set["att"+str(att_strategy_count)] = new_att
            att_strategy_count += 1
    '''

    # Upper Bounds
    def_u_ub.append(def_util_x)
    att_u_ub.append(att_util_y)

    # Nash Utilities
    def_u_nash.append(u_d)
    att_u_nash.append(u_a)

    print("\nDefender Oracle Utility:\t", def_u_ub[-1])
    print("Defender Nash Utility:\t", def_u_nash[-1])

    print("\nAttacker Oracle Utility:\t", att_u_ub[-1])
    print("Attacker Nash Utility:\t", att_u_nash[-1])

    # Check for convergence
    if def_u_ub[-1] < def_u_nash[-1] + 0.01 and att_u_ub[-1] < att_u_nash[-1] + 0.01:
        if iter > 10:
            print('Oracles have converged.')
            converge = True
            break

    # if iter % 2 == 0:
    plot(def_u_ub, def_u_nash, 'Defender', d_option)
    plot(att_u_ub, att_u_nash, 'Attacker', a_option)

    iter += 1

plot(def_u_ub, def_u_nash, 'Defender', d_option)
plot(att_u_ub, att_u_nash, 'Attacker', a_option)

print("\n", round((time.time() - start) / 60, 2), "min")
