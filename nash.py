import os
import numpy as np
import nashpy as nash
import math
import torch
import time
import gambit

import configuration as config
from game_simulation import GameSimulation


def compute_nash(d_utils, a_utils, i):
    # Uses nashpy library
    rps = nash.Game(np.array(d_utils), np.array(a_utils))
    if i > 1:
        k = 0
        eqs = rps.lemke_howson(initial_dropped_label=k)
        def_mix = eqs[0]
        att_mix = eqs[1]
        while math.isnan(def_mix[0]) or math.isnan(att_mix[0]):
            if k >= (len(d_utils) * 2) - 1:
                break
            else:
                k += 1
            eqs = rps.lemke_howson(initial_dropped_label=k)
            def_mix = eqs[0]
            att_mix = eqs[1]
    else:
        eqs = list(rps.support_enumeration())[0]
        def_mix = eqs[0]
        att_mix = eqs[1]

    if math.isnan(def_mix[0]) or math.isnan(att_mix[0]):
        print("\nGame is degenerate.")
        return def_mix, att_mix, 0.0, 0.0
    '''
  elif np.size(def_mix) != np.size(att_mix):
    print("\nStrategy shapes are not aligned; game is degenerate.")
    return def_mix, att_mix, 0.0, 0.0
  '''

    u_d, u_a = rps[def_mix, att_mix]

    print(rps)
    print("\nDefender Nash Strategy:\t", def_mix, "\nUtility:", u_d)
    print("Attacker Nash Strategy:\t", att_mix, "\nUtility:", u_a, "\n")

    return def_mix, att_mix, u_d, u_a


def gambit_nash(def_pure_set, att_pure_set, payoff):
    # Uses gambit executables
    def_strats = [strat for strat in def_pure_set.keys()]
    atk_strats = [strat for strat in att_pure_set.keys()]

    d_utils = []
    for strat in def_strats:
        d = [util[1][0] for util in payoff.items() if util[0][0] == strat]
        d_utils.append(d)

    a_utils = []
    for strat in atk_strats:
        a = [util[1][1] for util in payoff.items() if util[0][1] == strat]
        a_utils.append(a)

    utils = []
    for d in def_strats:
        u = []
        for a in atk_strats:
            u.append(payoff[(d, a)])
        utils.append(u)

    prologue = 'NFG 1 R  "Game ' +'"\n'
    players = '{"Defender" "Attacker"} {' + str(len(utils)) + ' ' + str(len(utils[0])) + '}\n\n'

    util_list = []
    for row in utils:
        row = list(sum(row, ()))
        util_list.extend(row)

    if os.path.isfile("gambit\in.txt"):
        os.remove("gambit\in.txt")

    f = open("gambit\in.txt", "w")
    f.write(prologue)
    f.write(players)

    for util in util_list:
        f.write(str(util))
        f.write(" ")
    f.close()

    start = time.time()
    os.system("gambit\gambit-gnm.exe -q < gambit\in.txt > gambit\out.txt")

    out = open("gambit\out.txt", "r")
    eqs = out.readlines()

    if len(eqs) < 1:
        # return [], [], 0.0, 0.0
        print("GNM returned no solutions. Trying EnumPoly --")
        os.system("gambit\gambit-enumpoly.exe -q < gambit\in.txt > gambit\out.txt")
        out = open("gambit\out.txt", "r")
        enum_eqs = out.readlines()
        eqs = []
        for line in enum_eqs:
            line = line.split(",")
            for i,item in enumerate(line):
                if "/" in item:
                    string = item.split("/")
                    num = float(string[0])/float(string[-1])
                    line[i] = str(num)
            eqs.append(",".join(line))
        if len(eqs) < 1:
            print("Game is degenerate.")
            return [], [], 0.0, 0.0

    if len(eqs) > 1:
        print("\nThere are %d Nash equilibria. The one giving the max Defender utility will be selected." % len(eqs))

    d_n_list = []
    a_n_list = []
    for eq in eqs:
        eq = eq.split(",")
        deq = eq[1:(len(utils) + 1)]
        aeq = eq[(len(utils) + 1):]
        deq = [float(x) for x in deq]
        aeq = [float(y) for y in aeq]

        def_eq_u = 0
        atk_eq_u = 0
        for i,x in enumerate(def_strats):
            for j,a in enumerate(atk_strats):
                def_eq_u += payoff[(x, a)][0] * deq[i] * aeq[j]
                atk_eq_u += payoff[(x, a)][1] * deq[i] * aeq[j]

        d_n_list.append(def_eq_u)
        a_n_list.append(atk_eq_u)

    best_eq = eqs[d_n_list.index(max(d_n_list))][:-1].split(",")
    u_d = max(d_n_list)
    u_a = a_n_list[d_n_list.index(max(d_n_list))]

    d_eq = best_eq[1:(len(utils) + 1)]
    a_eq = best_eq[(len(utils) + 1):]
    d_eq = [float(x) for x in d_eq]
    a_eq = [float(y) for y in a_eq]

    print("\nNE Calculation Time:", round((time.time() - start) / 60, 4), "min, for %d x %d matrix" % (len(d_eq), len(a_eq)))
    print("\nDefender Nash Strategy:\t", d_eq, len(d_eq), "\nUtility:", u_d)
    print("Attacker Nash Strategy:\t", a_eq, len(a_eq), "\nUtility:", u_a, "\n")

    return d_eq, a_eq, u_d, u_a


def strat_dom(play_set, utils):
    if len(utils) < 2:
        return [], utils

    dominated_list = []
    dom_check = True

    while dom_check:
        utils_T = np.array(utils).T
        min_list = []

        for i, col in enumerate(utils_T):
            min_idx = [j for j, x in enumerate(col) if x == min(col)]
            min_list.append(set(min_idx))
            if len(min_list) > 1:
                if len(min_list[-1].intersection(min_list[-2])) < 1:
                    print("No dominated strategies.")
                    return dominated_list, utils

        dom_idx = list(set.intersection(*map(set, min_list)))
        dom_idx.sort(reverse=True)

        new_utils = utils
        if dom_idx:
            if len(dom_idx) == len(utils):
                print("No dominated strategies left. Returning", dominated_list, "\n")
                return dominated_list, new_utils
            dominated_list.extend([strat[0] for k, strat in enumerate(play_set.items()) if k in dom_idx])
            for idx in dom_idx:
                del new_utils[idx]
        else:
            dom_check = False

    return dominated_list, new_utils


def strat_num(strat_name):
    num = []
    for i in range(1, len(strat_name)):
        if strat_name[-i].isnumeric():
            num.append(strat_name[-i])
        else:
            return int(''.join(num[::-1]))


def solveNash(def_pure_set, att_pure_set, payoff_matrix):
    # Uses gambit python extension library
    # Among different Nash, we will choose the Nash that provides the highest expected utility for the attacker
    gambitGame = gambit.Game.new_table([len(def_pure_set), len(att_pure_set)])
    playerNames = ["defender", "attacker"]
    for i in range(len(playerNames)):
        # label the player with its name
        gambitGame.players[i].label = playerNames[i]
    i = 0
    for x in def_pure_set:
        gambitGame.players[0].strategies[i].label = x
        i = i + 1

    i = 0
    for a in att_pure_set:
        gambitGame.players[1].strategies[i].label = a
        i = i + 1

    i = 0
    for x in def_pure_set:
        j = 0
        for a in att_pure_set:
            # iterate over matching player2 strategies
            gambitGame[i, j][0] = gambit.Decimal(payoff_matrix[(x, a)][0]).quantize(gambit.Decimal('1.0000'))
            gambitGame[i, j][1] = gambit.Decimal(payoff_matrix[(x, a)][1]).quantize(gambit.Decimal('1.0000'))
            j = j + 1
        i = i + 1

    solver = gambit.nash.ExternalEnumMixedSolver()
    eqResult = solver.solve(gambitGame)

    chosen_def_eq_strategy = []
    chosen_att_eq_strategy = []
    chosen_att_eq_u = -np.inf
    chosen_def_eq_u = -np.inf
    for curEq in eqResult:

        def_eq_strategy = curEq["defender"]
        att_eq_strategy = curEq["attacker"]

        def_eq_u = 0
        att_eq_u = 0
        i = 0
        for x in def_pure_set:
            j = 0
            for a in att_pure_set:
                def_eq_u = def_eq_u + payoff_matrix[(x, a)][0] * def_eq_strategy[i] * att_eq_strategy[j]
                att_eq_u = att_eq_u + payoff_matrix[(x, a)][1] * def_eq_strategy[i] * att_eq_strategy[j]
                j = j + 1
            i = i + 1

        # # Checking if the Nash is computed correctly
        # print("-------------Equilibrium check---------------\n")
        # i = 0
        # for x in def_pure_set:
        #     j = 0
        #     temp = 0
        #     for a in att_pure_set:
        #         temp = temp + payoff_matrix[(x, a)][0] * att_eq_strategy[j]
        #         j = j + 1
        #     print(temp, "\t", def_eq_strategy[i])
        #     i = i + 1
        #
        # j = 0
        # for a in att_pure_set:
        #     i = 0
        #     temp = 0
        #     for x in def_pure_set:
        #         temp = temp + payoff_matrix[(x, a)][1] * def_eq_strategy[i]
        #         i = i + 1
        #     print(temp, "\t", att_eq_strategy[j])
        #     j = j + 1

        if att_eq_u > chosen_att_eq_u:
            chosen_att_eq_u = att_eq_u
            chosen_def_eq_u = def_eq_u
            chosen_att_eq_strategy = att_eq_strategy
            chosen_def_eq_strategy = def_eq_strategy

    return chosen_def_eq_strategy, chosen_att_eq_strategy, chosen_def_eq_u, chosen_att_eq_u
