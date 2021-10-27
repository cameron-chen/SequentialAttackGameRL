import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import configuration as config


def get_action(act):
    action = torch.zeros((act.size()))
    act_code = []
    for i,res in enumerate(act):
        idx = (res == max(res)).nonzero()[0].item()
        action[i][idx] = 1
        act_code.append(idx)
    
    return action, tuple(act_code)


def dist_est(act_estimates):
    act_dist = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actions = []
    codes = []
    for act in act_estimates:
        action, act_code = get_action(act)
        if act_code in act_dist.keys():
            act_dist[act_code] += 1
        else:
            act_dist[act_code] = 1
        actions.append(action)
        codes.append(act_code)

    act_probs = []
    for c in codes:
        act_probs.append(act_dist[c]/len(actions))

    return actions, torch.tensor(act_probs, device=device), act_dist, codes


def create_mask(def_cur_loc, threshold=1):
    num_res, num_tar = def_cur_loc.size()
    pos = [res.nonzero() for res in def_cur_loc]
    mask = torch.ones(def_cur_loc.size(), dtype=torch.bool)

    for i,res in enumerate(mask):
        if pos[i] == 0:
            val1 = [n for n in range(0, threshold+1)]
            val2 = [n for n in range(num_tar-threshold, num_tar)]
            val = val1 + val2
        elif pos[i] == num_tar-1:
            val1 = [n for n in range(0, threshold)]
            val2 = [n for n in range(num_tar-1-threshold, num_tar)]
            val = val1 + val2
        else:
            val = [n for n in range(pos[i]-threshold, pos[i]+threshold+1)]
        res[val] = 0

    return mask


class Def_Action_Generator(nn.Module):
    def __init__(self, num_tar, num_res, device):
        super(Def_Action_Generator, self).__init__()
        self.num_tar = num_tar
        self.num_res = num_res
        self.l1 = nn.Linear(16*num_tar, 14*num_tar)
        self.l2 = nn.Linear(14*num_tar, 12*num_tar)
        self.l3 = nn.Linear(12*num_tar, num_tar*num_res)
        self.bn = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x, def_cur_loc):
        noise = torch.randn(x.size()).to(self.device)
        x = self.relu(self.l1(x + noise))
        x = self.relu(self.l2(x))
        x = self.sig(self.bn(self.l3(x).view(self.num_res, self.num_tar)))

        # Meeting adajency constraints
        mask = create_mask(def_cur_loc).to(self.device)
        x = torch.masked_fill(x, mask, value=0)

        return x
