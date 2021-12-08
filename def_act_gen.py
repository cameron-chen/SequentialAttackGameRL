import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import configuration as config
from graph_convolution import GraphConvolution
from utils import GameGeneration
from defender_discriminator import DefDiscriminator
from sampling import gen_init_def_pos, check_move, check_constraints


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


def create_mask(def_cur_loc, adj_matrix, threshold=1):
    num_res, num_tar = def_cur_loc.size()
    pos = [res.nonzero().item() for res in def_cur_loc]
    mask = torch.zeros(def_cur_loc.size(), dtype=torch.bool)

    for i,res in enumerate(mask):
        val = [n for n in range(num_tar) if adj_matrix[pos[i]][n] != config.MIN_VALUE]
        res[val] = 1

    return mask

def noiser(input, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return input + torch.randn(input.size()).to(device)

class Def_Action_Generator(nn.Module):
    def __init__(self, num_tar, num_res, adj_matrix, device):
        super(Def_Action_Generator, self).__init__()
        self.num_tar = num_tar
        self.num_res = num_res
        self.adj_matrix = adj_matrix
        self.l1 = nn.Linear((16+num_res)*num_tar, 18*num_tar)
        self.l2 = nn.Linear(18*num_tar, 14*num_tar)
        self.l3 = nn.Linear(14*num_tar, num_tar*num_res)
        self.bn = nn.BatchNorm1d(num_tar)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x, def_cur_loc):
        x = torch.cat((x.view(16, self.num_tar), def_cur_loc))
        x = self.relu(self.l1(noiser(x.view(-1))))
        x = self.relu(self.l2(x))
        x = self.bn(self.l3(x).view(self.num_res, self.num_tar))

        # Meeting adajency constraints
        mask = create_mask(def_cur_loc, self.adj_matrix)
        mask = torch.where(mask == 0, -9999.0, 0.0).float()
        x = self.softmax(x + mask)

        return x

class Def_A2C_Sample_Generator(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature,
                 num_resource, act_gen, device):
        super(Def_A2C_Sample_Generator, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norma_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)
        self.num_resource = num_resource
        self.device = device
        self.noise_feat = 2

        self.gc1 = GraphConvolution(num_feature+self.noise_feat, 32)
        self.bn = nn.BatchNorm1d(self.num_target)

        self.gc2 = GraphConvolution(32, 16)

        self.ln_value1 = nn.Linear(16 * self.num_target, 32)
        self.ln_value2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.act_gen = act_gen

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state, def_cur_loc):
        batch_size = len(state)
        noise = torch.rand((1, state.size(1), self.noise_feat)).to(self.device)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1), noise), 2)

        x = self.relu(self.bn(self.gc1(x, self.norma_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norma_adj_matrix)))
        # x = self.relu(self.ln1(x))
        x = x.view(-1, 16 * self.num_target)

        act_estimates = self.act_gen(x.squeeze(), def_cur_loc)
        for i in range(1000-1):
            act_estimates = torch.cat((act_estimates, self.act_gen(x.squeeze(), def_cur_loc)))

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        return act_estimates.view(1000, self.num_resource, self.num_target)


class Def_A2C_Graph_Convolutor(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature,
                 num_resource, device):
        super(Def_A2C_Graph_Convolutor, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norma_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)
        self.num_resource = num_resource
        self.device = device
        self.noise_feat = 2

        self.gc1 = GraphConvolution(num_feature+self.noise_feat, 32)
        self.bn = nn.BatchNorm1d(self.num_target)

        self.gc2 = GraphConvolution(32, 16)

        self.ln_value1 = nn.Linear(16 * self.num_target, 32)
        self.ln_value2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state):
        batch_size = len(state)
        noise = torch.rand((1, state.size(1), self.noise_feat)).to(self.device)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1), noise), 2)

        x = self.relu(self.bn(self.gc1(x, self.norma_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norma_adj_matrix)))
        # x = self.relu(self.ln1(x))
        x = x.view(-1, 16 * self.num_target)

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        return x, state_value


class DefActionGen():
    def __init__(self, num_targ, num_res, num_feature, payoff_matrix, adj_matrix, norm_adj_matrix,
                def_constraints, discriminator, device):
        self.num_targ = num_targ
        self.num_res = num_res
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.def_constraints = def_constraints
        self.device = device
        self.threshold = 1

        self.discriminator = discriminator
        self.disc_criterion = nn.BCELoss()
        self.graph_conv = Def_A2C_Graph_Convolutor(payoff_matrix, adj_matrix, norm_adj_matrix, num_feature, num_res, device)

    def train(self, episodes=100, test=0):
        act_gen = Def_Action_Generator(self.num_targ, self.num_res, self.adj_matrix, self.device)
        attempt_list = []
        act_list = []

        invalid_list = []
        gen_loss_list = []
        disc_loss_list = []

        for i in range(episodes):
            gen_lr = 0.001
            disc_lr = 0.001 
            gen_optimizer = optim.Adam(act_gen.parameters(), gen_lr)
            disc_optimizer = optim.Adam(self.discriminator.parameters(), disc_lr)

            state = torch.zeros(self.num_targ, 2, dtype=torch.int32, device=device)
            def_cur_loc = gen_init_def_pos(self.num_targ, self.num_res, self.adj_matrix, def_constraints, threshold=1)
            for t, res in enumerate(def_cur_loc):
                state[(res == 1).nonzero(), 0] += int(sum(res))

            num_atk = random.randint(0, 4)
            atk_pos = random.sample(range(0, self.num_targ-1), num_atk)
            for x in atk_pos:
                if state[x, 0] < 1:
                    state[x, 1] = 1

            # Action Generation
            action_generated = False
            attempt = 1
            while not action_generated:
                gen_optimizer.zero_grad()
                x, state_value = self.graph_conv(state.unsqueeze(0))
                act_estimates = act_gen(x.squeeze(), def_cur_loc)
                for i in range(1000-1):
                    act_estimates = torch.cat((act_estimates, act_gen(x.squeeze(), def_cur_loc)))
                act_estimates = act_estimates.view(1000, self.num_res, self.num_targ)

                actions, act_probs, act_dist, _ = dist_est(act_estimates)

                invalid_count = 0
                invalid_act = set()
                invalid_est = []
                for i,act in enumerate(actions):
                    meet_constraints = False
                    val = check_move(def_cur_loc, act, self.adj_matrix, self.threshold)
                    if val:
                        meet_constraints = check_constraints(act, self.adj_matrix, self.def_constraints, 
                                                            self.threshold)
                    if not meet_constraints:
                        invalid_count += 1
                        invalid_act.add(act)
                        invalid_est.append(act_estimates[i])
                
                if invalid_count > (len(actions)*0.25):        # Threshold: 25% invalid actions
                    # Update generator with discriminator
                    for i,act_est in enumerate(act_estimates):                # trying with all samples -- invalid_est only for invalid samples
                        inval_samp = torch.cat((def_cur_loc, act_est))
                        if i < 1:
                            inval_out = self.discriminator(inval_samp)
                        else:
                            inval_out = torch.cat((inval_out, self.discriminator(inval_samp)))
                    true_labels = torch.ones(inval_out.size()).to(self.device)
                    gen_loss = self.disc_criterion(inval_out, true_labels)
                    if test:
                        print("\nAttempts:", attempt)
                        print("Invalid Samples:", invalid_count)
                        print("Generator Loss:", gen_loss.item())
                        print("Actions:", len(act_dist.values()))
                    gen_loss.backward()
                    gen_optimizer.step()

                    attempt += 1
                    invalid_list.append(invalid_count)
                    gen_loss_list.append(gen_loss.item())

                    # Update discriminator
                    disc_err_rate = 1.0
                    disc_attempt = 1
                    disc_lr = 0.001
                    while disc_err_rate > 0.2:
                        disc_optimizer.zero_grad()
                        for i,act_est in enumerate(invalid_est):
                            samp = torch.cat((def_cur_loc, act_est.detach()))
                            if i < 1:
                                disc_pred = self.discriminator(samp)
                            else:
                                disc_pred = torch.cat((disc_pred, self.discriminator(samp)))
                        disc_error = disc_pred[torch.where(disc_pred > 0.5)]
                        false_labels = torch.zeros(disc_pred.size()).to(self.device)
                        disc_loss = self.disc_criterion(disc_pred, false_labels)
                        disc_loss.backward()
                        disc_optimizer.step()
                        disc_err_rate = len(disc_error)/len(disc_pred)
                        if test:
                            print("\nDiscriminator Error Rate:", disc_err_rate)
                            print("Discriminator Loss:", disc_loss.item())
                        disc_loss_list.append(disc_loss.item())

                        disc_lr = disc_lr * 0.95
                        for param_group in disc_optimizer.param_groups:
                            param_group['lr'] = disc_lr

                        if disc_attempt > 50:
                            break
                else:
                    print("\nAttempts:", attempt)
                    print("Invalid Samples:", invalid_count)
                    print("Actions:", len(act_dist.values()), "\n")
                    action_generated = True

                gen_lr = gen_lr * 0.95
                for param_group in gen_optimizer.param_groups:
                    param_group['lr'] = gen_lr
                
                if attempt > 50:
                    break
        
            attempt_list.append(attempt)
            act_list.append(len(act_dist.values()))

        if test:
            return act_gen, attempt_list, act_list
        else:
            return act_gen


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free',
                              num_res=config.NUM_RESOURCE, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
    # def_constraints = [[0, 2], [1, 3], [4]]

    disc_obj = DefDiscriminator(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, norm_adj_matrix,
                                def_constraints, device, threshold=1)
    def_disc = disc_obj.train(episodes=1600, option='CNN')

    act_gen_obj = DefActionGen(config.NUM_TARGET, config.NUM_RESOURCE, config.NUM_FEATURE, payoff_matrix,
                                adj_matrix, norm_adj_matrix, def_constraints, def_disc, device)
    act_gen, attempt_list, act_list = act_gen_obj.train(episodes=100, test=1)

    plt.figure(figsize=(20, 10))
    plt.title("# of Episodes to Generate Valid Action (0.25 threshold for invalid actions, 1000 samples)")
    plt.xlabel("GAN")
    plt.ylabel("Episodes")
    plt.plot(attempt_list)
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title("# of Unique Actions Generated (0.25 threshold for invalid actions, 1000 samples)")
    plt.xlabel("GAN")
    plt.ylabel("Number of Unique Actions")
    plt.plot(act_list)
    plt.show()

    