import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import GameGeneration
from graph_convolution import GraphConvolution
from defender_discriminator import DefDiscriminator
from sampling import check_move, check_constraints, gen_init_def_pos
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
    def __init__(self, device):
        super(Def_Action_Generator, self).__init__()
        self.l1 = nn.Linear(400, 200)
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, 50)
        self.bn = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x, def_cur_loc):
        noise = torch.rand(240).to(self.device)
        x = torch.cat((x, noise))
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.sig(self.bn(self.l3(x).view(5, 10)))

        # Meeting adajency constraints
        mask = create_mask(def_cur_loc).to(self.device)
        x = torch.masked_fill(x, mask, value=0)

        return x


class Def_A2C_GAN(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature,
                 num_resource, def_constraints, discriminator, device):
        super(Def_A2C_GAN, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norma_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)
        self.num_resource = num_resource
        self.def_constraints = def_constraints
        self.discriminator = discriminator
        self.disc_criterion = nn.BCELoss()
        self.device = device
        self.threshold = 1
        self.noise_feat = 2

        self.gc1 = GraphConvolution(num_feature + self.noise_feat, 32)
        self.bn = nn.BatchNorm1d(self.num_target)

        self.gc2 = GraphConvolution(32, 16)

        self.ln1 = nn.Linear(16, 8)

        self.ln_value1 = nn.Linear(16 * self.num_target, 32)
        self.ln_value2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.act_gen = Def_Action_Generator(device)

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

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        # Action Generation
        action_generated = False
        attempt = 1
        invalid_list = []
        gen_loss_list = []
        disc_loss_list = []
        lr = 0.001
        gen_optimizer = optim.Adam(self.act_gen.parameters(), lr)
        disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)
        while not action_generated:
            gen_optimizer.zero_grad()
            act_estimates = []
            for i in range(1000):
                act_estimates.append(self.act_gen(x.detach().squeeze(), def_cur_loc))

            actions, act_probs, act_dist, _ = dist_est(act_estimates)

            invalid_count = 0
            invalid_act = set()
            invalid_est = []
            for i,act in enumerate(actions):
                meet_constraints = False
                val = check_move(def_cur_loc, act, self.threshold)
                if val:
                    meet_constraints = check_constraints(act, self.def_constraints, 
                                                         self.threshold)
                if not meet_constraints:
                    invalid_count += 1
                    invalid_act.add(act)
                    invalid_est.append(act_estimates[i])
            
            if invalid_count > (len(actions)*0.25):        # Threshold: 25% invalid actions
                for i,act_est in enumerate(invalid_est):
                    inval_samp = torch.cat((def_cur_loc, act_est))
                    if i < 1:
                        inval_out = self.discriminator(inval_samp)
                    else:
                        inval_out = torch.cat((inval_out, self.discriminator(inval_samp)))
                true_labels = torch.ones(inval_out.size()).to(self.device)
                gen_loss = self.disc_criterion(inval_out, true_labels)
                print("\n#", attempt)
                print("Invalid Samples:", invalid_count)
                print("Generator Loss:", gen_loss.item())
                print("Actions:", len(act_dist.values()))
                gen_loss.backward() # retain_graph=True)
                gen_optimizer.step()

                attempt += 1
                invalid_list.append(invalid_count)
                gen_loss_list.append(gen_loss.item())

                # Update discriminator
                disc_err_rate = 1.0
                while disc_err_rate > 0.2:
                    disc_optimizer.zero_grad()
                    for i,act_est in enumerate(invalid_est):
                        samp = torch.cat((def_cur_loc, act_est.detach()))
                        if i < 1:
                            disc_pred = self.discriminator(samp)
                        else:
                            disc_pred = torch.cat((disc_pred, self.discriminator(samp)))
                    print(disc_pred[:10])
                    disc_error = disc_pred[torch.where(disc_pred > 0.5)]
                    false_labels = torch.zeros(disc_pred.size()).to(self.device)
                    disc_loss = self.disc_criterion(disc_pred, false_labels)
                    disc_loss.backward()
                    disc_optimizer.step()
                    disc_err_rate = len(disc_error)/len(disc_pred)
                    print("\nDiscriminator Error Rate:", disc_err_rate)
                    print("Discriminator Loss:", disc_loss.item())
                    disc_loss_list.append(disc_loss.item())
            else:
                print("\n#", attempt)
                print("Invalid Samples:", invalid_count)
                print("Actions:", len(act_dist.values()))
                action_generated = True

            lr = lr * 0.95
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] = lr

            if attempt % 100 == 0 or action_generated:
                plt.figure(figsize=(20, 10))
                plt.title("# of Invalid Samples")
                plt.xlabel("Attempt")
                plt.ylabel("Invalid Samples")
                plt.plot(invalid_list)
                plt.show()

                plt.figure(figsize=(20, 10))
                plt.title("Generator Loss")
                plt.xlabel("Attempt")
                plt.ylabel("Loss")
                plt.plot(gen_loss_list, color='orange')
                plt.show()

                plt.figure(figsize=(20, 10))
                plt.title("Discriminator Loss")
                plt.xlabel("Attempt")
                plt.ylabel("Loss")
                plt.plot(disc_loss_list, color='blue')
                plt.show()
            '''
            elif attempt == 30 and invalid_count > len(act_estimates)*0.8:
                return (0, 0), 0, attempt
            elif attempt == 100:
                return (0, 0), 0, attempt
            '''


        for i, act in enumerate(actions):
            if act not in invalid_act:
                select_act = act
                select_prob = act_probs[i]
                break

        return (select_act, select_prob), state_value, attempt


if __name__ == '__main__':
    option = 'CNN'
    if len(sys.argv) > 2:
        if 'GCN' in sys.argv[1]:
            option = 'GCN'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free',
                              num_res=config.NUM_RESOURCE, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, _ = game_gen.gen_game()
    def_constraints = [[0, 2], [1, 3], [4]]

    disc_obj = DefDiscriminator(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, norm_adj_matrix,
                                def_constraints, device, threshold=1)
    def_disc = disc_obj.train(option)

    state = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=device)
    def_cur_loc = gen_init_def_pos(config.NUM_TARGET, config.NUM_RESOURCE, def_constraints, threshold=1)
    for t, res in enumerate(def_cur_loc):
        state[(res == 1).nonzero(), 0] += int(sum(res))

    attempt_list = []
    train_start = time.time()

    gen = Def_A2C_GAN(payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                          norm_adj_matrix=norm_adj_matrix, num_feature=config.NUM_FEATURE,
                          num_resource=config.NUM_RESOURCE, def_constraints=def_constraints,
                          discriminator=def_disc, device=device).to(device)
    actor, critic, attempt = gen(state.unsqueeze(0), def_cur_loc)
    print("\nTotal Runtime:", round((time.time() - train_start) / 60, 4), "min\n")

    '''
    for i in range(100):
        start = time.time()
        print("\nGAN", i + 1)
        gen = Def_A2C_GAN(payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                          norm_adj_matrix=norm_adj_matrix, num_feature=config.NUM_FEATURE,
                          num_resource=config.NUM_RESOURCE, def_constraints=def_constraints,
                          discriminator=def_disc, device=device).to(device)
        actor, critic, attempt = gen(state.unsqueeze(0), def_cur_loc)
        attempt_list.append(attempt)
        print("\nRuntime:", round((time.time() - start) / 60, 4), "min\n")

    print("\nTotal Runtime:", round((time.time() - train_start) / 60, 4), "min\n")
    plt.figure(figsize=(20, 10))
    plt.title("# of Episodes to Generate Valid Action (0.25 threshold for invalid actions, 1000 samples)")
    plt.xlabel("GAN")
    plt.ylabel("Episodes")
    plt.plot(attempt_list)
    plt.show()
    '''
