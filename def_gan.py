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

    return actions, torch.tensor(act_probs, device=device), act_dist


class Def_Action_Generator(nn.Module):
    def __init__(self, device):
        super(Def_Action_Generator, self).__init__()
        self.l1 = nn.Linear(320, 150)
        self.l2 = nn.Linear(150, 100)
        self.l3 = nn.Linear(100, 50)
        self.bn = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        noise = torch.rand(240).to(self.device)
        x = torch.cat((x, noise))
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.sig(self.bn(self.l3(x).view(5, 10)))
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

        self.ln_value1 = nn.Linear(8 * self.num_target, 32)
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
        x = self.relu(self.ln1(x))
        x = x.view(-1, 8 * self.num_target)

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        # Action Generation
        action_generated = False
        attempt = 1
        invalid_list = []
        loss_list = []
        lr = 0.001
        gen_optimizer = optim.Adam(self.act_gen.parameters(), lr)
        while not action_generated:
            gen_optimizer.zero_grad()
            act_estimates = []
            for i in range(1000):
                act_estimates.append(self.act_gen(x.squeeze().detach()))

            actions, act_probs, act_dist = dist_est(act_estimates)

            invalid_count = 0
            invalid_act = set()
            invalid_est = []
            for i, act in enumerate(actions):
                meet_constraints = False
                val = check_move(def_cur_loc, act, self.threshold)
                if val:
                    meet_constraints = check_constraints(act, self.def_constraints, self.threshold)
                if not meet_constraints:
                    invalid_count += 1
                    invalid_act.update(act)
                    invalid_est.append(act_estimates[i])

            if invalid_count > (len(actions) * 0.25):  # Threshold: 25% invalid actions
                # disc_loss = 0
                for i,act_est in enumerate(act_estimates):
                    inval_samp = torch.cat((def_cur_loc, act_est))
                    # inval_out = self.discriminator(inval_samp)
                    # disc_loss += self.disc_criterion(inval_out, torch.tensor(1, dtype=torch.float, device=device).view(1))
                    if i < 1:
                        inval_out = self.discriminator(inval_samp)
                    else:
                        inval_out = torch.cat((inval_out, self.discriminator(inval_samp)))
                true_labels = torch.ones(inval_out.size()).to(self.device)
                disc_loss = self.disc_criterion(inval_out, true_labels)
                print("\n#", attempt)
                print("Invalid Samples:", invalid_count)
                print("Loss:", disc_loss.item())
                print("Action Count:", len(act_dist.values()))
                disc_loss.backward() # retain_graph=True)
                gen_optimizer.step()

                attempt += 1
                invalid_list.append(invalid_count)
                loss_list.append(disc_loss.item())
            else:
                action_generated = True

            '''
            lr = lr * 0.95
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] = lr
            '''

            if action_generated:
                plt.figure(figsize=(20, 10))
                plt.title("# of Invalid Samples")
                plt.xlabel("Attempt")
                plt.ylabel("Invalid Samples")
                plt.plot(invalid_list)
                plt.show()

                plt.figure(figsize=(20, 10))
                plt.title("Discriminator Loss")
                plt.xlabel("Attempt")
                plt.ylabel("Loss")
                plt.plot(loss_list, color='orange')
                plt.show()

            if attempt == 25 and invalid_count > len(act_estimates) * 0.8:
                return (0, 0), 0, attempt
            elif attempt == 50:
                return (0, 0), 0, attempt


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
