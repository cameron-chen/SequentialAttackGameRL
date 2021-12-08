import sys
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import GameGeneration
from graph_convolution import GraphConvolution
from defender_discriminator import DefDiscriminator
from def_act_gen import dist_est, Def_Action_Generator
from distribution_estimator import DistributionEstimator
from sampling import check_move, check_constraints, gen_init_def_pos
import configuration as config


class Def_A2C_GAN(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature,
                 num_resource, def_constraints, act_gen, discriminator, dist_estimator,
                 device):
        super(Def_A2C_GAN, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norma_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)
        self.num_resource = num_resource
        self.def_constraints = def_constraints
        self.discriminator = discriminator
        self.disc_criterion = nn.BCELoss()
        self.dist_estim_criterion = nn.MSELoss()
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

        self.act_gen = act_gen
        self.dist_estimator = dist_estimator

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state, def_cur_loc, test=0, nc=0, ent=0):
        batch_size = len(state)
        noise = torch.rand((1, state.size(1), self.noise_feat)).to(self.device)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1), noise), 2)

        x = self.relu(self.bn(self.gc1(x, self.norma_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norma_adj_matrix)))
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
        dist_estim_loss_list = []
        gen_lr = 0.001
        disc_lr = 0.001
        dist_estim_lr = 0.001
        gen_optimizer = optim.Adam(self.act_gen.parameters(), gen_lr)
        disc_optimizer = optim.Adam(self.discriminator.parameters(), disc_lr)
        dist_estim_optimizer = optim.Adam(self.dist_estimator.parameters(), dist_estim_lr)
        while not action_generated:
            gen_optimizer.zero_grad()
            act_estimates = self.act_gen(x.detach().squeeze(), def_cur_loc)
            for i in range(1000-1):
                act_estimates = torch.cat((act_estimates, self.act_gen(x.detach().squeeze(), def_cur_loc)))
            act_estimates = act_estimates.view(1000, self.num_resource, self.num_target)

            actions, act_probs, act_dist, _ = dist_est(act_estimates)

            invalid_count = 0
            invalid_act = set()
            invalid_est = []
            for i,act in enumerate(actions):
                if nc:
                    meet_constraints = True
                else:
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
                prob_ent = sum([p*(-math.log(p)) for p in act_probs])
                if not ent or prob_ent == 0:
                    prob_ent = 1
                gen_loss = self.disc_criterion(inval_out, true_labels)/prob_ent # /(len(act_dist.values())**2)
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
                    disc_attempt += 1
            else:
                print("\nAttempts:", attempt)
                print("Invalid Samples:", invalid_count)
                print("Actions:", len(act_dist.values()), "\n")
                action_generated = True

            gen_lr = gen_lr * 0.99
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] = gen_lr

            if attempt > 25 and invalid_count >= invalid_list[-2]:
                prob = torch.tensor(1/len(act_dist.values()))
                if test:
                    return 0, state_value, prob, attempt, len(act_dist.values())
                else:
                    return 0, state_value, prob

        # Update Distribution Estimator
        dist_estim_attempt = 1
        dist_estim_lr = 0.001
        distribution_check = False
        # act_estimates = torch.cat((act_estimates, act_estimates)).view(1000, self.num_resource, self.num_target)
        # target_probs = act_probs + act_probs
        while not distribution_check:
            dist_estim_optimizer.zero_grad()
            dist_estimates = self.dist_estimator(act_estimates.detach().unsqueeze(0))

            dist_estim_loss = self.dist_estim_criterion(dist_estimates, act_probs)
            dist_estim_loss_list.append(dist_estim_loss.item())
            if test: print("Distribution Estimator Loss:", dist_estim_loss.item())

            dist_estim_loss.backward()
            dist_estim_optimizer.step()

            if dist_estim_loss < 0.05:
                distribution_check = True
            elif dist_estim_attempt > 25:
                distribution_check = True

            dist_estim_lr = dist_estim_lr * 0.95
            for param_group in dist_estim_optimizer.param_groups:
                param_group['lr'] = dist_estim_lr

            dist_estim_attempt += 1

        # Select action with distribution estimate
        for i, act in enumerate(actions):
            if act not in invalid_act:
                select_act = act
                select_prob = dist_estimates[i]
                if test:
                    print("Estimated Probability:", select_prob.item())
                    print("Actual Probability:", act_probs[i].item())
                break

        if test:
            return select_act, state_value, select_prob, attempt, len(act_dist.values())
        else:
            return select_act, state_value, select_prob


if __name__ == '__main__':
    option = 'CNN'
    if len(sys.argv) > 2:
        if 'GCN' in sys.argv[1]:
            option = 'GCN'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_gen = GameGeneration(num_target=config.NUM_TARGET, graph_type='random_scale_free',
                              num_res=config.NUM_RESOURCE, device=device)
    payoff_matrix, adj_matrix, norm_adj_matrix, def_constraints = game_gen.gen_game()
    # def_constraints = [[0, 2], [1, 3], [4]]

    disc_obj = DefDiscriminator(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, norm_adj_matrix,
                                def_constraints, device, threshold=1)
    def_disc = disc_obj.train(episodes=1600, option=option)

    dist_estim_obj = DistributionEstimator(config.NUM_TARGET, config.NUM_RESOURCE, config.NUM_FEATURE, payoff_matrix,
                                            adj_matrix, norm_adj_matrix, def_constraints, device)
    # dist_estimator = dist_estim_obj.train()
    dist_estimator = dist_estim_obj.initial()

    state = torch.zeros(config.NUM_TARGET, 2, dtype=torch.int32, device=device)
    def_cur_loc = gen_init_def_pos(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, def_constraints, threshold=1)
    for t, res in enumerate(def_cur_loc):
        state[(res == 1).nonzero(), 0] += int(sum(res))

    attempt_list = []
    action_num_list = []
    train_start = time.time()

    for i in range(100):
        start = time.time()
        print("\nGAN", i + 1)
        act_gen = Def_Action_Generator(config.NUM_TARGET, config.NUM_RESOURCE, adj_matrix, device).to(device)
        gen = Def_A2C_GAN(payoff_matrix=payoff_matrix, adj_matrix=adj_matrix,
                          norm_adj_matrix=norm_adj_matrix, num_feature=config.NUM_FEATURE,
                          num_resource=config.NUM_RESOURCE, def_constraints=def_constraints,
                          act_gen=act_gen, discriminator=def_disc, dist_estimator=dist_estimator, 
                          device=device).to(device)
        actor, critic, prob, attempt, num_actions = gen(state.unsqueeze(0), def_cur_loc, test=1)
        j = 1
        while not torch.is_tensor(actor):
            print("\nGAN", i+1, "redo", j)
            actor, critic, prob, attempt, num_actions = gen(state.unsqueeze(0), def_cur_loc, test=1)
            j += 1
        attempt_list.append(attempt)
        action_num_list.append(num_actions)
        print("Action:", actor.tolist())
        print("State Value:", critic.item())
        print("\nRuntime:", round((time.time() - start) / 60, 4), "min\n")

    print("\nTotal Runtime:", round((time.time() - train_start) / 60, 4), "min\n")
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
    plt.plot(action_num_list)
    plt.show()
