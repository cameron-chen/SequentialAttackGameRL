import sys
import time
import random
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import GameGeneration
from graph_convolution import GraphConvolution
from sampling import gen_samples, gen_samples_greedy
import configuration as config


class Def_Disc(nn.Module):
    def __init__(self):
        super(Def_Disc, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 4, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(1)
        self.ln = nn.Linear(6, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.sig(self.ln(x.view(-1)))
        return x


class Def_Disc_CNN(nn.Module):
    def __init__(self, num_targ, num_res):
        super(Def_Disc_CNN, self).__init__()
        num_feat = round(num_res/2)
        self.num_targ = num_targ
        self.num_res = num_res
        self.conv1 = nn.Conv2d(2, 32, num_feat+1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.ln1 = nn.Linear(32*(num_res-num_feat)
                            *(num_targ-num_feat),
                            int(64*num_targ/10))
        self.ln2 = nn.Linear(int(64*num_targ/10), 1)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(1, 2, self.num_res, self.num_targ)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.ln1(x.view(-1)))
        x = self.sig(self.ln2(x.view(-1)))
        return x


class Def_Disc_GCN(nn.Module):
    def __init__(self, num_targ, num_res, norm_adj_matrix):
        super(Def_Disc_GCN, self).__init__()
        self.adj = norm_adj_matrix
        self.gc1 = GraphConvolution(num_targ, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.gc2 = GraphConvolution(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.ln1 = nn.Linear(16*num_targ,
                            int(64*num_targ/10))
        self.ln2 = nn.Linear(int(64*num_targ/10), 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.bn1(self.gc1(x, self.adj)))
        x = self.drop(x)
        x = self.relu(self.bn2(self.gc2(x, self.adj)))
        x = self.relu(self.ln1(x.view(-1)))
        x = self.sig(self.ln2(x.view(-1)))
        return x


class DefDiscriminator(object):
    def __init__(self, num_targ, num_res, adj_matrix, norm_adj_matrix, def_constraints, device, threshold=1):
        self.num_targ = num_targ
        self.num_res = num_res
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.def_constraints = def_constraints
        self.device = device
        self.threshold = threshold

    def gen_data(self, sample_size, samples=[]):
        return gen_samples_greedy(self.num_targ, self.num_res, self.def_constraints, self.threshold, sample_size, samples)

    def update_data(self, data_set, sample_size, samples):
        add_set, samples = gen_samples_greedy(self.num_targ, self.num_res, self.def_constraints, self.threshold,
                                              sample_size, samples)
        return data_set + add_set, samples

    def train(self, option='CNN', test=None):
        print("Generating Training Data...")
        train_set, samples = self.gen_data(sample_size=5000)
        print("\nGenerating Testing Data...")
        test_set, samples = self.gen_data(sample_size=1000, samples=samples)

        if option == 'CNN':
            disc = Def_Disc_CNN(self.num_targ, self.num_res).to(self.device)
        elif option == 'GCN':
            disc = Def_Disc_GCN(self.num_targ, self.num_res, self.norm_adj_matrix).to(device)
        lr = 0.001
        disc_optim = torch.optim.Adam(disc.parameters(), lr=lr)
        criterion = nn.BCELoss()

        loss_plot = []
        running_loss = 0.0
        epoch = 0
        models = []
        top_correct = 0
        overfit = False
        limit = 3

        start = time.time()
        while not overfit:
            print("\nEpoch %d" % (epoch + 1))
            random.shuffle(train_set)
            for i, (samp, label) in enumerate(train_set):
                disc_optim.zero_grad()
                out = disc(samp)

                loss = criterion(out, label.view(1))
                loss.backward()
                disc_optim.step()

                running_loss += loss.item()
                loss_plot.append(running_loss / ((epoch * len(train_set)) + (i + 1)))
                if i % (len(train_set) / 10) == (len(train_set) / 10) - 1:
                    print("Sample %d -- Average Loss: %f" % (i + 1, loss_plot[-1]))
                    lr = lr * 0.99
                    for param_group in disc_optim.param_groups:
                        param_group['lr'] = lr

            print("\nTesting model...")
            correct = 0
            with torch.no_grad():
                for i, (samp, label) in enumerate(test_set):
                    out = disc(samp)
                    prediction = round(out.item())
                    if prediction == label:
                        correct += 1

            print("%d correctly predicted out of %d samples" % (correct, len(test_set)))

            if correct <= top_correct:
                if limit > 2:
                    print(top_correct)
                    print(limit)
                    limit -= 1
                else:
                    overfit = True
            else:
                top_correct = correct
                limit = 3

            models.append((copy.deepcopy(disc), correct / len(test_set)))

            epoch += 1

        print("\nRuntime:", round((time.time() - start) / 60, 4), "min")

        if test:
            plt.figure(figsize=(20, 10))
            plt.xlabel("Sample")
            plt.ylabel("Loss")
            plt.plot(loss_plot, label="Training Loss")
            plt.legend()
            plt.show()

        best_disc, acc = max(models, key=lambda x: x[1])
        print("Best model has a test accuracy of", acc)

        return best_disc


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


