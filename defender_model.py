from __future__ import print_function

import torch
import torch.nn as nn

from graph_convolution import GraphConvolution
import configuration as config


class Def_A2C_GCN(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature):
        super(Def_A2C_GCN, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norma_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)

        self.gc1 = GraphConvolution(num_feature, 32)
        self.bn = nn.BatchNorm1d(self.num_target)

        self.gc2 = GraphConvolution(32, 16)

        self.ln1 = nn.Linear(16, 8)
        self.ln_actor1 = nn.Linear(8 * self.num_target, 32)
        self.ln_actor2 = nn.Linear(32, self.num_target)
        self.softmax_actor = nn.Softmax(dim=2)

        self.ln_value1 = nn.Linear(8 * self.num_target, 32)
        self.ln_value2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state):
        batch_size = len(state)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1)), 2)

        x = self.relu(self.bn(self.gc1(x, self.norma_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norma_adj_matrix)))
        x = self.relu(self.ln1(x))
        x = x.view(-1, 8 * self. num_target)

        # Policy

        temp = x.unsqueeze(1).repeat(1, self.num_target, 1)
        action_distribution = self.relu(self.ln_actor1(temp))
        temp = torch.where(self.adj_matrix == config.MIN_VALUE, -1000.0, 0.0).float()
        temp = temp.unsqueeze(0).repeat(batch_size, 1, 1)
        action_distribution = self.softmax_actor(self.ln_actor2(action_distribution) + temp)

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        return action_distribution, state_value


class Def_A2C_GCN_Full(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature):
        super(Def_A2C_GCN_Full, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norma_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)
        self.num_res = int(self.num_target/2)

        self.gc1 = GraphConvolution(num_feature, 32)
        self.bn = nn.BatchNorm1d(self.num_target)

        self.gc2 = GraphConvolution(32, 16)

        self.ln_actor1 = nn.Linear(16*self.num_target, 24*self.num_target)
        self.ln_actor2 = nn.Linear(24*self.num_target, self.num_target**self.num_res)
        self.softmax_actor = nn.Softmax(dim=2)
        self.softmax_nomask = nn.Softmax(dim=0)

        self.ln_value1 = nn.Linear(16 * self.num_target, 32)
        self.ln_value2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state, validity_mask=torch.tensor(0)):
        batch_size = len(state)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1)), 2)

        x = self.relu(self.bn(self.gc1(x, self.norma_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norma_adj_matrix)))
        x = x.view(-1, 16 * self.num_target)

        # Policy

        temp = x.unsqueeze(1).repeat(1, 1, 1)
        action_distribution = self.relu(self.ln_actor1(temp)).squeeze()
        if validity_mask.size()[0] > 0:
            temp = torch.where(validity_mask == 0, -9999.0, 0.0).float()
            temp = temp.unsqueeze(0).repeat(batch_size, 1, 1)
            action_distribution = self.softmax_actor(self.ln_actor2(action_distribution) + temp).squeeze()
        else:
            action_distribution = self.softmax_nomask(self.ln_actor2(action_distribution)).squeeze()

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        return action_distribution, state_value


class Def_A2C_GCN_LSTM(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature, lstm_hidden_size):
        super(Def_A2C_GCN_LSTM, self).__init__()
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.lstm_hidden_size = lstm_hidden_size
        self.num_target = payoff_matrix.size(dim=0)

        self.gc1 = GraphConvolution(num_feature, 32)
        self.bn = nn.BatchNorm1d(self.num_target)

        self.gc2 = GraphConvolution(32, 16)

        self.ln1 = nn.Linear(16, 8)

        # self.ln_actor1 = nn.Linear(32, 16)
        # self.ln_actor2 = nn.Linear(16, NUM_TARGET)
        self.lstm_actor = nn.LSTMCell(8 * self.num_target, self.lstm_hidden_size)
        self.ln_actor1 = nn.Linear(self.lstm_hidden_size, 32)
        self.ln_actor2 = nn.Linear(32, self.num_target)
        self.softmax_actor = nn.Softmax(dim=2)

        self.lstm_critic = nn.LSTMCell(8 * self.num_target, self.lstm_hidden_size)
        self.ln_critic1 = nn.Linear(self.lstm_hidden_size, 32)
        self.ln_critic2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state
                , action_hidden_state, action_cell_state, value_hidden_state, value_cell_state):
        batch_size = len(state)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1)), 2)

        x = self.relu(self.bn(self.gc1(x, self.norm_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norm_adj_matrix)))
        x = self.relu(self.ln1(x))
        x = x.view(-1, 8 * self.num_target)

        # Policy
        action_next_hidden_state, action_next_cell_state = self.lstm_actor(x, (action_hidden_state, action_cell_state))
        temp = action_next_hidden_state.unsqueeze(1).repeat(1, self.num_target, 1)
        action_distribution = self.relu(self.ln_actor1(temp))
        # action_distribution = self.relu(self.ln_actor1(x))
        temp = self.adj_matrix.clone()
        temp = torch.where(temp == config.MIN_VALUE, -1000.0, 0.0).float()
        action_distribution = self.ln_actor2(action_distribution) + temp.unsqueeze(0).repeat(batch_size, 1, 1)
        action_distribution = self.softmax_actor(action_distribution)

        # Value
        # x = x.view(-1, 32 * NUM_TARGET)
        value_next_hidden_state, value_next_cell_state = self.lstm_critic(x, (value_hidden_state, value_cell_state))
        state_value = self.relu(self.ln_critic1(value_next_hidden_state))
        state_value = self.ln_critic2(state_value)

        return action_distribution, state_value\
            , action_next_hidden_state, action_next_cell_state, value_next_hidden_state, value_next_cell_state


class Def_A2C_GCN_Gen(nn.Module):
    def __init__(self, payoff_matrix, adj_matrix, norm_adj_matrix, num_feature, num_resource, device):
        super(Def_A2C_GCN_Gen, self).__init__()
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

        self.ln1 = nn.Linear(16, 8)
        self.ln_actor1 = nn.Linear(8 * self.num_target, 32)
        self.ln_actor2 = nn.Linear(32, self.num_target)
        self.softmax_actor = nn.Softmax(dim=2)

        self.ln_value1 = nn.Linear(8 * self.num_target, 32)
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
        x = self.relu(self.ln1(x))
        x = x.view(-1, 8 * self.num_target)

        # Policy
        # Output is a (k x t) matrix where k is the number of resources and t is the number of targets
        # Each row of the actor output is the set of probabilities the corresponding resource moves to each target
        temp = x.unsqueeze(1).repeat(1, self.num_resource, 1)
        action_distribution = self.relu(self.ln_actor1(temp))
        action_distribution = self.softmax_actor(self.ln_actor2(action_distribution))

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        return action_distribution, state_value