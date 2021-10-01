from __future__ import print_function

import torch
import torch.nn as nn

from graph_convolution import GraphConvolution


# GCN for the attacker oracle
class Att_A2C_GCN(nn.Module):
    def __init__(self, payoff_matrix, norm_adj_matrix, num_feature):
        super(Att_A2C_GCN, self).__init__()
        # num_feature = 6  # state (2), defender action (NUM_TARGET), payoffs (4)
        self.payoff_matrix = payoff_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)

        self.gc1 = GraphConvolution(num_feature, 64)
        self.gc2 = GraphConvolution(64, 64)

        self.ln1 = nn.Linear(64, 32)

        self.ln_actor1 = nn.Linear(32 * self.num_target, 16)
        self.ln_actor2 = nn.Linear(16, self.num_target)
        self.softmax_actor = nn.Softmax(dim=1)

        self.ln_value1 = nn.Linear(32 * self.num_target, 32)
        self.ln_value2 = nn.Linear(32, 1)

        self.bn = nn.BatchNorm1d(self.num_target)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state):
        batch_size = len(state)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1)), 2)

        x = self.relu(self.bn(self.gc1(x, self.norm_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norm_adj_matrix)))
        x = self.relu(self.ln1(x))
        x = x.view(-1, 32 * self.num_target)
        # Policy
        action_distribution = self.relu(self.ln_actor1(x))

        temp = torch.where(state[:, :, 1] == 1, -1000.0, 0.0)
        action_distribution = self.softmax_actor(self.ln_actor2(action_distribution) + temp)

        # Value
        state_value = self.relu(self.ln_value1(x))
        state_value = self.ln_value2(state_value)

        return action_distribution, state_value


# GCN for the attacker oracle
class Att_A2C_GCN_LSTM(nn.Module):
    def __init__(self, payoff_matrix, norm_adj_matrix, num_feature, lstm_hidden_size):
        super(Att_A2C_GCN_LSTM, self).__init__()
        # num_feature = 6  # state (2), defender action (NUM_TARGET), payoffs (4)
        self.payoff_matrix = payoff_matrix
        self.norm_adj_matrix = norm_adj_matrix
        self.num_target = payoff_matrix.size(dim=0)

        self.gc1 = GraphConvolution(num_feature, 64)
        self.gc2 = GraphConvolution(64, 64)

        self.ln1 = nn.Linear(64, 32)

        self.lstm_actor = nn.LSTMCell(32 * self.num_target, lstm_hidden_size)
        self.ln_actor1 = nn.Linear(lstm_hidden_size, 16)
        self.ln_actor2 = nn.Linear(16, self.num_target)
        self.softmax_actor = nn.Softmax(dim=1)

        self.lstm_value = nn.LSTMCell(32 * self.num_target, lstm_hidden_size)
        self.ln_value1 = nn.Linear(lstm_hidden_size, 16)
        self.ln_value2 = nn.Linear(16, 1)

        self.bn = nn.BatchNorm1d(self.num_target)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # action batch: size of BATCH_SIZE * NUM_TARGET * NUM_TARGET
    def forward(self, state, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state):
        batch_size = len(state)
        x = torch.cat((state, self.payoff_matrix.unsqueeze(0).repeat(batch_size, 1, 1)), 2)

        x = self.relu(self.bn(self.gc1(x, self.norm_adj_matrix)))
        x = self.dropout(x)
        x = self.relu(self.bn(self.gc2(x, self.norm_adj_matrix)))
        x = self.relu(self.ln1(x))
        x = x.view(-1, 32 * self.num_target)
        # Policy

        actor_next_hidden_state, actor_next_cell_state = self.lstm_actor(x, (action_hidden_state, action_cell_state))
        action_distribution = self.relu(self.ln_actor1(actor_next_hidden_state))

        temp = torch.where(state[:, :, 1] == 1, -1000.0, 0.0)
        action_distribution = self.softmax_actor(self.ln_actor2(action_distribution) + temp)

        # Value
        critic_next_hidden_state, critic_next_cell_state = self.lstm_value(x, (value_hidden_state, value_cell_state))
        state_value = self.relu(self.ln_value1(critic_next_hidden_state))
        state_value = self.ln_value2(state_value)

        return action_distribution, state_value\
            , actor_next_hidden_state, actor_next_cell_state, critic_next_hidden_state, critic_next_cell_state
