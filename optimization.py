from __future__ import print_function

import torch
import torch.nn.functional as F
from utils import Transition, TransitionV
from sampling import gen_all_valid_actions, gen_val_mask


class Optimization(object):
    def __init__(self, batch_size, num_step, gamma, device, payoff_matrix, adj_matrix=[], norm_adj_matrix=[]):
        self.batch_size = batch_size
        self.num_step = num_step
        self.gamma = gamma
        self.device = device
        self.payoff_matrix = payoff_matrix
        self.adj_matrix = adj_matrix
        self.norm_adj_matrix = norm_adj_matrix

    def optimize_Att_A2C_LSTM(self, memory, policy_net, target_net, optimizer, lstm_hidden_size, entropy_coeff):
        if len(memory) < self.batch_size:
            return
        num_target = self.payoff_matrix.size(0)
        episodes = memory.sample(self.batch_size, self.num_step)
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []

        for episode in episodes:
            batch = Transition(*zip(*episode))
            state_batch.append(torch.cat(batch.state).unsqueeze(0))  # batch, sequence, data
            action_batch.append(torch.cat(batch.action).unsqueeze(0))
            next_state_batch.append(torch.cat(batch.next_state).unsqueeze(0))
            reward_batch.append(torch.cat(batch.reward).unsqueeze(0))

        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        next_state_batch = torch.cat(next_state_batch)
        reward_batch = torch.cat(reward_batch)

        action_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        action_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        value_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        value_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_action_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_action_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_value_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_value_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)

        loss = 0.0
        for sequence_idx in range(self.num_step):
            action_distribution_batch, state_value_batch \
                , action_hidden_state_batch, action_cell_state_batch, value_hidden_state_batch, value_cell_state_batch \
                = policy_net(state_batch[:, sequence_idx]
                             , action_hidden_state_batch, action_cell_state_batch, value_hidden_state_batch
                             , value_cell_state_batch)

            with torch.no_grad():
                _, next_state_value_batch, target_action_hidden_state_batch, target_action_cell_state_batch \
                    , target_value_hidden_state_batch, target_value_cell_state_batch \
                    = target_net(next_state_batch[:, sequence_idx],
                                 target_action_hidden_state_batch
                                 , target_action_cell_state_batch, target_value_hidden_state_batch
                                 , target_value_cell_state_batch)

            temp = next_state_batch[:, sequence_idx, :, 1].sum(dim=1).unsqueeze(dim=1)
            temp = torch.where(temp == self.num_step, 0., next_state_value_batch.type(torch.double))

            expected_state_action_value_batch = temp.type(torch.float32) * self.gamma + reward_batch[:, sequence_idx]
            advantage = (expected_state_action_value_batch - state_value_batch).squeeze(1)

            critic_loss = F.mse_loss(state_value_batch, expected_state_action_value_batch)

            log_distributions = torch.log(action_distribution_batch + 1e-10)

            temp_distributions = log_distributions * action_batch[:, sequence_idx]
            temp_distributions = temp_distributions.sum(dim=1)
            actor_loss = advantage.detach() * temp_distributions
            actor_loss = -actor_loss.mean()

            entropy_term = -(action_distribution_batch * log_distributions).sum()
            loss = loss + critic_loss + actor_loss + entropy_coeff * entropy_term / (num_target * num_target * self.batch_size)

        loss = loss / self.num_step
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def optimize_Att_A2C(self, memory, policy_net, target_net, optimizer, entropy_coeff):
        if len(memory) < self.batch_size:
            return
        num_target = self.payoff_matrix.size(0)
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        action_distribution_batch, state_value_batch \
            = policy_net(state_batch)

        with torch.no_grad():
            _, next_state_value_batch = target_net(next_state_batch)

        temp = next_state_batch[:, :, 1].sum(dim=1).unsqueeze(dim=1)
        temp = torch.where(temp == self.num_step, 0., next_state_value_batch.type(torch.double))

        expected_state_action_value_batch = temp.type(torch.float32) * self.gamma + reward_batch
        advantage = (expected_state_action_value_batch - state_value_batch).squeeze(1)

        # critic_loss = F.smooth_l1_loss(state_value_batch, expected_state_action_value_batch)
        critic_loss = F.mse_loss(state_value_batch, expected_state_action_value_batch)

        log_distributions = torch.log(action_distribution_batch + 1e-10)

        temp_distributions = log_distributions * action_batch
        temp_distributions = temp_distributions.sum(dim=1)
        actor_loss = advantage.detach() * temp_distributions
        actor_loss = -actor_loss.mean()

        entropy_term = -(action_distribution_batch * log_distributions).sum()
        loss = critic_loss + actor_loss + entropy_coeff * entropy_term / (num_target * num_target * self.batch_size)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def optimize_Def_A2C(self, memory, policy_net, target_net, optimizer, entropy_coeff):
        if len(memory) < self.batch_size:
            return
        num_target = self.payoff_matrix.size(0)
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        action_distribution_batch, state_value_batch = policy_net(state_batch)
        with torch.no_grad():
            _, next_state_value_batch = target_net(next_state_batch)

        temp = next_state_batch[:, :, 1].sum(dim=1).unsqueeze(dim=1)
        temp = torch.where(temp == self.num_step, 0., next_state_value_batch.type(torch.double))

        expected_state_action_value_batch = temp.type(torch.float32) * self.gamma + reward_batch
        advantage = expected_state_action_value_batch - state_value_batch

        critic_loss = F.mse_loss(state_value_batch, expected_state_action_value_batch)

        log_distributions = torch.log(action_distribution_batch + 1e-10)

        temp_distributions = log_distributions * action_batch
        temp_distributions = temp_distributions.sum(dim=2).sum(dim=1)
        actor_loss = advantage.detach() * temp_distributions.unsqueeze(1)
        actor_loss = -actor_loss.mean()

        entropy_term = -(action_distribution_batch * torch.log(action_distribution_batch + 1e-10)).sum()
        loss = critic_loss + actor_loss \
               + entropy_coeff * entropy_term / (num_target * num_target * self.batch_size)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
    def optimize_Def_A2C_Full(self, memory, policy_net, target_net, all_moves, def_constraints, optimizer, entropy_coeff):
        if len(memory) < self.batch_size:
            return 0
        num_target = self.payoff_matrix.size(0)
        transitions = memory.sample(self.batch_size)
        batch = TransitionV(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        mask_batch = torch.cat(batch.mask)

        action_distribution_batch, state_value_batch = policy_net(state_batch, mask_batch)
        with torch.no_grad():
            val_moves = gen_all_valid_actions(action_batch[0], def_constraints, threshold=1)
            next_mask_batch = gen_val_mask(all_moves, val_moves).unsqueeze(0)
            for state in action_batch[1:]:
                val_moves = gen_all_valid_actions(state, def_constraints, threshold=1)
                next_mask_batch = torch.cat((next_mask_batch, gen_val_mask(all_moves, val_moves).unsqueeze(0)))
            _, next_state_value_batch = target_net(next_state_batch, next_mask_batch)

        temp = next_state_batch[:, :, 1].sum(dim=1).unsqueeze(dim=1)
        temp = torch.where(temp == self.num_step, 0., next_state_value_batch.type(torch.double))

        expected_state_action_value_batch = temp.type(torch.float32) * self.gamma + reward_batch
        advantage = expected_state_action_value_batch - state_value_batch
    
        critic_loss = F.mse_loss(state_value_batch, expected_state_action_value_batch)
    
        log_distributions = torch.log(action_distribution_batch + 1e-10)

        # temp_distributions = log_distributions * action_batch
        # temp_distributions = temp_distributions.sum(dim=2).sum(dim=1)
        actor_loss = advantage.detach() # * temp_distributions.unsqueeze(1)
        actor_loss = -actor_loss.mean()

        entropy_term = -(action_distribution_batch * torch.log(action_distribution_batch + 1e-10)).sum()/(num_target*num_target*self.batch_size)
        print(critic_loss, actor_loss, entropy_term)
        loss = critic_loss + actor_loss + entropy_coeff * entropy_term

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss.item()

    def optimize_Def_A2C_LSTM(self, memory, policy_net, target_net, optimizer, lstm_hidden_size, entropy_coeff):
        if len(memory) < self.batch_size:
            return
        num_target = self.payoff_matrix.size(0)
        episodes = memory.sample(self.batch_size, self.num_step)
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []

        for episode in episodes:
            batch = Transition(*zip(*episode))
            state_batch.append(torch.cat(batch.state).unsqueeze(0))  # batch, sequence, data
            action_batch.append(torch.cat(batch.action).unsqueeze(0))
            next_state_batch.append(torch.cat(batch.next_state).unsqueeze(0))
            reward_batch.append(torch.cat(batch.reward).unsqueeze(0))

        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        next_state_batch = torch.cat(next_state_batch)
        reward_batch = torch.cat(reward_batch)

        action_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        action_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        value_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        value_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_action_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_action_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_value_hidden_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)
        target_value_cell_state_batch = torch.zeros(self.batch_size, lstm_hidden_size, device=self.device)

        loss = 0.0
        for sequence_idx in range(self.num_step):
            action_distribution_batch, state_value_batch \
                , action_hidden_state_batch, action_cell_state_batch, value_hidden_state_batch, value_cell_state_batch \
                = policy_net(state_batch[:, sequence_idx],
                             action_hidden_state_batch, action_cell_state_batch, value_hidden_state_batch,
                             value_cell_state_batch)

            with torch.no_grad():
                _, next_state_value_batch, target_action_hidden_state_batch, target_action_cell_state_batch \
                    , target_value_hidden_state_batch, target_value_cell_state_batch \
                    = target_net(next_state_batch[:, sequence_idx]
                                 , target_action_hidden_state_batch, target_action_cell_state_batch
                                 , target_value_hidden_state_batch, target_value_cell_state_batch)

            temp = next_state_batch[:, sequence_idx, :, 1].sum(dim=1).unsqueeze(dim=1)
            temp = torch.where(temp == self.num_step, 0., next_state_value_batch.type(torch.double))
            expected_state_action_value_batch = temp.type(torch.float32) * self.gamma \
                                                + reward_batch[:, sequence_idx].unsqueeze(1)
            advantage = (expected_state_action_value_batch - state_value_batch).squeeze(1)

            critic_loss = F.mse_loss(state_value_batch, expected_state_action_value_batch)
            # check = action_distribution_batch.sum(dim = 2)
            log_distributions = torch.log(action_distribution_batch + 1e-10)

            temp_distributions = log_distributions * action_batch[:, sequence_idx]
            temp_distributions = temp_distributions.sum(dim=2).sum(dim=1)
            actor_loss = advantage.detach() * temp_distributions
            actor_loss = -actor_loss.mean()

            entropy_term = -(action_distribution_batch * log_distributions).sum()
            loss = loss + critic_loss + actor_loss + entropy_coeff * entropy_term / (
                        num_target * num_target * self.batch_size)

        loss = loss / self.num_step
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
