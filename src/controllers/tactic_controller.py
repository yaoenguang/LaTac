import copy

import torch
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from modules.tactic_selectors import tactic_selector as tactic_selector

from components.epsilon_schedules import DecayThenFlatSchedule

import torch.nn as nn
import torch.nn.functional as F


class EpsilonGreedyTacticSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_tactic(self, agent_inputs, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)  # eval 方法用于根据当前的训练步数或时间步长来计算当前的学习率

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "test_noise", 0.0)

        # mask actions that are excluded from selection

        _, max_indices = agent_inputs.max(dim=2)
        # 创建一个与 max_indices 形状相同的随机张量，以 epsilon 的概率设置为 1
        random_mask = (torch.rand(max_indices.shape) < self.epsilon).to(max_indices.device)
        # 如果随机数小于 epsilon，则随机选择一个战术；否则选择Q值最大的战术
        random_tactics = torch.randint(0, agent_inputs.size(2), max_indices.shape, device=max_indices.device)
        chosen_tactics = torch.where(random_mask, random_tactics, max_indices)

        return chosen_tactics


# This multi-agent controller shares parameters between agents
class TMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(TMAC, self).__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.obs_dim = int(np.prod(args.obs_shape))
        args.input_shape = self._get_input_shape(scheme)
        self.hidden_states = None
        self.tactic_num = args.tactic_num

        self.device = args.device

        self.tactic_selector = tactic_selector.Tactic_Selector(args.input_shape, args).to(self.device)
        self.tactic_selector_drop = tactic_selector.Tactic_Selector(args.input_shape, args).to(self.device)
        self.tactic_ep_sele = EpsilonGreedyTacticSelector(args)

        self.fc1 = nn.Linear(32, 128).to(self.device)
        self.fc21 = nn.Linear(128, 32).to(self.device)
        self.fc22 = nn.Linear(128, 32).to(self.device)

        self.prmlp = nn.Linear(32, args.tactic_num).to(self.device)
        self.bn = nn.BatchNorm1d(args.tactic_num).to(self.device)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, drop_prob=1):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals, drop_inputs, _, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode, drop_prob=drop_prob)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, th.squeeze(drop_inputs)[bs], self.agent_tactic_id[bs]

    def forward(self, ep_batch, t, test_mode=False, drop_prob=1, train_drop=False):
        if test_mode:
            self.agent.eval()

        agent_inputs, drop_inputs = self._build_inputs(ep_batch, t)  # agent_inputs 48*192
        agent_inputs = agent_inputs.view(-1, self.n_agents, self.obs_dim + self.n_actions + self.n_agents)  # 处理输入obs  8*6*192

        if test_mode:  # 如果是测试阶段，就是用drop的——tactic selector
            drop_inputs = self.generate_drop(drop_inputs, drop_prob)
            tactic_pr, self.tactic_hidden_drop, mha_out = self.tactic_selector_drop(agent_inputs, self.tactic_hidden_drop, drop_inputs)  # n*32
            if train_drop:
                tactic_pr = self.prmlp(tactic_pr)
                tactic_pr = F.softmax(tactic_pr, dim=2)
                return tactic_pr, mha_out

        else:
            agent_tactic_prob, self.tactic_hidden, mha_out = self.tactic_selector(agent_inputs, self.tactic_hidden, drop_inputs)

            h1 = F.relu(self.fc1(agent_tactic_prob))
            mu = self.fc21(h1)
            log_var = self.fc22(h1)
            std = th.exp(0.5 * log_var)
            eps = th.clamp(th.randn(std.shape), min=-1, max=1).to(self.device)
            tactic_pr = mu + std * eps

        tactic_pr = tactic_pr.reshape(-1, 32)
        tactic_pr = F.sigmoid(F.relu(self.bn(self.prmlp(tactic_pr)).view(agent_inputs.shape[0], self.args.n_agents, -1)))

        agent_tactic, agent_tactic_id, tactic_em = self.tactic_selector.select_tactic(tactic_pr, agent_inputs, test_mode, t)  # 选择战术对应的embeddings
        self.agent_tactic_id = agent_tactic_id

        self.tactic_now = agent_tactic
        new_agent_inputs = th.cat((agent_inputs, self.tactic_now), -1)
        agent_outs, self.hidden_states = self.agent(new_agent_inputs, self.hidden_states)

        return agent_outs, drop_inputs.view(ep_batch.batch_size, self.n_agents, -1), tactic_pr, agent_tactic, mha_out, tactic_em

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).to(self.device)  # bav
        self.tactic_hidden = self.tactic_selector.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).to(self.device)
        self.tactic_hidden_drop = self.tactic_selector.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).to(self.device)

    def generate_drop(self, drop_inputs, drop_prob):
        for i in range(0, self.n_agents):
            for j in range(0, self.n_agents):
                if (np.random.random() > drop_prob) and i != j:
                    drop_inputs[:, i, j] = 1
        return drop_inputs

    def parameters(self):
        params = list(self.agent.parameters())
        params += list(self.tactic_selector.parameters())
        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.tactic_selector.load_state_dict(other_mac.tactic_selector.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.tactic_selector.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.tactic_selector.state_dict(), "{}/tactic_selector.th".format(path))

    def load_models(self, path):

        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.tactic_selector.load_state_dict(th.load("{}/tactic_selector.th".format(path),
                                       map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        if bs < 10:
            drop_inputs = torch.zeros(bs, self.n_agents, self.n_agents).to(self.device)
            # drop_inputs = None
        else:
            drop_inputs = batch["drop_inputs"][:, t]

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs, drop_inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


# class Comm_attn(nn.Module):
#     def __init__(self, input_shape, args):
#         super(Comm_attn, self).__init__()
#         # args.tactic_dim=input_shape
#         self.args = args
#         self.device = args.device
#         self.n_agents = args.n_agents
#         self.attention = nn.MultiheadAttention(args.rnn_hidden_dim, 1, batch_first=True)  # batch_first是否将批次维度放在第一位
#         self.att = nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim)
#         self.fc_gru = nn.Linear(input_shape, args.rnn_hidden_dim)
#
#     def forward(self, inputs, drop_inputs):
#         x = F.relu(self.fc_gru(inputs))
#         x_, _ = self.attention(x, x, x, attn_mask=drop_inputs.bool())
#         x_ = th.cat((x, x_), dim=-1)
#         x_ = F.relu(self.att(x_))
#         return x_
