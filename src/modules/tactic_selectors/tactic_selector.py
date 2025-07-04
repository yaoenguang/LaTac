import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

from components.epsilon_schedules import DecayThenFlatSchedule


class Tactic_Selector(nn.Module):
    def __init__(self, input_shape, args):
        super(Tactic_Selector, self).__init__()
        # args.tactic_dim=input_shape
        self.args = args
        # self.device = th.device('cuda:1')
        self.device = args.device
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.input_shape = input_shape
        self.attention = nn.MultiheadAttention(args.rnn_hidden_dim, 1, batch_first=True).to(self.device)  # batch_first是否将批次维度放在第一位
        self.att = nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim).to(self.device)
        self.commander = nn.Linear(input_shape, input_shape).to(self.device)
        self.fc_gru = nn.Linear(input_shape, args.rnn_hidden_dim).to(self.device)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim).to(self.device)  # 每次只处理一个时间步的数据,输入是二维数据batch_size*input_size
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 32).to(self.device)
        self.drop_num = args.drop_num
        self.linear = nn.Linear(input_shape * 2, input_shape).to(self.device)
        self.a = nn.Parameter(th.randn((1, args.tactic_num, input_shape)).to(self.device)).to(self.device)

        self.tactic_ep_sele = EpsilonGreedyTacticSelector(args)

    def forward(self, inputs, hidden_state, drop_inputs):
        x = F.relu(self.fc_gru(inputs))
        x_, _ = self.attention(x, x, x, attn_mask=drop_inputs.bool())
        # x_ = x_ + x
        x_ = th.cat((x, x_), dim=-1)
        x_ = F.relu(self.att(x_))
        xx_ = x_.reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(xx_, h_in)
        h = h.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)
        tactic_q = self.fc2(h)
        return tactic_q, h, xx_

    def select_tactic(self, tactic_qs, inputs, test_mode, t):
        tactic_em = torch.expand_copy(self.commander(F.relu(self.a)), (inputs.shape[0], self.args.tactic_num, self.input_shape))
        if not test_mode:
            tactic_prob = F.gumbel_softmax(tactic_qs, hard=True, dim=-1)
            tactic = th.bmm(tactic_prob, tactic_em)
            tactic_id = th.argmax(tactic_prob, dim=-1)
        else:
            tactic_id = self.tactic_ep_sele.select_tactic(tactic_qs, t, test_mode)
            tactic = tactic_em[torch.arange(inputs.shape[0])[:, None], tactic_id]

        tactic = th.cat((inputs, tactic), dim=-1)
        tactic = F.relu(self.linear(tactic))
        return tactic, tactic_id, tactic_em

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc_gru.weight.new(1, self.args.rnn_hidden_dim).zero_()


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