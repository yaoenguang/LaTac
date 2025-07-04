# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch as th
#
#
# class Tactic_Selector(nn.Module):
#     def __init__(self, input_shape, args):
#         super(Tactic_Selector, self).__init__()
#         # args.tactic_dim=input_shape
#         self.args = args
#         self.device = args.device
#         self.n_agents = args.n_agents
#         self.n_actions = args.n_actions
#         self.input_shape = input_shape
#         self.tactic_cls = nn.Parameter(th.randn((1, 1, input_shape)).to(self.device))
#         self.fc1 = nn.Linear(input_shape * 2, input_shape)
#         self.attention = nn.MultiheadAttention(input_shape, 1, batch_first=True)
#         self.commander = nn.Linear(input_shape, input_shape)
#         self.fc_gru = nn.Linear(input_shape * 2, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.tactic_num)
#         self.drop_num = args.drop_num
#         # self.linear_c = nn.Linear(input_shape*self.n_agents, input_shape)
#         # self.Commander = None
#         self.linear = nn.Linear(input_shape * 2, input_shape)
#         # self.a = F.one_hot(torch.tensor([q for q in range(self.args.tactic_num)])).float().to(self.device)
#         self.a = nn.Parameter(th.randn((1, args.tactic_num, input_shape)).to(self.device))
#
#     def forward(self, inputs, hidden_state, drop_inputs, t, test_mode=False):
#         if drop_inputs is None:
#             drop_inputs = torch.ones(inputs.shape[0], self.n_agents, 1).to(self.device)
#             if test_mode:
#                 for i in range(0, self.n_agents):
#                     if np.random.random() < self.drop_num:
#                         drop_inputs[:, i, :] = 0
#         x_t = inputs
#         mask = torch.abs(drop_inputs.squeeze(-1).unsqueeze(1) - 1)
#         tactic_cls = torch.expand_copy(self.tactic_cls, (inputs.shape[0], 1, self.input_shape))
#         # x_t = th.cat((x_t, tactic_cls), dim=1)
#         x_t, _ = self.attention(tactic_cls, x_t, x_t, attn_mask=mask)
#         x_t = th.expand_copy(x_t, (inputs.shape[0], self.n_agents, self.input_shape)) * drop_inputs
#         x_t = th.cat((inputs, x_t), dim=-1)
#         x_t = F.relu(self.fc1(x_t))
#         x_t = th.cat((inputs, x_t), dim=-1)
#         x = F.relu(self.fc_gru(x_t))
#         x = x.reshape(-1, self.args.rnn_hidden_dim)
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         h = h.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)
#         tactic_q = self.fc2(h)
#
#         return tactic_q, h, drop_inputs
#
#     def select_tactic(self, tactic_qs, inputs):
#         tactic = torch.expand_copy(F.relu(self.commander(F.relu(self.a))),
#                                    (inputs.shape[0], self.args.tactic_num, self.input_shape))
#         tactic_prob = F.gumbel_softmax(tactic_qs, hard=True, dim=-1)
#         tactic = th.bmm(tactic_prob, tactic)
#         tactic = th.cat((inputs, tactic), dim=-1)
#         tactic = F.relu(self.linear(tactic))
#         return tactic
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc_gru.weight.new(1, self.args.rnn_hidden_dim).zero_()
import time

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch as th
#
# class Tactic_Selector(nn.Module):
#     def __init__(self, input_shape, args):
#         super(Tactic_Selector, self).__init__()
#         # args.tactic_dim=input_shape
#         self.args = args
#         self.device = args.device
#         self.n_agents = args.n_agents
#         self.n_actions = args.n_actions
#         self.input_shape = input_shape
#         self.tactic_cls = nn.Parameter(th.randn((1, 1, args.rnn_hidden_dim)).to(self.device))
#         self.attention = nn.MultiheadAttention(args.rnn_hidden_dim, 4, batch_first=True)
#         self.commander = nn.Linear(args.tactic_dim, args.tactic_dim)
#         self.fc_gru = nn.Linear(input_shape, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc = nn.Linear(args.rnn_hidden_dim + args.tactic_dim, args.tactic_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim+args.tactic_dim, args.tactic_num)
#         self.drop_num = args.drop_num
#         self.linear_c = nn.Linear(args.rnn_hidden_dim, args.tactic_dim)
#         self.a = nn.Parameter(th.randn((1, args.tactic_num, args.tactic_dim)).to(self.device))
#     def forward(self, inputs, hidden_state, drop_inputs, t, test_mode=False):
#         if drop_inputs is None:
#             drop_inputs = torch.zeros(inputs.shape[0], self.n_agents, 1).to(self.device)
#             #if not test_mode:
#             for i in range(0, self.n_agents):
#                 if np.random.random() > self.drop_num:
#                     drop_inputs[:, i, :] = 1
#
#         x = F.relu(self.fc_gru(inputs))
#         x = x.reshape(-1, self.args.rnn_hidden_dim)
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         h = h.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)
#         x_t = h * drop_inputs
#
#         tactic_cls = torch.expand_copy(self.tactic_cls, (inputs.shape[0], 1, self.args.rnn_hidden_dim))
#         x_t = th.cat((x_t, tactic_cls), dim=1)
#         x_t, _ = self.attention(tactic_cls, x_t, x_t)
#         x_t = self.linear_c(F.relu(x_t))
#         x_t = th.expand_copy(x_t, (inputs.shape[0], self.n_agents, self.args.tactic_dim))*drop_inputs
#
#         # tactic_cls = torch.expand_copy(self.tactic_cls, (inputs.shape[0], 1, self.args.rnn_hidden_dim))
#         # x_t, _ = self.attention(tactic_cls, x_t, x_t)
#         # x_t = th.expand_copy(x_t, (inputs.shape[0], self.n_agents, self.args.tactic_dim))*drop_inputs
#
#         x_t = th.cat((h, x_t), dim=-1)
#         x_t=F.relu(self.fc(x_t))
#         x_t = th.cat((h, x_t), dim=-1)
#         tactic_q = self.fc2(x_t)
#
#         return tactic_q, h, drop_inputs
#
#     def select_tactic(self, tactic_qs, inputs):
#         tactic = torch.expand_copy(F.relu(self.commander(F.relu(self.a))), (inputs.shape[0], self.args.tactic_num, self.args.tactic_dim))
#         tactic_prob = F.gumbel_softmax(tactic_qs, hard=True, dim=-1)
#         tactic = th.bmm(tactic_prob, tactic)
#         return tactic
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc_gru.weight.new(1, self.args.rnn_hidden_dim).zero_()


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class Tactic_Selector(nn.Module):
    def __init__(self, input_shape, args):
        super(Tactic_Selector, self).__init__()
        # args.tactic_dim=input_shape
        self.args = args
        self.device = args.device
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.input_shape = input_shape
        self.attention = nn.MultiheadAttention(args.rnn_hidden_dim, 1, batch_first=True)
        self.att = nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim)
        self.commander = nn.Linear(input_shape, input_shape)
        self.fc_gru = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.tactic_num)
        self.drop_num = args.drop_num
        self.linear = nn.Linear(input_shape * 2, input_shape)
        self.a = nn.Parameter(th.randn((1, args.tactic_num, input_shape)).to(self.device))

    def forward(self, inputs, hidden_state, drop_inputs, t, test_mode=False):
        if drop_inputs is None:
            drop_inputs = torch.zeros(inputs.shape[0], self.n_agents, self.n_agents).to(self.device)
            # if test_mode:
            for i in range(0, self.n_agents):
                for j in range(0, self.n_agents):
                    if (np.random.random() > self.drop_num) and i != j:
                        drop_inputs[:, i, j] = 1
        x = F.relu(self.fc_gru(inputs))
        x_, _ = self.attention(x, x, x, attn_mask=drop_inputs.bool())
        x_ = th.cat((x, x_), dim=-1)
        x_ = F.relu(self.att(x_))
        x_ = x_.reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x_, h_in)
        h = h.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)
        tactic_q = self.fc2(h)
        return tactic_q, h, drop_inputs

    def select_tactic(self, tactic_qs, inputs):
        tactic = torch.expand_copy(self.commander(F.relu(self.a)), (inputs.shape[0], self.args.tactic_num, self.input_shape))
        # tactic = torch.expand_copy(F.relu(self.a), (inputs.shape[0], self.args.tactic_num, self.input_shape))
        tactic = tactic[torch.arange(inputs.shape[0])[:, None], tactic_qs]
        # tactic_prob = F.gumbel_softmax(tactic_qs, hard=True, dim=-1)
        # tactic = th.bmm(tactic_prob, tactic)
        tactic = th.cat((inputs, tactic), dim=-1)
        tactic = F.relu(self.linear(tactic))
        return tactic

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc_gru.weight.new(1, self.args.rnn_hidden_dim).zero_()
