import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.attention = nn.MultiheadAttention(args.rnn_hidden_dim, 4, batch_first=True)
        self.att = nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim)
        
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state,drop_inputs, test_mode):
        b, a, e = inputs.size()

        # inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs))
        x_, _ = self.attention(x, x, x, attn_mask=drop_inputs.repeat(4, 1, 1).bool())
        x_ = th.cat((x, x_), dim=-1)
        x_ = F.relu(self.att(x_))
        x_= x_.view(-1,self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x_, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)
    
    
    
    
