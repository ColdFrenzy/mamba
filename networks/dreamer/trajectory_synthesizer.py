import torch
import random
import torch.nn as nn
import numpy as np


class TrajectorySynthesizerRNN(nn.Module):
    """Trajectory Synthesizer implemented through an LSTM network.
    """
    def __init__(
        self, action_size, deter_size, stoch_size, horizon, hidden_size, num_layers, activation=nn.ELU, dropout=0.1, 
        ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size

        self.horizon = horizon
        self.embedding_size = deter_size + stoch_size + action_size
        # As in the transformer we keep the same dimension as output
        self.output_size = self.embedding_size
        self.act_fn = activation
        self.dropout = dropout
        self.n_layers = num_layers
        self.hidden = hidden_size

        self.model = self._build_model()

    def _build_model(self):

        model = [nn.LSTM(self.embedding_size, self.hidden, num_layers=self.n_layers,  batch_first=False, dropout=self.dropout, bidirectional=False)]
        model += [nn.Linear(self.hidden, self.output_size)]
        model += [self.act_fn()]
        return nn.Sequential(*model)

    def forward(self, model_state):
        """        
        LSTM expects input of shape (seq_len, batch_size, input_size) and outputs a tuple (seq_len, batch_size, hidden_size)
        :params model_state: tensor of shape (horizon, seq_len*batch_size, num_agents, deter_size + stoch_size + action_size)
        :return output: tensor of shape (seq_len*batch_size, deter_size + stoch_size + action_size)
        """
        output, (h_n, c_n) = self.model[0](model_state)
        # last element of the output contains info about all the sequence
        output = self.model[1:](output[-1])
        return output


class TrajectorySynthesizerAtt(nn.Module):
    """Trajectory Synthesizer implemented through a Transformer network.
    """
    def __init__(
        self, action_size, deter_size, stoch_size, horizon, hidden_size, num_layers, n_heads = 8, activation=nn.ELU, dropout=0.1, 
        ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.horizon = horizon
        self.act_fn = activation
        self.dropout = dropout
        self.n_layers = num_layers
        self.hidden =  hidden_size
        self.n_heads = n_heads
        self.embedding_size = deter_size + stoch_size + action_size

        self.pos_enc  = PositionalEncoding(self.horizon, self.embedding_size)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=5,
                                                                        dim_feedforward=self.hidden,
                                                                        dropout=self.dropout, batch_first=False), num_layers=self.n_layers)
        self.fc = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, model_state):
        """One way to use transformers as seq-to-one is to average the results
        :params model_state: tensor of shape (horizon, seq_len*batch_size, deter_size + stoch_size + action_size)
        :return output: tensor of shape (seq_len*batch_size, deter_size + stoch_size + action_size)
        """
        model_state = self.pos_enc(model_state)
        output = self.transformer(model_state)
        output = torch.mean(output, dim=0)
        output = self.fc(output)
        return output


class PositionalEncoding(nn.Module):
    __author__ = "Yu-Hsiang Huang"

    def __init__(self, max_len, d_hid):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(max_len, d_hid))

    def _get_sinusoid_encoding_table(self, max_len, d_hid):
        """ Sinusoid position encoding table of dimension (max_len, d_hid)"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(max_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)

    def forward(self, x):
        """ Input x of shape (seq_len, batch_size, d_hid)"""
        return x + self.pos_table[:x.size(0), :].detach().clone().unsqueeze(1)


if __name__ == "__main__":
    num_strategies = 3
    horizon = 5
    stoch_size = 1024
    deter_size = 256
    action_size = 5
    embedding_size = 256
    batch_size = 20
    input_tensor = torch.rand(num_strategies, horizon, batch_size,stoch_size + deter_size + action_size)
    traj_embed = []
    ts_info={"type": "rnn", # rnn or attention
            "num_layers": 8,
            "node_size": 400,
            "activation": nn.ELU,
            "n_heads": 8, # only for "attention"
            "dropout": 0.1,
        }
    if ts_info["type"] == "rnn":
        model = TrajectorySynthesizerRNN(action_size, deter_size, stoch_size, horizon, ts_info=ts_info)
        for tj in range(num_strategies):
            traj_embed.append(model(input_tensor[tj]))
        traj_embed = torch.stack(traj_embed, dim=0)
    elif ts_info["type"] == "attention":
        model = TrajectorySynthesizerAtt(action_size, deter_size, stoch_size, horizon, ts_info=ts_info)
        for tj in range(num_strategies):
            traj_embed.append(model(input_tensor[tj]))
        traj_embed = torch.stack(traj_embed, dim=0)
    print("done")