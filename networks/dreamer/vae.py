import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.dreamer.utils import build_model


class Decoder(nn.Module):

    def __init__(self, embed, hidden, out_dim, layers=2):
        super().__init__()
        self.fc1 = build_model(embed, hidden, layers, hidden, nn.ReLU)
        self.rgb_input = type(out_dim) is tuple
        if type(out_dim) is tuple:
            self.c, self.h, self.w = out_dim
            self.d = 16
            self.k = 3
            activation = nn.ELU
            conv1_shape = conv_out_shape(out_dim[1:], 0, self.k, 1)
            # conv2_shape = conv_out_shape(conv1_shape, 0, self.k, 1)
            # conv3_shape = conv_out_shape(conv2_shape, 0, self.k, 1)
            # self.conv_shape = (4 * self.d, *conv3_shape)
            self.conv_shape = (self.d, *conv1_shape)
            self.fc2 = nn.Linear(hidden, np.prod(self.conv_shape))
            self.output_shape = out_dim
            self.deconv = nn.Sequential(
                # nn.ConvTranspose2d(4 * d, 2 * d, k, 1),
                # activation(),
                # nn.ConvTranspose2d(2 * d, d, k, 1),
                # activation(),
                nn.ConvTranspose2d(self.d, self.c, self.k, 1),
            )
        else:
            self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, z):
        if self.rgb_input:
            batch_shape = z.shape[:-1]
            embed_size = z.shape[-1]
            squeezed_size = np.prod(batch_shape).item()
            x = z.reshape(squeezed_size, embed_size)
            x = F.relu(self.fc1(x))
            y = F.relu(self.fc2(x))
            y = torch.reshape(y, (squeezed_size, *self.conv_shape))
            y = self.deconv(y)
            y = torch.reshape(y, (*batch_shape, *self.output_shape))
        else:
            
            x = F.relu(self.fc1(z))
            y = self.fc2(x)
        return y, x


class Encoder(nn.Module):

    def __init__(self, in_dim, hidden, embed, layers=2):
        super().__init__()
        self.shape = in_dim
        self.rgb_input = type(in_dim) is tuple
        if type(in_dim) is tuple:
            self.d = 16
            self.k = 3
            activation = nn.ELU
            self.convolutions = nn.Sequential(
                nn.Conv2d(in_dim[0], self.d, self.k),
                activation(),
                # nn.Conv2d(self.d, 2 * self.d, self.k),
                # activation(),
                # nn.Conv2d(2 * self.d, 4 * self.d, self.k),
                # activation(),
            )
            self.fc1 = nn.Linear(self.embed_size, hidden)
        else:
            self.fc1 = nn.Linear(in_dim, hidden)
        self.encoder = build_model(hidden, embed, layers, hidden, nn.ReLU)

    def forward(self, x):
        if self.rgb_input:
            batch_shape = x.shape[:-3]
            img_shape = x.shape[-3:]
            x = self.convolutions(x.reshape(-1, *img_shape))
            x = torch.reshape(x, (*batch_shape, -1))
        embed = F.relu(self.fc1(x))
        return self.encoder(F.relu(embed))

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, self.k, 1)
        # conv2_shape = conv_out_shape(conv1_shape, 0, self.k, 1)
        # conv3_shape = conv_out_shape(conv2_shape, 0, self.k, 1)
        # embed_size = int(4 * self.d * np.prod(conv3_shape).item())
        embed_size = int(self.d * np.prod(conv1_shape).item())
        return embed_size


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)