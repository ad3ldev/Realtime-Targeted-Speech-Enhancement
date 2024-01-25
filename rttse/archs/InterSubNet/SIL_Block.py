import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


EPSILON = np.finfo(np.float32).eps

class subband_interaction(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(subband_interaction, self).__init__()
        """
        Subband Interaction Module
        """

        self.input_linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.PReLU()
        )
        self.mean_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU()
        )
        self.output_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, input_size),
            nn.PReLU()
        )
        self.norm = nn.GroupNorm(1, input_size)

    def forward(self, input):
        """
        input: [B, F, F_s, T]
        """
        B, G, N, T = input.shape

        # Transform
        group_input = input  # [B, F, F_s, T]
        group_input = group_input.permute(0, 3, 1, 2).contiguous().view(-1, N)  # [B * T * F, F_s]
        group_output = self.input_linear(group_input).view(B, T, G, -1)  # [B, T, F, H]

        # Avg pooling
        group_mean = group_output.mean(2).view(B * T, -1)  # [B * T, H]

        # Concate and transform
        group_output = group_output.view(B * T, G, -1)  # [B * T, F, H]
        group_mean = self.mean_linear(group_mean).unsqueeze(1).expand_as(group_output).contiguous()  # [B * T, F, H]
        group_output = torch.cat([group_output, group_mean], 2)  # [B * T, F, 2H]
        group_output = self.output_linear(group_output.view(-1, group_output.shape[-1]))  # [B * T * F, F_s]
        group_output = group_output.view(B, T, G, -1).permute(0, 2, 3, 1).contiguous()  # [B, F, F_s, T]
        group_output = self.norm(group_output.view(B * G, N, T))  # [B * F, F_s, T]
        output = input + group_output.view(input.shape)  # [B, F, F_s, T]

        return output


class SIL_Block(nn.Module):
    def __init__(
            self,
            input_size,
            tac_hidden_size,
            lstm_hidden_size,
            bidirectional,
            sequence_model="GRU"
    ):
        super().__init__()
        self.SubInter = subband_interaction(input_size=input_size, hidden_size=tac_hidden_size)
        self.sequence_model_type = sequence_model
        if self.sequence_model_type == "LSTM":
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1,
                               batch_first=True, bidirectional=bidirectional)
        elif self.sequence_model_type == "GRU":
            self.RNN = nn.GRU(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1,
                              batch_first=True, bidirectional=bidirectional)
        self.norm = nn.GroupNorm(1, lstm_hidden_size)

    def forward(self, x):
        """
        Args:
            [B, F, N(H), T]
        Returns:
            [B, F, N(H), T]
        """
        # SubInter processing
        B, G, N, T = x.size()
        # x = x.reshape(B/nums_group, nums_group, N, T)
        x = self.SubInter(x)

        # RNN processing
        self.RNN.flatten_parameters()
        x = x.reshape(B * G, N, T)
        x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
        rnn_o, _ = self.RNN(x)
        o = self.norm(rnn_o.permute(0, 2, 1))  # [B, T, H] => [B, H, T]
        _, H, _ = o.size()
        return o.reshape(B, G, H, T)


class stacked_SIL_blocks_SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            norm=None,
            sequence_model="GRU",
            output_activate_function="Tanh",
            middle_tac_hidden_times=0.66
    ):
        """
        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            norm: 使用的normalizaion
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        # self.norm = norm
        self.sequence_model_type = sequence_model
        self.num_layers = num_layers
        if self.sequence_model_type == "LSTM":
            self.sequence_list = nn.ModuleList()
            first_SIL = SIL_Block(input_size=input_size, tac_hidden_size=3 * input_size,
                                  lstm_hidden_size=hidden_size, bidirectional=bidirectional,
                                  sequence_model=self.sequence_model_type)
            self.sequence_list.append(first_SIL)
            for i in range(1, self.num_layers):
                self.sequence_list.append(
                    SIL_Block(input_size=hidden_size, tac_hidden_size=int(middle_tac_hidden_times * hidden_size),
                              lstm_hidden_size=hidden_size, bidirectional=bidirectional,
                              sequence_model=self.sequence_model_type))

        elif self.sequence_model_type == "GRU":
            self.sequence_list = nn.ModuleList()
            first_SIL = SIL_Block(input_size=input_size, tac_hidden_size=3 * input_size,
                                  lstm_hidden_size=hidden_size, bidirectional=bidirectional,
                                  sequence_model=self.sequence_model_type)
            self.sequence_list.append(first_SIL)
            for i in range(1, self.num_layers):
                self.sequence_list.append(
                    SIL_Block(input_size=hidden_size, tac_hidden_size=int(middle_tac_hidden_times * hidden_size),
                              lstm_hidden_size=hidden_size, bidirectional=bidirectional,
                              sequence_model=self.sequence_model_type))

        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        if self.sequence_model_type == "LSTM" or self.sequence_model_type == "GRU":
            # Fully connected layer
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, G=F, N, T]
        Returns:
            [B, F, T]
        """

        # 经过连续的SIL block
        for SIL_block in self.sequence_list:
            x = SIL_block(x)

        # 修改o的shape
        B, G, H, T = x.size()
        x = x.reshape(B * G, H, T)
        x = x.permute(0, 2, 1).contiguous()

        o = self.fc_output_layer(x)
        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        return o
