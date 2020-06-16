# -*-coding:utf-8 -*-

import torch
import torch.nn as nn

## use_attention: use attention module or not
## share_conv: share conv of x1 and x2 or not
class SymptomNet(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channel_1, out_channel_2, max_len_x1, max_len_x2, use_attention=True, share_conv=False):
        super(SymptomNet, self).__init__()
        self.use_attention = use_attention
        self.share_conv = share_conv

        self.word_embedding = nn.Embedding(vocab_size, embed_size)

        self.x1_conv = SymptomConvNet(embed_size, out_channel_1)

        if not self.share_conv:
            self.x2_conv = SymptomConvNet(embed_size, out_channel_1)

        if self.use_attention:
            self.label_embedding = nn.Embedding(24, 300)
            self.x1_att = SymptomAttNet(out_channel_1)

        self.conv = nn.Sequential(
            nn.Conv2d(4, out_channel_2, 3, padding=1), nn.ReLU(), 
            nn.MaxPool2d(3, stride=3), nn.Flatten()
            )
        self.fc = nn.Sequential(nn.Linear(out_channel_2*(max_len_x1//3)*(max_len_x2//3), 1), nn.Sigmoid())

    
    def forward(self, input_x1, input_x2, input_label):
        x1_conv1, x1_conv2 = self.x1_conv(self.word_embedding(input_x1))

        if not self.share_conv:
            x2_conv1, x2_conv2 = self.x2_conv(self.word_embedding(input_x2))
        else:
            x2_conv1, x2_conv2 = self.x1_conv(self.word_embedding(input_x2))

        if self.use_attention:
            label_embed = self.label_embedding(input_label)
            x1_conv1 = self.x1_att(label_embed, x1_conv1)
            x1_conv2 = self.x1_att(label_embed, x1_conv2)

        sim_matrix_1 = torch.bmm(x1_conv1, x2_conv1.permute(0, 2, 1))
        sim_matrix_2 = torch.bmm(x1_conv1, x2_conv2.permute(0, 2, 1))

        sim_matrix_3 = torch.bmm(x1_conv2, x2_conv1.permute(0, 2, 1))
        sim_matrix_4 = torch.bmm(x1_conv2, x2_conv2.permute(0, 2, 1))

        cube_data = torch.cat((
            sim_matrix_1.unsqueeze(3), sim_matrix_2.unsqueeze(3),
            sim_matrix_3.unsqueeze(3), sim_matrix_4.unsqueeze(3),), dim=-1)
        
        out = self.fc(self.conv(cube_data.permute(0, 3, 1, 2)))

        return out


class SymptomConvNet(nn.Module): # bi-channel conv
    def __init__(self, embed_size, out_channel_1):
        super(SymptomConvNet, self).__init__()
        # print(str(out_channel_1))
        self.conv1 = nn.Sequential(nn.Conv1d(embed_size, out_channel_1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(embed_size, out_channel_1, 2, padding=1), nn.ReLU())
    
    def forward(self, x_embed):
        x_conv1 = self.conv1(x_embed.permute(0,2,1)).permute(0,2,1)
        x_conv2 = self.conv2(x_embed.permute(0,2,1))[:, :, :-1].permute(0,2,1)

        return x_conv1, x_conv2


class SymptomAttNet(nn.Module): # Attention module integrating the HSI
    def __init__(self, out_channel_1):
        super(SymptomAttNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(300, out_channel_1), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv1d(out_channel_1, 1, 3, padding=1), nn.PReLU(), nn.Softmax(dim=2))
        self.norm = nn.LayerNorm((64, 50))

    def forward(self, label_embed, x_conv):
        label_embed = self.fc(label_embed.squeeze(1)).unsqueeze(1)

        x_att = torch.mul(x_conv, label_embed)
        x_att = self.conv(x_att.permute(0, 2, 1))
        x_att = torch.mul(x_conv.permute(0, 2, 1), x_att).permute(0, 2, 1)
        x_att = self.norm(x_conv + x_att)

        return x_att
