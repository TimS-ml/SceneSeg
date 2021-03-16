import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import pdb


class Cos(nn.Module):
    def __init__(self, cfg):
        super(Cos, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
        self.conv1 = nn.Conv2d(1, self.channel,
                               kernel_size=(self.shot_num // 2, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num // 2] * 2, dim=2)
        # batch_size*seq_len, 1, [self.shot_num//2], feat_dim
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size, channel
        return x


class BNet(nn.Module):
    def __init__(self, cfg):
        super(BNet, self).__init__()
        self.shot_num = cfg.shot_num  # 4
        self.channel = cfg.model.sim_channel  # 512
        # Conv: in channel=1, out channel=512, kernel_size
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = Cos(cfg)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        # B_r
        context = x.view(x.shape[0] * x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)  # batch_size*seq_len, 512, 1, feat_dim
        context = self.max3d(context)  # batch_size*seq_len, 1, 1, feat_dim
        context = context.squeeze()
        # B_d
        sim = self.cos(x)  # split and cosine similarity
        bound = torch.cat((context, sim), dim=1)
        return bound


class BNet_light(nn.Module):
    def __init__(self, cfg):
        super(BNet_light, self).__init__()
        self.cos = Cos(cfg)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        # B_d
        sim = self.cos(x)  # split and cosine similarity
        bound = torch.cat((sim, sim), dim=1)
        return bound


class BNet_aud_light(nn.Module):
    def __init__(self, cfg):
        super(BNet_aud_light, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
        # for cosine_similarity, kernel_size same as Cos's conv1
        self.conv2 = nn.Conv2d(1, self.channel,
                               kernel_size=(cfg.shot_num // 2, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))

    # (257, 90) is the output size of stft
    # old: [batch_size, seq_len, shot_num, 257, 90]
    def forward(self, x):  # x: [batch_size, seq_len, shot_num, 512, 1]
        # B_d
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num // 2] * 2, dim=2)
        part1 = self.conv2(part1).squeeze()
        part2 = self.conv2(part2).squeeze()
        sim = F.cosine_similarity(part1, part2, dim=2)
        bound = sim
        return bound


class LGSSone(nn.Module):
    def __init__(self, cfg, mode="place"):
        super(LGSSone, self).__init__()
        self.seq_len = cfg.seq_len
        self.num_layers = 1
        self.lstm_hidden_size = cfg.model.lstm_hidden_size
        if mode == "place":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.place_feat_dim + cfg.model.sim_channel)
        elif mode == "cast":
            self.bnet = BNet_light(cfg)
            self.input_dim = (cfg.model.cast_feat_dim + cfg.model.sim_channel)
        elif mode == "act":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.act_feat_dim + cfg.model.sim_channel)
        elif mode == "aud":
            self.bnet = BNet_aud_light(cfg)
            self.input_dim = cfg.model.aud_feat_dim
        else:
            pass
        self.lstm = nn.LSTM(input_size=self.input_dim,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=cfg.model.bidirectional)

        if cfg.model.bidirectional:
            self.fc1 = nn.Linear(self.lstm_hidden_size * 2, 100)
        else:
            self.fc1 = nn.Linear(self.lstm_hidden_size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.bnet(x)
        x = x.view(-1, self.seq_len, x.shape[-1])
        # torch.Size([128, seq_len, 3*channel])
        self.lstm.flatten_parameters()
        out, (_, _) = self.lstm(x, None)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(-1, 2)
        return out


# Simple bagging
class LGSS(nn.Module):
    def __init__(self, cfg):
        super(LGSS, self).__init__()
        self.seq_len = cfg.seq_len
        self.mode = cfg.dataset.mode
        self.num_layers = 1
        self.lstm_hidden_size = cfg.model.lstm_hidden_size
        self.ratio = cfg.model.ratio
        if 'place' in self.mode:
            self.bnet_place = LGSSone(cfg, "place")
        if 'cast' in self.mode:
            self.bnet_cast = LGSSone(cfg, "cast")
        if 'act' in self.mode:
            self.bnet_act = LGSSone(cfg, "act")
        if 'aud' in self.mode:
            self.bnet_aud = LGSSone(cfg, "aud")

    def forward(self, place_feat, cast_feat, act_feat, aud_feat):
        out = 0
        if 'place' in self.mode:
            place_bound = self.bnet_place(place_feat)
            out += self.ratio[0] * place_bound
        if 'cast' in self.mode:
            cast_bound = self.bnet_cast(cast_feat)
            out += self.ratio[1] * cast_bound
        if 'act' in self.mode:
            act_bound = self.bnet_act(act_feat)
            out += self.ratio[2] * act_bound
        if 'aud' in self.mode:
            aud_bound = self.bnet_aud(aud_feat)
            out += self.ratio[3] * aud_bound
        return out


if __name__ == '__main__':
    from mmcv import Config
    cfg = Config.fromfile("./config/test/all.py")
    model = LGSS(cfg)
    place_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 2048)
    cast_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 512)
    act_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 512)
    aud_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 257, 90)
    output = model(place_feat, cast_feat, act_feat, aud_feat)
    print(cfg.batch_size)
    print(output.data.size())
