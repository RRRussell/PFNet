import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip

        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if (self.skip > 0):
            self.pt = (self.P - self.Ck) / self.skip
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):

        # debug = True
        debug = False
        if debug:
            print("x:",x.shape)

        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        if debug:
            print("c:",c.shape)
        c = F.relu(self.conv1(c))
        if debug:
            print("c:",c.shape)
        c = self.dropout(c)
        if debug:
            print("c:",c.shape)
        c = torch.squeeze(c, 3)
        if debug:
            print("c:",c.shape)

        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        if debug:
            print("r:",r.shape)
        _, r = self.GRU1(r)
        if debug:
            print("r:",r.shape)

        r = self.dropout(torch.squeeze(r, 0))
        if debug:
            print("r:",r.shape)

        # skip-rnn

        if (self.skip > 0):
            self.pt=int(self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            if debug:
                print("s:",s.shape)

            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            if debug:
                print("s:",s.shape)
            s = s.permute(2, 0, 3, 1).contiguous()
            if debug:
                print("s:",s.shape)
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            if debug:
                print("s:",s.shape)
            _, s = self.GRUskip(s)
            if debug:
                print("s:",s.shape)
            s = s.view(batch_size, self.skip * self.hidS)
            if debug:
                print("s:",s.shape)
            s = self.dropout(s)
            if debug:
                print("s:",s.shape)
            r = torch.cat((r, s), 1)
            if debug:
                print("r:",r.shape)

        res = self.linear1(r)
        if debug:
            print("res:",res.shape)

        # highway
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res
