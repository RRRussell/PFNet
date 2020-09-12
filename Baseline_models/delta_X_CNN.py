import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.window = args.window
        self.variables = data.m
        # print(self.variables)
        self.hw=args.highway_window
        # self.hw = 0
        self.conv1 = nn.Conv1d(self.variables, 32, kernel_size=3)
        self.activate1=F.relu
        self.conv2=nn.Conv1d(32,32,kernel_size=3)
        self.maxpool1=nn.MaxPool1d(kernel_size=2)
        self.conv3=nn.Conv1d(32,16,kernel_size=3)
        self.maxpool2=nn.MaxPool1d(kernel_size=2)
        self.linear1=nn.Linear(16*11,100)
        # self.linear1=nn.Linear(16*79,100)
        self.out=nn.Linear(100,self.variables)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw-1, 1)

        self.dropout = nn.Dropout(p=args.dropout)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        # debug = True
        debug = False
        if debug:
            print("************************")
            print("delta_X_CNN")
        c = x.permute(0,2,1).contiguous()
        if debug:
            print("before c:",c.shape)
        c=self.conv1(c)
        c=self.activate1(c)
        if debug:
            print("after conv1:",c.shape)
        c=self.conv2(c)
        c=self.activate1(c)
        if debug:
            print("after conv2",c.shape)
        c=self.maxpool1(c)
        if debug:
            print("after maxpool2:",c.shape)
        c=self.conv3(c)
        c=self.activate1(c)
        if debug:
            print("after conv3:",c.shape)
        c=c.view(c.size(0),c.size(1)*c.size(2))
        c=self.dropout(c)
        c=self.linear1(c)
        if debug:
            print("after linear1:",c.shape)
        c=self.dropout(c)
        out=self.out(c)
        if debug:
            print("after linear2:",out.shape)

        if (self.hw > 0):
            if debug:
                print("?????")

            z = x[:, -self.hw+1:, :]
            # print("????",z.shape)
            if debug:
                print("z",z.shape)
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw-1)
            if debug:
                print("z",z.shape)
            z = self.highway(z)
            if debug:
                print("z",z.shape)
            z = z.view(-1, self.variables)
            if debug:
                print("z",z.shape)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out

