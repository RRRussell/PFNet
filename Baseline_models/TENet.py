import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layer import DeGINConv,DenseGraphConv

class Model(nn.Module):

    def __init__(self, args,data):
        super(Model,self).__init__()
        self.use_cuda = args.cuda

        A = np.loadtxt(args.A)
        A = np.array(A,dtype=np.float32)
        A = A/np.sum(A,0)
        A_new = np.zeros((args.batch_size,args.n_e,args.n_e),dtype=np.float32)
        for i in range(args.batch_size):
            A_new[i,:,:]=A

        self.A = torch.from_numpy(A_new)#.cuda()
        self.n_e=args.n_e
        self.decoder = args.decoder

        self.conv1=nn.Conv2d(1, args.channel_size, kernel_size = (1,args.k_size[0]),stride=1)
        self.conv2=nn.Conv2d(1, args.channel_size, kernel_size = (1,args.k_size[1]),stride=1)
        self.conv3=nn.Conv2d(1, args.channel_size, kernel_size = (1,args.k_size[2]),stride=1)

        d = (len(args.k_size)*(args.window) -sum(args.k_size)+ len(args.k_size))*args.channel_size
        self.BATCH_SIZE=args.batch_size
        self.dropout = args.dropout
        self.hw = args.highway_window

        if self.decoder == 'GNN':
        # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(d, args.hid1)
            self.gnn2 = DenseGraphConv(args.hid1, args.hid2)
            self.gnn3 = DenseGraphConv(args.hid2, 1)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        if self.decoder == 'GIN':
            ginnn = nn.Sequential(
                nn.Linear(d,args.hid1),
                # nn.ReLU(True),
                # nn.Linear(args.hid1, args.hid2),
                nn.ReLU(True),
                nn.Linear(args.hid1,1),
                nn.ReLU(True)
            )
            self.gin = DeGINConv(ginnn)

    def forward(self,x):
        # print("x:",x.shape)
        c=x.permute(0,2,1)
        # print("c:",c.shape)
        c=c.unsqueeze(1)
        # print("c:",c.shape)

        a1=self.conv1(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        # print("a1:",a1.shape)
        a2=self.conv2(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        # print("a2:",a2.shape)
        a3=self.conv3(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        # print("a3:",a3.shape)
        x_conv = F.relu(torch.cat([a1, a2, a3], 2))
        # print("x_conv:",x_conv.shape)

        if self.decoder == 'GNN':
            x1 = F.relu(self.gnn1(x_conv,self.A))
            # print("x1",x1.shape)
            x2 = F.relu(self.gnn2(x1,self.A))
            # print("x2",x2.shape)
            x3 = self.gnn3(x2,self.A)
            # print("x3",x3.shape)
            x3 = x3.squeeze()
            # print("x3",x3.shape)

        if self.decoder == 'GIN':
            x3 = F.relu(self.gin(x_conv, self.A))
            x3 = x3.squeeze()

        if self.hw>0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1)
            z = self.highway(z)
            z = z.squeeze(2)
            x3 = x3 + z

        return x3
