import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 2, triplet_loss = False):
        self.cuda = cuda
        self.P = window
        self.h = horizon

        # self.triplet_loss = True
        self.triplet_loss = triplet_loss
        if self.triplet_loss:
            self.h -= 1

        fin = open(file_name)
        self.rawdat = np.loadtxt(fin,delimiter=',')
        # self.rawdat = self.rawdat[:,0:20]
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        # print("n",self.n)
        # print("m",self.m)
        self.normalize = 1
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train+valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
            
        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        #normalized by the maximum value of entire matrix.
       
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
            
        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))
            
        
    def _split(self, train, valid, test):
        
        train_set = range(self.P+self.h-1, train)
        valid_set = range(train, valid)
        # if self.triplet_loss:
        #     test_set = range(valid, self.n-1)
        # else:    
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
        
        
    def _batchify(self, idx_set, horizon):
        
        n = len(idx_set)
        X = torch.zeros((n,self.P,self.m))
        Y = torch.zeros((n,self.m))

        # x = torch.zeros((n,self.m))
        delta_X = torch.zeros((n,self.P-1,self.m))
        delta_Y = torch.zeros((n,self.m))
        Y_1 = torch.zeros((n,self.m))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            # if i==0:
            #     print("start",start)
            #     print("end",end)
            #     print("idx_set[i]",idx_set[i])


            if self.triplet_loss:
                # x[i,:] = X[i,-1,]
                X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
                Y[i, :] = torch.from_numpy(self.dat[idx_set[i]-1, :])
                Y_1[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
                temp_a = X[i,1:,]
                temp_b = X[i,0:-1,]
                delta_X[i,:,:] = temp_a - temp_b
                delta_Y[i,:] = Y_1[i,:] - Y[i,:]
                # k=delta_Y[i,:]
                # if(k.sum()==0):
                #     dx=delta_X[i,:,:]
                #     dy=Y_1[i, :]
                #     dyy=Y[i,:]
                #     kk=idx_set[i]
                #     data0 = self.dat[idx_set[i]-2:idx_set[i]+2, :]
                #     # data1 = self.dat[idx_set[i], :]
                #     print("?")
            else:
                X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
                Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
            # if i == 0:
            #     print("X:",X[i,:,:])
            #     print("Y:",Y[i,:])
            #     print("temp_a:",temp_a)
            #     print("temp_b:",temp_b)
            #     print("delta_X:",delta_X)
            # print('Y',self.dat[idx_set[i], :])
        # print("X:",X.shape)
        # print("Y:",Y.shape)
        # print("delta_X:",delta_X.shape)
        # print("delta_Y:",delta_Y.shape)
        # print("Y_1:",Y_1.shape)
        
        if self.triplet_loss:
            return [X, Y, delta_X, delta_Y, Y_1]
        else:
            return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt] 
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()  
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    def triplet_get_batches(self, inputs, targets, delta_inputs, delta_targets, targets_1, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt] 
            Y = targets[excerpt]
            delta_X = delta_inputs[excerpt]
            delta_Y = delta_targets[excerpt]
            Y_1 = targets_1[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()  
                delta_X = delta_X.cuda()
                delta_Y = delta_Y.cuda()
                Y_1 = Y_1.cuda()
            yield Variable(X), Variable(Y), Variable(delta_X), Variable(delta_Y), Variable(Y_1)
            start_idx += batch_size


class STS_Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_train, file_test, cuda):
        self.cuda = cuda
        # self.P = window
        # self.h = horizon
        self.x_train,self.y_train = self.loaddata(file_train)
        self.x_test,self.y_test = self.loaddata((file_test))
        self.num_nodes = len(self.x_train.shape[0])+len(self.x_test.shape[0])
        self.num_class = len(np.unique(self.y_test))
        self.batch_size = min(self.x_train.shape[0] / 10, 16)

        x_train_mean = self.x_train.mean()
        x_train_std = self.x_train.std()
        self.x_train = (self.x_train - x_train_mean) / (x_train_std)
        self.x_test = (self.x_test - x_train_mean) / (x_train_std)

        self.y_train = (self.y_train - self.y_train.min()) / (self.y_train.max() - self.y_train.min()) * (self.num_class - 1)
        self.y_test = (self.y_test - self.y_test.min()) / (self.y_test.max() - self.y_test.min()) * (self.num_class - 1)

    def loaddata(self,filename):
        data = np.loadtxt(filename, delimiter=',')
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y

    def get_batches(self, inputs, targets, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        end_idx = length
        excerpt = index[start_idx:end_idx]
        X = inputs[excerpt]
        Y = targets[excerpt]
        if (self.cuda):
            X = X.cuda()
            Y = Y.cuda()
        return Variable(X), Variable(Y)
