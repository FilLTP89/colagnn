import sys
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from parse_h5_traces import ParseSEM3DH5Traces

def generate_quarterly_data(data):
    quarterly_data = -999.0*np.ones((data.shape[0],4*data.shape[1]),dtype=np.float32)
    ddata = np.diff(data,n=1,axis=1,prepend=data[:,0].reshape(-1,1))
    quarterly_data[:,0::4] = data
    quarterly_data[:,1::4] = data+0.25*ddata
    quarterly_data[:,2::4] = data+0.50*ddata
    quarterly_data[:,3::4] = data+0.75*ddata
    return quarterly_data

class DataBasicLoader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window
        self.h = args.horizon
        self.d = 0
        self.add_his_day = False
        self.dataorigin = args.dataorigin
        if "sem3d" in self.dataorigin.lower():
            options = {"wkd":"../data/traces","fmt":"h5",
                "nam":["all"],"var":"Veloc","rdr":["x"],
                "mon":[0],"plt":False}
            stream = ParseSEM3DH5Traces(**options)
            self.rawdat = self.rawdat = stream['all'].data['Veloc'].squeeze()
            # self.rawdat = stream['all'].data['Veloc'].T
            # self.rawdat = self.rawdat.reshape(-1,self.rawdat.shape[-1])
        elif "amit" in self.dataorigin.lower():
            df = pd.read_csv("../data/{}.txt".format(args.dataset),
                header=1,sep="\t",skiprows=0,index_col=1).iloc[:,1:].astype(np.float32)
            self.rawdat = generate_quarterly_data(df.astype(np.float32).values).T
        else:
            self.rawdat = np.loadtxt(open("../data/{}.txt".format(args.dataset)),
                delimiter=',')
        print('data shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)
 
        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape # n_sample, n_group
        print(self.n, self.m)
        self.scale = np.ones(self.m)

        self._pre_train(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        print('size of train/val/test sets',len(self.train[0]),len(self.val[0]),len(self.test[0]))
    
    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(open("../data/{}.txt".format(args.sim_mat)), delimiter=','))
        self.orig_adj = self.adj
        rowsum = 1. / torch.sqrt(self.adj.sum(dim=0))
        self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()
            self.orig_adj = self.orig_adj.cuda()

    def _pre_train(self, train, valid, test):
        self.train_set = train_set = range(self.P+self.h-1, train)
        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n)
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy() #199, 47
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_thold = np.mean(train_mx, 0)
        self.dat  = (self.rawdat  - self.min ) / (self.max  - self.min + 1e-12)
        print(self.dat.shape)
         
    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h)
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if (train == valid):
            self.val = self.test
 
    def _batchify(self, idx_set, horizon, useraw=False):

        n = len(idx_set)
        Y = torch.zeros((n, self.m))
        if self.add_his_day and not useraw:
            X = torch.zeros((n, self.P+1, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P

            if useraw: # for narmalization
                X[i,:self.P,:] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i,:] = torch.from_numpy(self.rawdat[idx_set[i], :])
            else:
                his_window = self.dat[start:end, :]
                if self.add_his_day:
                    if idx_set[i] > 51 : # at least 52
                        his_day = self.dat[idx_set[i]-52:idx_set[i]-51, :] #
                    else: # no history day data
                        his_day = np.zeros((1,self.m))

                    his_window = np.concatenate([his_day,his_window])
                    # print(his_window.shape,his_day.shape,idx_set[i],idx_set[i]-52,idx_set[i]-51)
                    X[i,:self.P+1,:] = torch.from_numpy(his_window) # size (window+1, m)
                else:
                    X[i,:self.P,:] = torch.from_numpy(his_window) # size (window, m)
                Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    # original
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:]
            Y = targets[excerpt,:]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)

            data = [model_inputs, Variable(Y)]
            yield data
            start_idx += batch_size
