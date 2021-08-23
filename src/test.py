# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp
from scipy.stats.stats import pearsonr
from models import *
from data import *

import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='japan', help="Dataset string")
ap.add_argument('--dataorigin', type=str, default='sem3d', help="Dataset origin")
ap.add_argument('--sim_mat', type=str, default='japan-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=32, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.6, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.2, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='cola_gnn', choices=['cola_gnn','CNNRNN_Res','RNN','AR','ARMA','VAR','GAR','SelfAttnRNN','lstnet','stgcn','dcrnn'], help='')
ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=False,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=10, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=3,  help='choose gpu 0-10')
ap.add_argument('--lambda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=200, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')

args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dcrnn_model import *
from matplotlib import pyplot as plt
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available() 
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)

if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    shutil.rmtree(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

data_loader = DataBasicLoader(args)


if args.model == 'CNNRNN_Res':
    model = CNNRNN_Res(args, data_loader)  
elif args.model == 'RNN':
    model = RNN(args, data_loader)
elif args.model == 'AR':
    model = AR(args, data_loader)
elif args.model == 'ARMA':
    model = ARMA(args, data_loader)
elif args.model == 'VAR':
    model = VAR(args, data_loader)
elif args.model == 'GAR':
    model = GAR(args, data_loader)
elif args.model == 'SelfAttnRNN':
    model = SelfAttnRNN(args, data_loader)
elif args.model == 'lstnet':
    model = LSTNet(args, data_loader)      
elif args.model == 'stgcn':
    model = STGCN(args, data_loader, data_loader.m, 1, args.window, 1)  
elif args.model == 'dcrnn':
    model = DCRNNModel(args, data_loader)   
elif args.model == 'cola_gnn':
    model = cola_gnn(args, data_loader)        
else: 
    raise LookupError('can not find the model')
 
logger.info('model %s', model)
if args.cuda:
    model.cuda()

def evaluate(data_loader, data, tag='val'):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    k=0
    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        output,_  = model(X)
        fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        # import pdb
        # pdb.set_trace()
        ax.plot(np.linspace(1,Y.shape[0],Y.shape[0]),Y[:,0].detach().cpu().numpy(),label=r'$y(t)$')
        ax.plot(np.linspace(1,Y.shape[0],Y.shape[0]),output[:,0].detach().cpu().numpy(),
            label=r'$\hat{y}(t)$')
        
        cm = plt.cm.get_cmap('RdYlBu')
        ax.set_xlim([0.0,Y.shape[0]+1])
        ax.set_xticks(np.linspace(0,Y.shape[0]+1,Y.shape[0]+2))
        ax.set_xlabel(r'$t$ [s]')
        ax.set_ylabel(r'$y$ [1]')
        ax.set_title(r'Window size {:>d} - Horizon {:>d}'.format(args.window,args.horizon))
        plt.legend(frameon=False)
        fig.savefig('test_{:>d}.png'.format(k),format="png", bbox_inches='tight')
        k +=1
        plt.close()
        loss_train = F.l1_loss(output, Y) # mse_loss
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m);

        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx) # [n_samples, 47] 
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  #(#n_samples, 47)
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) # mean of 47
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places 
    pcc_tmp = []
    for k in range(data_loader.m):
        pcc_tmp.append(pearsonr(y_true_states[:,k],y_pred_states[:,k])[0])
    pcc_states = np.mean(np.array(pcc_tmp)) 
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states,(-1))
    y_pred = np.reshape(y_pred_states,(-1))
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pcc = pearsonr(y_true,y_pred)[0]
    r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples), mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae


# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test')

